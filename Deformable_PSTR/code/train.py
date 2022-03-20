#!/usr/bin/env python
from distutils.command.config import config
from db.datasets import datasets
from torch.multiprocessing import Process, Queue, Pool
from nnet.py_factory import NetworkFactory
from config import system_configs
from utils import stdout_to_tqdm
from tqdm import tqdm
import traceback
import threading
import importlib
import argparse
import queue
import numpy as np
import torch
import json
import cv2
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train PoseAttNet")
    parser.add_argument("--cfg_file", help="config file",
                        type=str, default='sfsp')
    parser.add_argument("--mkdemo", help="demo file", default=0, type=bool)
    parser.add_argument("--test_pstr", help="test pstr", default=0, type=bool)
    parser.add_argument("--test", help="test", default=0, type=bool)
    parser.add_argument("--iter", dest="start_iter",
                        help="train at iteration i",
                        default=0, type=int)
    parser.add_argument("--threads", dest="threads", default=4, type=int)
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args()
    return args


def make_video(tmp_dir, video_dir):
    fps = 30
    size = (384, 384)
    videoWriter = cv2.VideoWriter(
        video_dir, cv2.VideoWriter_fourcc('I', '4', '2', '0'), fps, size)
    for img in range(len(os.listdir(tmp_dir))):
        img = '{}_{}.jpg'.format(tmp_dir.split('/')[-1], img)
        img1 = cv2.imread(os.path.join(tmp_dir, img))
        videoWriter.write(img1)
    videoWriter.release()


def prefetch_data(db, queue, sample_data, data_aug):
    ind = 0
    print("start prefetching data...")
    np.random.seed(os.getpid())
    while True:
        try:
            data, ind = sample_data(db, ind, data_aug=data_aug)
            queue.put(data)
        except Exception as e:
            traceback.print_exc()
            raise e


def pin_memory(data_queue, pinned_data_queue, sema):
    while True:
        data = data_queue.get()

        data["xs"] = [x.pin_memory() for x in data["xs"]]
        data["ys"] = [y.pin_memory() for y in data["ys"]]

        pinned_data_queue.put(data)

        if sema.acquire(blocking=False):
            return


def init_parallel_jobs(dbs, queue, fn, data_aug):
    tasks = [Process(target=prefetch_data, args=(
        db, queue, fn, data_aug)) for db in dbs]
    for task in tasks:
        task.daemon = True
        task.start()
    return tasks


def train(training_dbs, validation_db, demo_db, start_iter=0, freeze=False):
    learning_rate = system_configs.learning_rate
    max_iteration = system_configs.max_iter
    pretrained_model = system_configs.pretrain
    snapshot = system_configs.snapshot
    val_iter = system_configs.val_iter
    infer_iter = system_configs.infer_iter
    display = system_configs.display
    decay_rate = system_configs.decay_rate
    stepsize = system_configs.stepsize

    # queues storing data for training
    training_queue = Queue(system_configs.prefetch_size)  # 5
    validation_queue = Queue(5)
    demo_queue = Queue(5)

    # queues storing pinned data for training
    pinned_training_queue = queue.Queue(system_configs.prefetch_size)  # 5
    pinned_validation_queue = queue.Queue(5)
    pinned_demo_queue = queue.Queue(5)

    # load data sampling function
    print("building model...")
    nnet = NetworkFactory(
        training_dbs[0], flag=True, freeze=freeze, params=net_params)
    # test
    if args.test_pstr:
        test_dataset_dir = system_configs.test_data_dir
        test_show_dir = system_configs.test_data_show
        if not os.path.exists(test_dataset_dir):
            print('no demo dir')
            raise ValueError
        if not os.path.exists(os.path.join(test_show_dir)):
            os.makedirs(test_show_dir)
        test_iter = system_configs.test_ckpt
        test_threshold = system_configs.test_threshold
        threshold_pck = system_configs.threshold_pck
        nnet.load_params(test_iter)
        nnet.cuda()
        nnet.eval_mode()
        test_file = "test.test_pstr"
        testing = importlib.import_module(test_file).make_demo
        precision, recall = testing(
            nnet, test_dataset_dir, test_show_dir, threshold=test_threshold, threshold_pck=threshold_pck)
        print('precision:{}, recall:{}'.format(precision, recall))
        exit()
    # make demo
    if args.mkdemo:
        demo_dir = configs["demo"]["demo_dir"]
        tmp = configs["demo"]["tmp"]
        tmp = configs["demo"]["tmp"]
        annt_result_dir = configs["demo"]["annt_result_dir"]
        threshold_demo = configs["demo"]["threshold"]
        video_dir = configs["demo"]["video_dir"]
        if not os.path.exists(os.path.join(demo_dir)):
            print('no demo dir')
            raise ValueError
        print(os.path.exists(tmp))
        if not os.path.exists(tmp):
            os.makedirs(tmp)
        test_iter = system_configs.test_ckpt
        nnet.load_params(test_iter)
        nnet.cuda()
        nnet.eval_mode()
        test_file = "test.pslot"
        testing = importlib.import_module(test_file).make_demo
        testing(nnet, demo_dir, tmp, annt_result_dir, threshold=threshold_demo)
        make_video(tmp, video_dir)
        exit()
    # load pretrain model
    if pretrained_model is not None:
        if not os.path.exists(pretrained_model):
            raise ValueError("pretrained model does not exist")
        print("loading from pretrained model")
        nnet.load_pretrained_params(pretrained_model)
    data_file = "sample.{}".format(training_dbs[0].data)  # "sample.coco"
    sample_data = importlib.import_module(data_file).sample_data
    # allocating resources for parallel reading
    training_tasks = init_parallel_jobs(
        training_dbs, training_queue, sample_data, True)
    if val_iter:
        validation_tasks = init_parallel_jobs(
            [validation_db], validation_queue, sample_data, False)
    if infer_iter:
        demo_tasks = init_parallel_jobs(
            [demo_db], demo_queue, sample_data, False)

    training_pin_semaphore = threading.Semaphore()
    validation_pin_semaphore = threading.Semaphore()
    demo_pin_semaphore = threading.Semaphore()
    training_pin_semaphore.acquire()
    validation_pin_semaphore.acquire()
    demo_pin_semaphore.acquire()

    training_pin_args = (
        training_queue, pinned_training_queue, training_pin_semaphore)
    training_pin_thread = threading.Thread(
        target=pin_memory, args=training_pin_args)
    training_pin_thread.daemon = True
    training_pin_thread.start()

    validation_pin_args = (
        validation_queue, pinned_validation_queue, validation_pin_semaphore)
    validation_pin_thread = threading.Thread(
        target=pin_memory, args=validation_pin_args)
    validation_pin_thread.daemon = True
    validation_pin_thread.start()

    demo_pin_args = (demo_queue, pinned_demo_queue, demo_pin_semaphore)
    demo_pin_thread = threading.Thread(target=pin_memory, args=demo_pin_args)
    demo_pin_thread.daemon = True
    demo_pin_thread.start()

    if start_iter:
        learning_rate /= (decay_rate ** (start_iter // stepsize))
        nnet.load_params(start_iter)
        nnet.set_lr(learning_rate)
        print("training starts from iteration {} with learning_rate {}".format(
            start_iter + 1, learning_rate))
    else:
        nnet.set_lr(learning_rate)
    print("training start...")
    nnet.cuda()
    nnet.train_mode()

    with stdout_to_tqdm() as save_stdout:
        for iteration in tqdm(range(start_iter + 1, max_iteration + 1), file=save_stdout, ncols=67):
            training = pinned_training_queue.get(block=True)
            save = True if (display and iteration % display == 0) else False
            viz_split = 'train'
            (set_loss, loss_dict) \
                = nnet.train(iteration, save, viz_split, **training)
            if display and iteration % display == 0:
                print("iteration {}\nset loss:\t{}".format(
                    iteration, set_loss.item()))
                for k, v in loss_dict.items():
                    print("{}:\t{}".format(k, v.item()))
            del set_loss
            if infer_iter and iteration % infer_iter == 0:
                viz_split = 'infer'
                save = True
                inference = pinned_demo_queue.get(block=True)
                _, _ = nnet.inference(iteration, save, viz_split, **inference)
            if val_iter and validation_db.db_inds.size and iteration % val_iter == 0:
                viz_split = 'val'
                save = True
                validation = pinned_validation_queue.get(block=True)
                (val_set_loss, val_loss_dict) \
                    = nnet.validate(iteration, save, viz_split, **validation)
                print("iteration {}\nvalidation set loss:\t{}".format(
                    iteration, val_set_loss.item()))
                for k, v in val_loss_dict.items():
                    print("validation {}:\t{}".format(k, v.item()))
                nnet.train_mode()
            if iteration % snapshot == 0:
                nnet.save_params(iteration)
            if iteration % stepsize == 0:
                learning_rate /= decay_rate
                nnet.set_lr(learning_rate)

    # sending signal to kill the thread
    training_pin_semaphore.release()
    validation_pin_semaphore.release()
    demo_pin_semaphore.release()

    # terminating data fetching processes
    for training_task in training_tasks:
        training_task.terminate()
    for validation_task in validation_tasks:
        validation_task.terminate()
    for demo_task in demo_tasks:
        demo_task.terminate()


if __name__ == "__main__":
    args = parse_args()
    cfg_file = os.path.join(system_configs.config_dir, args.cfg_file + ".json")
    with open(cfg_file, "r") as f:
        configs = json.load(f)
    configs["system"]["snapshot_name"] = args.cfg_file
    system_configs.update_config(configs["system"])
    net_params = configs["net_params"]
    train_split = system_configs.train_split
    val_split = system_configs.val_split
    demo_split = system_configs.demo_split
    dataset = system_configs.dataset
    print("loading all datasets {}...".format(dataset))
    threads = args.threads  # 4 every 4 epoch shuffle the indices
    print("using {} threads".format(threads))
    training_dbs = [datasets[dataset](
        configs["db"], train_split) for _ in range(threads)]  # training datasets
    validation_db = datasets[dataset](
        configs["db"], val_split)  # validation datasets
    demo_db = datasets[dataset](configs["db"], demo_split)  # demo datasets
    print("len of training db: {}".format(len(training_dbs[0].db_inds)))
    print("len of testing db: {}".format(len(validation_db.db_inds)))
    print("freeze the pretrained network: {}".format(args.freeze))
    train(training_dbs, validation_db, demo_db,
          args.start_iter, args.freeze)  # 0
