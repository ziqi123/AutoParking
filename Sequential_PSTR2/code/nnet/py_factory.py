import os
import torch
import importlib
import torch.nn as nn

from config import system_configs
from models.py_utils.data_parallel import DataParallel

torch.manual_seed(317)


class Network(nn.Module):
    def __init__(self, model, loss):
        super(Network, self).__init__()

        self.model = model
        self.loss = loss

    def forward(self, iteration, save, viz_split,
                xs, ys, use_indices=True, **kwargs):

        srcs = self.model(*xs, **kwargs)

        loss = self.loss(iteration,
                         save,
                         viz_split,
                         srcs,
                         ys, use_indices=use_indices,
                         **kwargs)
        return loss

# for model backward compatibility
# previously model was wrapped by DataParallel module


class DummyModule(nn.Module):
    def __init__(self, model):
        super(DummyModule, self).__init__()
        self.module = model

    def forward(self, *xs, **kwargs):
        return self.module(*xs, **kwargs)


class NetworkFactory(object):
    def __init__(self, db, flag=False, freeze=False, params=None):
        super(NetworkFactory, self).__init__()

        module_file = "models.{}".format(system_configs.snapshot_name)
        nnet_module = importlib.import_module(module_file)

        self.model = DummyModule(nnet_module.model(
            db, flag=flag, freeze=freeze, params=params))
        self.loss = nnet_module.loss(params=params)
        self.network = Network(self.model, self.loss)
        self.network = DataParallel(
            self.network, chunk_sizes=system_configs.chunk_sizes)
        self.flag = flag

        # Count total parameters
        total_params = 0
        for params in self.model.parameters():
            num_params = 1
            for x in params.size():
                num_params *= x
            total_params += num_params
        print("total parameters: {}".format(total_params))  # 11685570

        if system_configs.opt_algo == "adam":
            self.optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, self.model.parameters())
            )
        elif system_configs.opt_algo == "sgd":
            self.optimizer = torch.optim.SGD(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                momentum=0.9, weight_decay=0.0001
            )
        elif system_configs.opt_algo == 'adamW':
            self.optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                lr=system_configs.learning_rate,
                weight_decay=1e-4
            )
        else:
            raise ValueError("unknown optimizer")

    def cuda(self):
        self.model.cuda()

    def train_mode(self):
        self.network.train()

    def eval_mode(self):
        self.network.eval()

    def train(self,
              iteration,
              save,
              viz_split,
              xs,
              ys,
              **kwargs):
        # train function
        xs = [x.cuda(non_blocking=True) for x in xs]
        ys = [y.cuda(non_blocking=True) for y in ys]
        self.optimizer.zero_grad()
        loss_kp = self.network(iteration,
                               save,
                               viz_split,
                               xs,
                               ys)
        set_loss = loss_kp[0]
        loss_dict = loss_kp[1]
        set_loss = set_loss.mean()
        set_loss.backward()
        self.optimizer.step()

        return set_loss, loss_dict

    def inference(self,
                  iteration,
                  save,
                  viz_split,
                  xs,
                  ys,
                  **kwargs):
        # don't use matching results to write intermediate results
        with torch.no_grad():
            # print('inference!!')
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]
            loss_kp = self.network(iteration,
                                   save,
                                   viz_split,
                                   xs,
                                   ys, use_indices=False)
            set_loss = loss_kp[0]
            loss_dict = loss_kp[1]

            return set_loss, loss_dict

    def validate(self,
                 iteration,
                 save,
                 viz_split,
                 xs,
                 ys,
                 **kwargs):
        # use matching results to write intermediate results
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            ys = [y.cuda(non_blocking=True) for y in ys]
            loss_kp = self.network(iteration,
                                   save,
                                   viz_split,
                                   xs,
                                   ys)
            set_loss = loss_kp[0]
            loss_dict = loss_kp[1]

            return set_loss, loss_dict

    def test(self, xs, **kwargs):
        with torch.no_grad():
            xs = [x.cuda(non_blocking=True) for x in xs]
            return self.model(*xs, **kwargs)

    def set_lr(self, lr):
        print("setting learning rate to: {}".format(lr))
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def load_pretrained_params(self, pretrained_model):
        print("loading from {}".format(pretrained_model))
        with open(pretrained_model, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()

            if len(params) != len(model_dict):
                pretrained_dict = {
                    'module.' + k: v for k, v in params.items() if 'module.' + k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)

    def load_params(self, iteration, is_bbox_only=False):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("loading [J] model from {}".format(cache_file))

        with open(cache_file, "rb") as f:
            params = torch.load(f)
            model_dict = self.model.state_dict()
            if len(params) != len(model_dict):
                pretrained_dict = {k: v for k,
                                   v in params.items() if k in model_dict}
            else:
                pretrained_dict = params
            model_dict.update(pretrained_dict)

            self.model.load_state_dict(model_dict)

    def save_params(self, iteration):
        cache_file = system_configs.snapshot_file.format(iteration)
        print("saving model to {}".format(cache_file))
        with open(cache_file, "wb") as f:
            params = self.model.state_dict()
            torch.save(params, f)
