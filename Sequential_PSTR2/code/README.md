**PSTR**: Parking Slot Transformer
=======

* End-to-end predicting parking slots from a single image from Brids'-Eye-View(BEV).

## Model Zoo
The pretrained models are available in model_ckpt/nnet/sfsp

## Data Preparation
Directory structure of train dataset:
```
path/to/data_dir/
    --data
        --train
        --val
    --labels
        --train
        --val
```

Directory structure of test dataset:
```
path/to/test_data_dir/
    --data
    --annotation
```
## Set Environment

* Linux ubuntu 16.04
* Pytorch 1.7.1
* opencv-python
* scipy 1.5.4
* scikit-image 0.17.2
* numpy 1.19.5
* pycocotools
* matplotlib 3.3.4

## Training
* set "data_dir"  in config/sfsp.json. 

To train PSTR on a single gpu for 300000 iterations run:
```
python train.py sfsp
```

## testing
* set "test_data_dir" in config/sfsp.json

To test PSTR run:
```
python train.py sfsp --test_pstr 1
```
## Making demo
* set "demo_dir" in config/sfsp.json
* set video name -> "video_dir" in config/sfsp.json

To make a demo run:
```
python train.py sfsp --mkdemo 1
```





