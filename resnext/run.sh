#!/usr/bin/env sh
export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 -u train_se_resnext_w_d.py --lr 0.001 --data-train  /home/jzhengas/Jason/img_data/data/all_data/new_all  --data-val /home/jzhengas/Jason/img_data/data/low_val/low_val_all.rec --data-type imagenet --num-classes=4934 --num-examples=1000000 --depth 50 --batch-size 64 --model-load-epoch=2 --drop-out 0.0 --gpus=0 --num-epoch=100 --retrain  --freeze 0  --finetune 0 --model-name save_model

