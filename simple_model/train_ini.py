import pickle
import random
import numpy as np
from keras.models import *
from keras.layers import *
from keras import optimizers
from keras.utils.np_utils import to_categorical
from keras import regularizers
from keras.callbacks import LearningRateScheduler
import keras
import argparse

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
session = tf.Session(config=config)
KTF.set_session(session)


def load_from_raw_pickle(pickle_path='data/feats_val.pickle', allow_empty=False, threshold=None, drop=0.8):
    with open(pickle_path, 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='iso-8859-1')
    with open('/home/jzhengas/Jason/data/simple_model/dict.pickle', 'rb') as fin:
        data_dict = pickle.load(fin, encoding='iso-8859-1')
    x = []
    y = []
    val_x = []
    val_y = []
    for video_name in feats_dict:
        drop_cond = random.random() > drop
        feats = feats_dict[video_name]
        if not data_dict is None:
            if video_name in data_dict:        
                label = data_dict[video_name]
            else:
                label = 4935
        else:
            label = 4935
        if len(feats) == 0:
            continue
        label = label - 1
        feats.sort(key = lambda x: x[3], reverse=True)

        select_count = 0
        for feat in feats:
            [frame_num, bbox, det_score, qua_score, feat_arr] = feat
            if threshold is None or (qua_score > threshold[0] and qua_score < threshold[1]):
                if drop_cond:
                    val_x.append(feat_arr)
                    val_y.append(label)
                else:
                    x.append(feat_arr)
                    y.append(label)
                select_count += 1

        if select_count == 0 and not allow_empty:
            [frame_num, bbox, det_score, qua_score, feat_arr] = feats[int(len(feats)/2)]
            if drop_cond:
                val_x.append(feat_arr)
                val_y.append(label)
            else:
                x.append(feat_arr)
                y.append(label)

    return np.array(x), np.array(y), np.array(val_x), np.array(val_y)



def lr_scheduler(epoch, lr_base = 0.001):
    if epoch >= 1:
        lr_base = lr_base * 0.8

    print('lr: %f' % lr_base)
    return lr_base


def read_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='iso-8859-1')
    random.shuffle(feats_dict)
    x, y, det_score, face_score, video_name = zip(*feats_dict)
    x = np.array(x)
    y = np.array(y)
    video_name = np.array(video_name)
    return x, y


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--threshold', type=str, default='0,200', help='threshold')
    parser.add_argument('--save-model', default='../save_model/tmp_model/drop_model.hdf5', help='directory to save model.')
    parser.add_argument('--file-list', default='/home/jzhengas/Jason/data/simple_model/train.pickle,/home/jzhengas/Jason/data/simple_model/val.pickle', help='training data')
    parser.add_argument('--epoch', type=int, default=6, help='threshold')
    parser.add_argument('--seed', type=int, default=2018)
    parser.add_argument('--allow-empty', type=int, default=0)
    parser.add_argument('--drop', type=float, default=1.0)
    args = parser.parse_args()
    random.seed(args.seed)

    threshold = [int(d) for d in args.threshold.split(',')]


    train_x1, train_y1, val_x1, val_y1 = load_from_raw_pickle('/home/jzhengas/Jason/data/simple_model/pickle/feats_train.pickle', threshold=threshold, drop=args.drop, allow_empty=args.allow_empty)
    train_x2, train_y2, val_x2, val_y2 = load_from_raw_pickle('/home/jzhengas/Jason/data/simple_model/pickle/feats_val.pickle', threshold=threshold, drop=args.drop, allow_empty=args.allow_empty)
    train_x = np.concatenate([train_x1, train_x2])
    train_y = np.concatenate([train_y1, train_y2])
    val_x = np.concatenate([val_x1, val_x2])
    val_y = np.concatenate([val_y1, val_y2])

    val_data = (val_x, val_y)

    print('Finish loading data, train len {}'.format(len(train_x)))
    
    input_tensor = Input(shape=(512,))
    x = Dense(1024, activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(4935, activation='softmax')(x)
    model = Model(input_tensor, x)

    opt = optimizers.adam(lr=0.001)
    scheduler = LearningRateScheduler(lr_scheduler) 

    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(x=train_x, y=train_y, epochs=args.epoch, batch_size=256, validation_data=val_data, callbacks=[scheduler], shuffle=True)
    model.save(args.save_model)


