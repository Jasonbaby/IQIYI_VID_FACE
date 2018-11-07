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
import os
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True   
session = tf.Session(config=config)
KTF.set_session(session)

random.seed(2018)


def get_data(pickle_path='', threshold=20, allow_empty=False, ratio=0):
    with open(pickle_path, 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='iso-8859-1')
    x = []
    name_list = []
    video_count = 0
    for video_name in feats_dict:
        feats = feats_dict[video_name]
        if len(feats) == 0:
            continue
        video_count += 1
        feats.sort(key = lambda x: x[3], reverse=True)

        if ratio > 0:
            end_pos = int(len(feats) / ratio) if int(len(feats) / ratio) >=1 else 1
            feats = feats[0: end_pos]
        select_feats = [f for f in feats if f[3] > threshold]

        if (len(select_feats) == 0) and not allow_empty:
            select_feats = [feats[0]]
            
        for feat in select_feats:
            [frame_num, bbox, det_score, qua_score, feat_arr] = feat
            x.append(feat_arr)
            name_list.append(video_name)
    print(len(x), len(name_list), video_count)
    d = list(zip(x, name_list))
    random.shuffle(d)
    x, name_list = zip(*d)
    return np.array(x), name_list

def read_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='iso-8859-1')
    x, y, det_score, face_score, video_name = zip(*feats_dict)
    x = np.array(x)
    video_name = np.array(video_name)
    return x, video_name, face_score


def get_result_dict(name_data, result):
    result_dict = {}
    for i in range(0, len(name_data)):
        tmp_name = name_data[i]
        if tmp_name not in result_dict:
            result_dict[tmp_name] = [result[i]]
        else:
            result_dict[tmp_name].append(result[i])
    all_result = []
    for name in result_dict:
        all_result.append([name, np.mean(result_dict[name], axis=0)])
    print('Result count:', len(all_result))
    return all_result


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train face network')
    # general
    parser.add_argument('--threshold', type=int, default=0, help='threshold')
    parser.add_argument('--model-prefix', default='../save_model/tmp_model/model_all_mix.hdf5_0_200')
    parser.add_argument('--save-path', default='/home/jzhengas/Jason/result/result/')
    parser.add_argument('--pickle', default='/home/jzhengas/Jason/data/simple_model/test_mean.pickle')
    parser.add_argument('--allow-empty',type=int, default=0)
    parser.add_argument('--ratio',type=int, default=2)
    args = parser.parse_args()
    threshold = args.threshold
    x, name_data = get_data(args.pickle,  threshold=threshold, allow_empty=args.allow_empty, ratio=args.ratio)
    
    prefix = args.model_prefix
    model_list = [prefix+'_0_200.hdf5',prefix+'_20_200.hdf5',prefix+'_40_200.hdf5',
                  prefix+'_0_80.hdf5']
    result_list = []
    for model_name in model_list:
        model = load_model(model_name)
        result = model.predict(x, batch_size=256)
        print('Finish Predict...')

        all_result = get_result_dict(name_data, result)
        result = None
        result_list.append(all_result)

    merge_result = []
    for i in range(0, len(result_list[0])):
        video_name = result_list[0][i][0]
        result = result_list[0][i][1]
        for j in range(1, len(result_list)):
            result += result_list[j][i][1]
        result = result / len(result_list)
        merge_result.append((video_name, result))

    pickle_path = args.save_path
    with open(pickle_path, 'wb+') as f:
        pickle.dump(merge_result, f)



