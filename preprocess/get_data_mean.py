# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import time
import random
from util import remove_outlier, get_video_annotation


def get_data(pickle_path='',  save_path='', only_new=False):
    with open(pickle_path, 'rb') as fin:
        feats_dict = pickle.load(fin, encoding='iso-8859-1')
    with open('/home/jzhengas/Jason/data/simple_model/dict.pickle', 'rb') as fin:
        data_dict = pickle.load(fin, encoding='iso-8859-1')

    window_len = [2,3,4,5,6]
    data_dict = None
    x = []
    y = []
    det_score_list = []
    face_score_list = []
    name_list = []
    for video_name in feats_dict:
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
        
        label = int(label)-1
        tmp_x = []
        tmp_det = []
        tmp_score = []

        tmp_y = []
        tmp_name = []
        for feat in feats:
            [frame_num, bbox, det_score, qua_score, feat_arr] = feat
            tmp_x.append(feat_arr)
            tmp_det.append(det_score)
            tmp_score.append(qua_score)

        new_feats = [] if only_new else feats

        for d in window_len:
            s_list = list(zip(tmp_x, tmp_det, tmp_score))
            random.shuffle(s_list)
            tmp_x, tmp_det, tmp_score = zip(*s_list)
            for i in range(0, len(tmp_x)-d, 2):
                x = np.mean(tmp_x[i:i+d], axis=0)
                det = np.mean(tmp_det[i:i+d], axis=0)
                score = np.mean(tmp_score[i:i+d], axis=0)
                new_feats.append([0, 0, det, score, x])

        feats_dict[video_name] = new_feats

    with open(save_path, 'wb+') as f:
        pickle.dump(feats_dict, f)


if __name__ == '__main__':
    save_path = '../data/test_mean.pickle'
    pickle_path = '/home/jzhengas/Jason/data/feats_test.pickle'
    get_data(pickle_path=pickle_path, save_path=save_path, only_new=False)

