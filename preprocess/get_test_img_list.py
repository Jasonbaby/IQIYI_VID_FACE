# -*- coding: utf-8 -*-
import os
import numpy as np
import pickle
import time


output_file1 = '../data/test_part1.lst'
output_file2 = '../data/test_part2.lst'
output_file3 = '../data/test_part3.lst'
pickle_path = '/home/jzhengas/Jason/data/feats_test.pickle'
img_prefix = ''


with open(pickle_path, 'rb') as fin:
    feats_dict = pickle.load(fin, encoding='iso-8859-1')

count = 0
identity = {}

out_fin1 = open(output_file1, 'w')
out_fin2 = open(output_file2, 'w')
out_fin3 = open(output_file3, 'w')

for video_name in feats_dict:
    feats = feats_dict[video_name]
    
    video_name = video_name.split('.')[0].split('_')[-1]
    
    if len(feats) == 0:
        out_fin1.write(video_name+'\n') 
        continue
    


    score_sum = 0
    for feat in feats: 
        [frame_num, bbox, det_score, qua_score, feat_arr] = feat
        score_sum += qua_score
    mean_score = 1.0 * score_sum / len(feats)
    if mean_score < 20 and mean_score > 0:
        out_fin1.write(video_name+'\n') 
    elif mean_score < 30 and mean_score > 20:
        out_fin2.write(video_name+'\n') 
    elif mean_score < 40 and mean_score > 30:
        out_fin3.write(video_name+'\n') 


  
out_fin1.close()  
out_fin2.close()  
out_fin3.close()  



