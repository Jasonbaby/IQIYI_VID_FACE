import sys
import mxnet as mx
import numpy as np
import argparse
import random
import os
from PIL import Image
from PIL import ImageFilter
from PIL import ImageEnhance
from collections import namedtuple  
from mxnet import io,nd
import pickle

prefix = 'save_model/se-resnext-imagenet-50-0'
epoch = 16
ctx = mx.gpu()

batch_size = 64

model = mx.module.Module.load(prefix = prefix, epoch = epoch, context = ctx) 
model.bind(for_training=False, data_shapes=[('data', (batch_size, 3, 224, 224))] )
Batch = namedtuple('Batch', ['data'])




def transform(image):
    image = mx.image.resize_short(image, 224) #minimum 224x224 images
    image, crop_info = mx.image.center_crop(image, (224, 224))
    
    image = image.transpose((2,0,1))  # Transposing from (224, 224, 3) to (3, 224, 224)
    # image = transposed.expand_dims(axis=0) # change the shape from (3, 224, 224) to (1, 3, 224, 224)
    image = nd.array(image, ctx)
    return image



def predict(batch_file):
    batch_data = nd.empty((batch_size, 3, 224, 224), ctx)
    for i in range(0, batch_size):
        fn = batch_file[i]
        img = mx.image.imread(fn)
        img = transform(img)
        batch_data[i][:] = img
        # img = nd.array(img, ctx)
    model.forward(Batch([batch_data]), is_train = False)
    outputs = model.get_outputs()[0].asnumpy()
    return outputs


def predict_directory(video_list, file_list):
    all_result = []
    file_num = len(file_list)
    print('Image num: ', file_num)
    begin, end = 0, 0
    for i in range(0, file_num, batch_size):
        begin = i
        end = i + batch_size
        batch_file = []
        if end < file_num:
            batch_file = file_list[begin: end]
        else:
            extra_num = end - file_num + 1
            batch_file = file_list[begin: file_num]
            batch_file = batch_file + file_list[0: extra_num]
    
        predict_result = predict(batch_file)
        all_result.append(predict_result)
    all_result = np.concatenate(all_result)
    print('finish predict')
    result_dict = {}
    for i in range(0, file_num):
        video = video_list[i]
        if video in result_dict:
            result_dict[video].append(all_result[i])
        else:
            result_dict[video] = [all_result[i]]
    for video in result_dict:
        result_dict[video] = np.mean(result_dict[video], axis=0)
    return result_dict


def get_img_list(directory, video_name_prefix):
    path_iter = os.walk(directory)
    file_list = []
    count = 0
    for root, dirs, files in path_iter:
        count += 1
        if count == 1:
            video_list = dirs
        else:
            for f in files:
                file_list.append([root.split('/')[-1], os.path.join(root, f)])
    random.shuffle(file_list)
    video_list, file_list = zip(*file_list)
    video_list = list(video_list)
    for i in range(0, len(video_list)):
        video_list[i] = video_name_prefix + video_list[i] + '.mp4'
    return video_list, file_list




if __name__ == "__main__":
    video_list, file_list = get_img_list('/home/jzhengas/Jason/img_data/test_data_part1', 'IQIYI_VID_TEST_')
    result_dict = predict_directory(video_list, file_list)
    with open('result/tmp_resultv1.pickle', 'wb+') as f:
        pickle.dump(result_dict, f)

    video_list, file_list = get_img_list('/home/jzhengas/Jason/img_data/test_data_part2', 'IQIYI_VID_TEST_')
    result_dict = predict_directory(video_list, file_list)
    with open('result/tmp_resultv2.pickle', 'wb+') as f:
        pickle.dump(result_dict, f)

    video_list, file_list = get_img_list('/home/jzhengas/Jason/img_data/test_data_part3', 'IQIYI_VID_TEST_')
    result_dict = predict_directory(video_list, file_list)
    with open('result/tmp_resultv3.pickle', 'wb+') as f:
        pickle.dump(result_dict, f)

