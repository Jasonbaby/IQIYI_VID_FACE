import pickle
import random
import numpy as np
import argparse


def read_from_pickle(pickle_path):
    with open(pickle_path, 'rb') as fin:
        result = pickle.load(fin, encoding='iso-8859-1')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train face network')

    parser.add_argument('--pickle-file', default='../submit_result/best_face.pickle')
    parser.add_argument('--img-file', default='result/tmp_resultv1.pickle')
    parser.add_argument('--img-file2', default='result/tmp_resultv2.pickle')
    parser.add_argument('--img-file3', default='result/tmp_resultv3.pickle')
    parser.add_argument('--target', default='../submit_result/best_merge.pickle', help='directory to save model.')
    parser.add_argument('--txt', type=int, default=1)
    args = parser.parse_args()


    pre_result = read_from_pickle(args.pickle_file)
    img_result = read_from_pickle(args.img_file)
    img_result2 = read_from_pickle(args.img_file2)
    img_result3 = read_from_pickle(args.img_file3)
    ratio_1, ratio_2, only2 = 0.5, 0.5, 0.8


    merge_result = []
    result_dict = {}
    for i in range(0, len(pre_result)):
        result_dict[pre_result[i][0]] = pre_result[i][1]

    for video_name in img_result:
        img_result[video_name] = np.concatenate([img_result[video_name],[0]])
        if video_name in result_dict:
            result_dict[video_name] = 0.5*result_dict[video_name] + 0.5*img_result[video_name]
            
        else:
            result_dict[video_name] = only2*img_result[video_name]

    for video_name in img_result2:
        img_result2[video_name] = np.concatenate([img_result2[video_name],[0]])
        if video_name in result_dict:
            result_dict[video_name] = 0.6*result_dict[video_name] + 0.4*img_result2[video_name]
            
        else:
            result_dict[video_name] = only2*img_result2[video_name]

    for video_name in img_result3:
        img_result3[video_name] = np.concatenate([img_result3[video_name],[0]])
        if video_name in result_dict:
            result_dict[video_name] = 0.7*result_dict[video_name] + 0.3*img_result3[video_name]
        else:
            result_dict[video_name] = only2*img_result3[video_name]
            

    merge_result = []
    for video_name in result_dict:
        merge_result.append([video_name, result_dict[video_name]])

 
    if args.txt:
        classify_result = []
        for i in range(0, 4934):
            classify_result.append([])
        print('Start sorting...')
        for i in range(0, 4934):
            merge_result.sort(key = lambda x: x[1][i], reverse=True)
            classify_result[i] = merge_result[0:100]

        result_path = args.target + '.txt'
        with open(result_path, 'w+') as fin:
            for i in range(0, 4934):
                output_str = str(i+1)
                data = classify_result[i]
                for d in data:
                    output_str += (' ' + d[0])
                output_str += '\n'
                fin.write(output_str)

    with open(args.target+'.pickle', 'wb+') as f:
        pickle.dump(merge_result, f)


