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

    parser.add_argument('--pickle-file', default='../save_model/tmp_model/model_all_mix.hdf5', help='directory to save model.')
    parser.add_argument('--target', default='merge/result.txt', help='directory to save model.')
    parser.add_argument('--txt', type=int, default=1)
    args = parser.parse_args()
    
    pickle_file = args.pickle_file.split(',')

    pickles = [read_from_pickle(f) for f in pickle_file]

    for pickle_data in pickles:
        print(len(pickle_data))
        pickle_data.sort(key = lambda x: x[0])

    merge_result = []
    for i in range(0, len(pickles[0])):
        video_name = pickles[0][i][0]
        result = pickles[0][i][1]
        for j in range(1, len(pickles)):
            result += pickles[j][i][1]
        result = result/ len(pickles)
        merge_result.append((video_name, result))

    print(len(merge_result))

    result_path = args.target  
    if args.txt:
        classify_result = []
        for i in range(0, 4934):
            classify_result.append([])
        print('Start sorting...')
        for i in range(0, 4934):
            merge_result.sort(key = lambda x: x[1][i], reverse=True)
            classify_result[i] = merge_result[0:100]

        
        with open(result_path, 'w+') as fin:
            for i in range(0, 4934):
                output_str = str(i+1)
                data = classify_result[i]
                for d in data:
                    output_str += (' ' + d[0])
                output_str += '\n'
                fin.write(output_str)

    with open(result_path+'.pickle', 'wb+') as f:
        pickle.dump(merge_result, f)


        