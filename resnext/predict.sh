export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
python2 predict.py

python2 merge.py  --pickle-file  ../submit_result/best_face.pickle   --img-file  result/tmp_resultv1.pickle  --img-file2  result/tmp_resultv2.pickle   --img-file3   result/tmp_resultv3.pickle   --target  ../submit_result/best_merge.pickle

