"""
Add dropout layer followed by last pooling layer.
Updated by Lin Xiong Jul-21, 2017
"""
import argparse,logging,os
import mxnet as mx
from symbol_se_resnext_w_d import resnext
import mxnet.optimizer as optimizer
logger = logging.getLogger()
logger.setLevel(logging.INFO)

formatter = logging.Formatter('%(asctime)s - %(message)s')
console = logging.StreamHandler()
console.setFormatter(formatter)
logger.addHandler(console)

# load and tune model
def get_fine_tune_model(model_name, epoch):
    # load model
    symbol, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch)
    # model tuning
    all_layers = symbol.get_internals()
    net = all_layers['flatten0_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=args.num_classes, name='newfc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    # eliminate weights of new layer
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args,aux_params)



def main():
    ratio_list = [0.25, 0.125, 0.0625, 0.03125]   # 1/4, 1/8, 1/16, 1/32
    if args.depth == 18:
        units = [2, 2, 2, 2]
    elif args.depth == 34:
        units = [3, 4, 6, 3]
    elif args.depth == 50:
        units = [3, 4, 6, 3]
    elif args.depth == 101:
        units = [3, 4, 23, 3]
    elif args.depth == 152:
        units = [3, 8, 36, 3]
    elif args.depth == 200: 
        units = [3, 24, 36, 3]
    elif args.depth == 269:
        units = [3, 30, 48, 8]
    else:
        raise ValueError("no experiments done on detph {}, you can do it youself".format(args.depth))
    symbol = resnext(units=units, num_stage=4, filter_list=[64, 256, 512, 1024, 2048] if args.depth >=50
                        else [64, 64, 128, 256, 512], ratio_list=ratio_list, num_class=args.num_classes, num_group=args.num_group, data_type="imagenet", drop_out=args.drop_out, bottle_neck = True
                        if args.depth >= 50 else False, bn_mom=args.bn_mom, workspace=args.workspace,
                        memonger=args.memonger)

    kv = mx.kvstore.create(args.kv_store)
    devs = mx.cpu() if args.gpus is None else [mx.gpu(int(i)) for i in args.gpus.split(',')]
    epoch_size = max(int(args.num_examples / args.batch_size / kv.num_workers), 1)
    begin_epoch = args.model_load_epoch if args.model_load_epoch else 0
    if not os.path.exists("./"+args.model_name):
        os.mkdir("./"+args.model_name)
    model_prefix = args.model_name+"/se-resnext-{}-{}-{}".format(args.data_type, args.depth, kv.rank)
    checkpoint = mx.callback.do_checkpoint(model_prefix)
    arg_params = None
    aux_params = None
    load_model_prefix = 'model/se-resnext-imagenet-50-0'
    if args.retrain:
        if args.finetune:
            (symbol,arg_params,aux_params)=get_fine_tune_model(load_model_prefix, args.model_load_epoch)
        else:
            symbol, arg_params, aux_params = mx.model.load_checkpoint(model_prefix, args.model_load_epoch)
    if args.memonger:
        import memonger
        symbol = memonger.search_plan(symbol, data=(args.batch_size, 3, 32, 32) if args.data_type=="cifar10"
                                                    else (args.batch_size, 3, 224, 224))
    train = mx.io.ImageRecordIter(
        path_imgrec         = args.data_train+'.rec',
        path_imgidx         = args.data_train+'.idx',
        label_width         = 1,
        data_name           = 'data',
        label_name          = 'softmax_label',
        data_shape          = (3, 224, 224),
        batch_size          = args.batch_size,
        pad                 = 0,
        fill_value          = 0,  # only used when pad is valid
        rand_crop           = False,
        shuffle             = True)
    train.reset()
    if(args.data_val == 'None'):
        val = None
    else:
        val = mx.io.ImageRecordIter(
            path_imgrec         = args.data_val,
            label_width         = 1,
            data_name           = 'data',
            label_name          = 'softmax_label',
            batch_size          = args.batch_size,
            data_shape          = (3, 224, 224),
            rand_crop           = False,
            rand_mirror         = False,
            num_parts           = kv.num_workers,
            part_index          = kv.rank)

    fix_param = None
    if args.freeze:
        fix_param = [k for k in arg_params if 'fc' not in k]

    model = mx.mod.Module(symbol=symbol, context=devs, fixed_param_names = fix_param)
    model.bind(data_shapes = train.provide_data, label_shapes = train.provide_label)
    # sgd = mx.optimizer.Optimizer.create_optimizer('sgd')
    # finetune_lr = dict({k: 0 for k in arg_params})
    # sgd.set_lr_mult(finetune_lr)

    opt = optimizer.SGD(learning_rate=args.lr, momentum=0.9, wd=0.0005, rescale_grad=1.0/args.batch_size/(len(args.gpus.split(','))))
    
    # training
    model.fit(train, val,
        num_epoch=args.num_epoch,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        kvstore='device',
        optimizer=opt,
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        batch_end_callback = mx.callback.Speedometer(args.batch_size, args.frequent),
        epoch_end_callback = checkpoint,
        eval_metric=['acc', 'ce'])



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="command for training resnet-v2")
    parser.add_argument('--gpus', type=str, default='0', help='the gpus will be used, e.g "0,1,2,3"')
    parser.add_argument('--data-type', type=str, default='imagenet', help='the dataset type')
    parser.add_argument('--data-train', type=str, default='data/train.rec', help='the train dataset')
    parser.add_argument('--data-val', type=str, default='None', help='the val dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='initialization learning reate')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum for sgd')
    parser.add_argument('--bn-mom', type=float, default=0.9, help='momentum for batch normlization')
    parser.add_argument('--wd', type=float, default=0.0001, help='weight decay for sgd')
    parser.add_argument('--batch-size', type=int, default=256, help='the batch size')
    parser.add_argument('--num-epoch', type=int, default=100, help='the epoch num')
    parser.add_argument('--num-group', type=int, default=32, help='the number of convolution groups')
    parser.add_argument('--drop-out', type=float, default=0.0, help='the probability of an element to be zeroed')
    parser.add_argument('--workspace', type=int, default=512, help='memory space size(MB) used in convolution, if xpu '
                        ' memory is oom, then you can try smaller vale, such as --workspace 256')
    parser.add_argument('--depth', type=int, default=50, help='the depth of resnet')
    parser.add_argument('--num-classes', type=int, default=1000, help='the class number of your task')
    parser.add_argument('--aug-level', type=int, default=2, choices=[1, 2, 3],
                        help='level 1: use only random crop and random mirror\n'
                             'level 2: add scale/aspect/hsv augmentation based on level 1\n'
                             'level 3: add rotation/shear augmentation based on level 2')
    parser.add_argument('--num-examples', type=int, default=1281167, help='the number of training examples')
    parser.add_argument('--kv-store', type=str, default='device', help='the kvstore type')
    parser.add_argument('--model-name', type=str, default='save_model', help='the name of classifier')
    parser.add_argument('--model-load-epoch', type=int, default=0,
                        help='load the model on an epoch using the model-load-prefix')

    parser.add_argument('--frequent', type=int, default=50, help='frequency of logging')
    parser.add_argument('--freeze', type=int, default=0)
    parser.add_argument('--finetune', type=int, default=1)
    parser.add_argument('--memonger', action='store_true', default=False,
                        help='true means using memonger to save momory, https://github.com/dmlc/mxnet-memonger')
    parser.add_argument('--retrain', action='store_true', default=False, help='true means continue training')
    args = parser.parse_args()
    hdlr = logging.FileHandler('./log/log-se-resnext-{}-{}.log'.format(args.data_type, args.depth))
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr)
    logging.info(args)
    main()

