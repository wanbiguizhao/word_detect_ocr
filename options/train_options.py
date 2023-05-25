import argparse
def trainparser():
    parser = argparse.ArgumentParser(description="训练一个神经网络用于预测wordslice是汉字的一部分还是两个字之间的部分")
    parser.add_argument("--data",type=str,default="tmp/dataset02", metavar="DIR", help="path to dataset,指向按行切割的图片的文件夹目录")
    parser.add_argument(
        "--expansion",
        default=3,
        type=int,
        metavar="EXPAN",
        help="对于连着的图片的数据，复制倍数",
    )
    parser.add_argument(
        "--test-size",
        default=0.3,
        type=float,
        metavar="TS",
        help="test size of all data ",
        #dest="lr",
    )
    parser.add_argument("--moco_model",type=str,default="tmp/checkpoint/epoch_011_bitchth_003500_model.pdparams", metavar="Backnone", help="对比模型的存储目录")
    parser.add_argument("--freeze_flag", action="store_true", help="训练模型是否冻结backbone")# paddle的学习率使用策略和pytorch不一样
    parser.add_argument("--checkpoint", type=str,default="tmp/nobackbone", help="训练模型的保存位置")# paddle的学习率使用策略和pytorch不一样
    parser.add_argument("--checkpoint_steps", type=int,default=2, help="每过多少轮保存一下模型")# paddle的学习率使用策略和pytorch不一样

    parser.add_argument("--logdir", type=str,default="tmp/moco_cls", help="训练日志存储路径")# paddle的学习率使用策略和pytorch不一样

    parser.add_argument(
        "-j",
        "--workers",
        default=4,
        type=int,
        metavar="N",
        help="加载dataset的work",
    )
    parser.add_argument(
        "--epochs", default=21, type=int, metavar="N", help="总共训练的轮数" 
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="manual epoch number (useful on restarts，主要是方便计算学习率)",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help=" 这个意思莫非是，多卡GPU的情况，256被多卡平均使用，想多了，用不到多卡",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.01,
        type=float,
        metavar="LR",
        help="initial learning rate",
        dest="lr",
    )
    parser.add_argument(
        "--schedule",
        default=[5, 20,50,100,120],
        nargs="*",
        type=int,
        help="learning rate schedule (when to drop lr by 10x)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum of SGD solver"
    )
    parser.add_argument(
        "--wd",
        "--weight-decay",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
        dest="weight_decay",
    )
    parser.add_argument(
        "-p",
        "--print-freq",
        default=10,
        type=int,
        metavar="N",
        help="print frequency (default: 10)",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--seed", default=None, type=int, help="seed for initializing training. "
    )
    parser.add_argument("--cos", action="store_true", help="use cosine lr schedule")# paddle的学习率使用策略和pytorch不一样
    parser.add_argument("--cpu", action="store_true", help="使用cpu训练")# paddle的学习率使用策略和pytorch不一样
    return parser