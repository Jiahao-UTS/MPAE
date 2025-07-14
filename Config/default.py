from yacs.config import CfgNode as CN

_C = CN()
_C.GPUS = (0, )
_C.WORKERS = 0
_C.PIN_MEMORY = True
_C.AUTO_RESUME = True
_C.PRINT_FREQ = 10


_C.DATASET = CN()
_C.DATASET.ROOT = "./Dataloader"
_C.DATASET.CHANNEL = 3
_C.DATASET.DATASET = 'PartImage_Seg'

_C.SYSTEM = CN()
_C.SYSTEM.NUM_GPUS = 1

_C.MODEL = CN()
_C.MODEL.NAME = "MPAE"
_C.MODEL.IMG_SIZE = 224
_C.MODEL.RATIO = 0.90
_C.MODEL.OUT_DIM = 256
_C.MODEL.BACKGROUND = 1

_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

_C.LOSS = CN()

_C.TRAIN = CN()
_C.TRAIN.TRAIN = True
_C.TRAIN.SHUFFLE = True
_C.TRAIN.LR = 0.0005
_C.TRAIN.LR_FACTOR = 0.2
_C.TRAIN.OPTIMIZER = "adam"
_C.TRAIN.BATCH_SIZE_PER_GPU = 1
_C.TRAIN.BEGIN_EPOCH = 0
_C.TRAIN.NUM_EPOCH = 100

_C.TEST = CN()
_C.TEST.BATCH_SIZE_PER_GPU = 2

_C.CELE = CN()
_C.CELE.ROOT = '../Data/CelebA/'
_C.CELE.FRACTION = 1.0
_C.CELE.SCALE = 0.05
_C.CELE.ROTATION = 15
_C.CELE.TRANSLATION = 0.05
_C.CELE.FLIP = 0.5

_C.CUB = CN()
_C.CUB.ROOT = '../Data/CUB/'
_C.CUB.FRACTION = 1.0
_C.CUB.SCALE = 0.05
_C.CUB.ROTATION = 15
_C.CUB.TRANSLATION = 0.05
_C.CUB.FLIP = 0.5

_C.PartImageNet = CN()
_C.PartImageNet.ROOT = '../Data/PartImageNetOOD/'
_C.PartImageNet.Seg_ROOT = '../Data/PartImageNet_Seg/'
_C.PartImageNet.FRACTION = 1.0
_C.PartImageNet.SCALE = 0.05
_C.PartImageNet.ROTATION = 15
_C.PartImageNet.TRANSLATION = 0.05
_C.PartImageNet.FLIP = 0.5

def update_config(cfg, args):
    cfg.defrost()

    if args.Num_Part:
        cfg.MODEL.NUM_PART = args.Num_Part

    if args.Dataset:
        cfg.DATASET.DATASET = args.Dataset
        if args.Dataset == "CUB":
            cfg.MODEL.IMG_SIZE = 448

    if args.Epoch:
        cfg.TRAIN.NUM_EPOCH = args.Epoch

    if args.LR_step:
        cfg.TRAIN.LR_STEP = args.LR_step

    cfg.OUTPUT_DIR = './Checkpoint'
    cfg.LOG_DIR = './log'

    cfg.freeze()
