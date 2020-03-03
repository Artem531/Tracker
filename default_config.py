from yacs.config import CfgNode as CN

def get_default_config():
    cfg = CN()

    # reid model
    cfg.model = CN()
    cfg.model.name = 'osnet_x0_25'
    cfg.model.pretrained = True # automatically load pretrained model weights if available
    cfg.model.load_weights = '/home/qwe/Downloads/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth' # path to model weights
    cfg.model.num_classes = 6
    cfg.model.loss = 'softmax'
    cfg.model.pretrained = True
    cfg.model.threshold = 0.1

    # detector
    cfg.detector = CN()
    cfg.detector.load_weights = './model/frozen_inference_graph.pb'
    cfg.detector.threshold = 0.95
    cfg.detector.iou_threshold = 0.99

    # image
    cfg.image = CN()
    cfg.image.width = 1280
    cfg.image.height = 780

    # video
    cfg.video = CN()
    cfg.video.path = 'results/recording.avi'

    # optimizer
    cfg.visualisation = CN()
    cfg.visualisation.key = False # use only for testing in terminal mode


    return cfg