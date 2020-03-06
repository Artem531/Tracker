REID-Tracker
===========

REID-Tracker is a tracker with person re-identification in `PyTorch <https://pytorch.org/>`_.

It features:

- intel realsense camera support
- Torchreid models

REID model
===========
Code: https://github.com/KaiyangZhou/deep-person-reid.

Documentation: https://kaiyangzhou.github.io/deep-person-reid/.

How-to instructions: https://kaiyangzhou.github.io/deep-person-reid/user_guide.

Model zoo: https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.

What's new
---------------
- [March 2020] Add first version of REID-tracker on model from model zoo.

Installation
---------------

Make sure `conda <https://www.anaconda.com/distribution/>`_ is installed.


.. code-block:: bash

    # cd to your preferred directory and clone this repo
    git clone https://github.com/Artem531/Tracker

    # create environment and install dependencies
    conda env create -f Tracker.yml

Get started:
-------------------------------------
1. Main

Start main.py to run tracking on intel realsense camera

2. Config

.. code-block:: python

    # reid model
    cfg.model.name = 'osnet_x0_25'
    cfg.model.load_weights = 'osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth' # path to model weights
    cfg.model.num_classes = 6 # Default value to init reid-model
    cfg.model.loss = 'softmax' # Default value to init reid-model
    cfg.model.pretrained = True # Default value to init reid-model
    cfg.model.threshold = 0.1  # reid new person threshold

    # detector
    cfg.detector.load_weights = './model/frozen_inference_graph.pb' # path to detector model
    cfg.detector.threshold = 0.95 # detector class threshold
    cfg.detector.iou_threshold = 0.80  # detector iou threshold

    # image
    cfg.image.width = 1280 # image width
    cfg.image.height = 720 # image height

    # video
    cfg.video.path = 'results/recording.avi' # path to save results

    # optimizer
    cfg.visualisation.key = False # this is visualisation of reid feature vector. Use only for testing in terminal mode

3. DetectorAPI

Class for SSD detector

.. code-block:: python

    # init SSD
    odapi = DetectorAPI(path_to_ckpt=cfg.detector.load_weights)
    # get bboxes of objects
    detections, scores, classes, num = odapi.processFrame(frameL)


4. Utils

Package with functions for Tracker

.. code-block:: python

    def extract_features(input) #  Extract features of person from reid model
    def bb_intersection_over_union(boxA, boxB) # Calculate iou of boxes
    def get_middle(p) # get middle point of bbox
    def return_orig_point_size(p, im) # rescale coordinates for new image shape
    def check_coords(p) # if coordinate have negative values clip them to zero
    def crop_im(p, img_resized) # get image of person
    def get_dist(p_arr, img_resized, gallary_features) # Get euclidean distance between features of persons


