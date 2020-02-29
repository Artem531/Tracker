from PIL import Image
from torch import nn
import cv2
import torch
import torchreid
from torch.nn import functional as F
import numpy as np
from default_config import get_default_config

cfg = get_default_config()

model = torchreid.models.build_model(
    name=cfg.model.name,
    num_classes=cfg.model.num_classes,
    loss=cfg.model.loss,
    pretrained=cfg.model.pretrained
)

torchreid.utils.load_pretrained_weights(model, cfg.model.load_weights)
model = nn.DataParallel(model).cuda()


@torch.no_grad()
def extract_features(input):
    """
    Extract features function
    :param input: image of type numpy array
    :return: vector of features
    """
    model.eval()
    return model(input)


def bb_intersection_over_union(boxA, boxB):
    """

    :param boxA: box of first person [x1,y1,x2,y2]
    :param boxB: box of second person [x1,y1,x2,y2]
    :return: intersection over union value
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def get_middle(p):
    """
    get middle point of bb
    :param p: bbox array [x1,y1,x2,y2]
    :return: x, y point
    """
    return p[0] + (p[2] - p[0]) / 2, p[1] + (p[3] - p[1]) / 2


def return_orig_point_size(p, im):
    """
    rescale coordinates for new image shape
    :param p: bbox array [y1,x1,y2,x2]
    :param im:
    :return: bbox array [y1,x1,y2,x2] for new image
    """
    startX = p[1]
    startY = p[0]
    endX = p[3]
    endY = p[2]

    bbox = np.array([startX, startY, endX, endY])

    # convert_to_original_size
    detection_size, original_size = np.array([416, 416]), np.array(Image.fromarray(im, "RGB").size)
    ratio = original_size / detection_size
    bbox = list((bbox.reshape(2, 2) * ratio).reshape(-1))

    startX = bbox[0]
    startY = bbox[1]
    endX = bbox[2]
    endY = bbox[3]
    return startX, startY, endX, endY


def check_coords(p):
    """
    if coordinate have negative values clip them to zero
    :param p: bbox array [y1,x1,y2,x2]
    :return: bbox array [y1,x1,y2,x2]
    """
    idx = 0
    for i in p:
        p[idx] = i if i > 0 else 0
        idx += 1

    return p


def crop_im(p, img_resized):
    """
    get image of person
    :param p: bbox array [y1,x1,y2,x2]
    :param img_resized: image where need to do a crop
    :return: image of person
    """
    p = check_coords(p)
    imgs_query = img_resized[int(p[0]):int(p[2]), int(p[1]):int(p[3])]
    return imgs_query


def get_dist(p_arr, img_resized, gallary_features):
    """
    Get euclidean distance between features of persons
    :param p_arr: detected bboxes of persons
    :param img_resized: image where need to do a crop
    :param gallary_features: gallary of features from previous frames
    :param body_bank_bb:
    :return: euclidean distance
    """
    # get id
    imgs_query = []

    i = 0
    for p in p_arr:
        img_query = crop_im(p, img_resized)
        imgs_query.append(img_query)
        i += 1

    distmat_arr = np.zeros((len(imgs_query), len(gallary_features)))
    print("shape", distmat_arr.shape)
    query_f = []

    for i, img_query in enumerate(imgs_query):
        for j, features_img_gallery in enumerate(gallary_features):
            # # Load images
            if type(features_img_gallery) is list:
                print("features_img_gallery", features_img_gallery)
                query_f.append([])
                distmat_arr[i, j] = 10000
                continue
            img_query = cv2.resize(img_query, (256, 128))
            img = np.transpose(img_query).astype('f') / 255.
            img = np.expand_dims(img, axis=0)
            print(np.shape(img))
            features_img_query = F.normalize(extract_features(torch.from_numpy(img).cuda()), p=2, dim=1).cpu().numpy()[
                0]
            query_f.append(features_img_query)

            distmat = np.sum((features_img_query[:] - features_img_gallery[:]) ** 2)
            distmat_arr[i, j] = distmat  # / bb_intersection_over_union(p_arr[i], body_bank_bb[j])

    print("dist", distmat_arr)

    return distmat_arr, query_f


def iou(box1, box2):
    """
    calculate intersection over union
    """
    xa = max(box1[1], box2[1])
    ya = max(box1[0], box2[0])
    xb = min(box1[3], box2[3])
    yb = min(box1[2], box2[2])

    interArea = max(0, xb - xa) * max(0, yb - ya)

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou_value = float(interArea) / float(box1Area + box2Area - interArea)

    # return the intersection over union value
    return iou_value


def make_rgb(img):
    """
    intel img format to rgb format
    """
    img[:,:, [0, 2]] = img[:,:, [2, 0]]
    return img
