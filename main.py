import pyrealsense2 as rs
width = 640
height = 480

pipe = rs.pipeline()
config = rs.config()

# Start streaming
profile = pipe.start()

from PIL import Image
import tensorflow as tf
from torch import nn
import imutils
import cv2
import torch
import torchreid
from torch.nn import functional as F
import numpy as np
from DetectorAPI import DetectorAPI
import matplotlib.pyplot as plt

import sys

orig_stdout = sys.stdout
f = open('out.txt', 'w')

file_path = '/home/qwe/Downloads/osnet_x0_25_market_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth'

num_classes = 6
model = torchreid.models.build_model(
    name='osnet_x0_25',
    num_classes=num_classes,
    loss='softmax',
    pretrained=True
)

torchreid.utils.load_pretrained_weights(model, file_path)
model = nn.DataParallel(model).cuda()

reid_network = model

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
    imgs_query = img_resized[ int(p[0]):int(p[2]), int(p[1]):int(p[3])]
    return imgs_query


def get_dist(p_arr, img_resized, gallary_features, body_bank_bb):
    """
    Get euclidean distance between features of persons
    :param p_arr: detected bboxes of persons
    :param img_resized: image where need to do a crop
    :param gallary_features: gallary of features from previous frames
    :param body_bank_bb:
    :return:
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
            features_img_query =  F.normalize(extract_features(torch.from_numpy(img).cuda()), p=2, dim=1).cpu().numpy()[0]
            query_f.append(features_img_query)

            distmat = np.sum((features_img_query[:] - features_img_gallery[:]) ** 2)
            distmat_arr[i, j] = distmat #/ bb_intersection_over_union(p_arr[i], body_bank_bb[j])

    print("dist", distmat_arr)

    return distmat_arr, query_f



curTime = 0
img = np.zeros((10, 5))
W, H = 416, 416


model_path = './model/frozen_inference_graph.pb'
odapi = DetectorAPI(path_to_ckpt=model_path)
threshold = 0.8
iou_threshold = 0.6


body_bank = []
body_bank_dist = []
body_bank_bb = []
body_dist = []
body_im_array = []
plt.ion()

with tf.Session() as sess:
    args_save = True
    if args_save:
        video_writer = cv2.VideoWriter('recording.avi', cv2.VideoWriter_fourcc(*'XVID'), 6, (W, H))


    while True:
        # grab the next frame and handle if we are reading from either
        # VideoCapture or VideoStream
        curTime += 1
        print(curTime)
        #r, frameL = vs.read()

        frameset = pipe.wait_for_frames()
        color_frame = frameset.get_color_frame()

        color = np.asanyarray(color_frame.get_data())

        frameL = color


        frameL = cv2.resize(frameL, (1280, 720))

        if curTime < 10:
            continue

        if curTime % 4 != 0:
            continue

        if frameL is None:
            break

        detections, scores, classes, num = odapi.processFrame(frameL)

        mask = np.array([a & b for a, b in zip(np.array(classes) == 1, np.array(scores) > threshold )])
        detections = np.array(detections)

        if np.sum(mask) != 0:
            detections = detections[mask]

            #scor = scores[mask]
            #lab = classes[mask]
        else:
            detections = []
        print("detections", detections)
        if len(detections) != 0:
            #    continue

            # init body_bank

            result_arr = []
            result_features = []
            result_dist_arr = []
            body_bank_tmp = body_bank.copy()
            body_bank_bb_tmp = body_bank_bb.copy()
            len_body_bank_tmp = len(body_bank_tmp)

            for i, body_detection_xy in enumerate(detections):
                if len(body_bank) != 0:
                    print(len(body_bank))

                    if len_body_bank_tmp == 0:
                        print("add new user909090909090")
                        result = len(body_bank) + 1
                        p = detections[i]
                        p = check_coords(p)
                        # img = cv2.resize(frameL[int(p[1]):int(p[3]), int(p[0]):int(p[2])], (256, 128)).astype('f') / 255.
                        img = np.transpose(frameL[int(p[0]):int(p[2]), int(p[1]):int(p[3])]).astype('f') / 255.
                        img = np.expand_dims(img, axis=0)
                        print(np.shape(img))
                        features_img_query = F.normalize(extract_features(torch.from_numpy(img).cuda())).cpu().numpy()[0]
                        body_bank.append(features_img_query)
                        body_bank_bb.append(detections[i])

                        body_bank_dist.append(0)
                        continue

                    distmat_arr, query_f = get_dist([np.array(body_detection_xy)], frameL, body_bank_tmp, body_bank_bb_tmp)
                    result = np.argmin(distmat_arr, 1)[0]
                    print(distmat_arr[0][result])
                    # if (distmat_arr[0][result] < body_bank_dist[result] + 100):
                    if (distmat_arr[0][result] < 0.3):
                        result_arr.append(result)
                        body_bank_tmp[result] = []
                        body_bank_bb_tmp[result] = []
                        len_body_bank_tmp -= 1
                        result_features.append(query_f[result])
                        result_dist_arr.append(np.min(distmat_arr, 1)[0])
                        # choose one and remove him

                    else :
                        print("add new user")
                        body_bank.append(query_f[result])
                        body_bank_bb.append(detections[i])
                        body_bank_dist.append(0)

                else:
                    print("add new user")
                    result = len(body_bank) + 1
                    p = detections[i]
                    p = check_coords(p)

                    img = np.transpose(frameL[int(p[0]):int(p[2]), int(p[1]):int(p[3])]).astype('f') / 255.
                    img = np.expand_dims(img, axis=0)
                    print(np.shape(img))
                    features_img_query = F.normalize(extract_features(torch.from_numpy(img).cuda())).cpu().numpy()[0]
                    body_bank.append(features_img_query)
                    body_bank_bb.append(detections[i])
                    body_bank_dist.append(0)

            if len(body_bank) != 0:
                fig, ax = plt.subplots(nrows=len(body_bank), ncols=3, gridspec_kw={'width_ratios': [2, 10, 10]}, figsize=(10, 2))

                if len(body_bank) == 1:
                    ax = [ax]

            for i, pack in enumerate(zip(result_arr, result_dist_arr, ax)):
                id, result, row = pack
                # add new human if % is too low

                # refresh bank
                # if result < body_bank_dist[id] + 500:

                if len(body_bank) != 0:
                    for idx, col in enumerate(row):
                        if idx == 0:
                            p = detections[i]
                            p = check_coords(p)
                            col.imshow(frameL[int(p[0]):int(p[2]), int(p[1]):int(p[3])])
                        if idx == 1:
                            col.plot(range(0, len(body_bank[id]) + 1), list(body_bank[id]) + [0.40])
                        if idx == 2:
                            print(int("".join(str(int(x)) for x in [ x > 0.1 and y > 0.1 for x, y in zip(body_bank[id],result_features[i])]), 2))
                            print(str(int("".join(str(int(x)) for x in [ x > 0.15 and y > 0.15 for x, y in zip(body_bank[id],result_features[i])]), 2)) + '\n', file=f)
                            #col.plot(int("".join(str(int(x)) for x in [ x > 0.1 and y > 0.1 for x, y in zip(body_bank[id],result_features[i])]), 2))

                if result < 0.3:
                    p = detections[i]
                    p = check_coords(p)
                    body_bank_dist[id] = result# + body_bank_dist[id]) / 2
                    print("query_f == detections ", len(query_f), len(detections))
                    body_bank[id] = result_features[i] #+ body_bank[id]) / 2  #frameL[int(p[1]):int(p[3]), int(p[0]):int(p[2])]
                    #body_bank_bb.append(detections[i])

                print("ID", id)



                text = "_ID_{} <{}>".format(str(id), str(round(result, 3)))
                # cv2.putText(img_resized1, text,
                #             (int(detections1[idx_r][0]) - 10, int(detections1[idx_r][1]) - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255))
                print("detect", detections)
                cv2.putText(frameL, text,
                            (int(detections[i][1]) - 10, int(detections[i][0]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            #frameL = cv2.cvtColor(frameL, cv2.COLOR_BGR2RGB)

        plt.show()
        plt.pause(4)
        plt.close()
        frameL = frameL / 255
        for p in detections:
            cv2.rectangle(frameL, (p[1], p[0]), (p[3], p[2]), (50, 50, 250), 2)
        #cv2.imshow("result", img)
        #frameL = cv2.resize(frameL, (IMAGE_W, IMAGE_H))

        cv2.imshow("L", frameL)
        #img_resized = np.transpose(img_resized)

        print("shape ", frameL.shape)

        if args_save:
            video_writer.write((frameL * 255).astype(np.uint8))
        # cv2.imshow("R", img_resized1)
        cv2.waitKey(1)
