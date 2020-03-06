import pyrealsense2 as rs

from utils import check_coords, extract_features, get_dist, bb_intersection_over_union
import tensorflow as tf
import cv2
import torch
from default_config import get_default_config
from torch.nn import functional as F
import numpy as np
from DetectorAPI import DetectorAPI
import matplotlib.pyplot as plt
import sys

orig_stdout = sys.stdout
#f = open('out.txt', 'w')


def main():
    cfg = get_default_config()

    # init detector network
    print('init detector network...')
    odapi = DetectorAPI(path_to_ckpt=cfg.detector.load_weights)
    print("done!")

    # init intel camera
    print('init intel camera...')
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    # Start streaming
    pipe.start(config)
    print("done!")

    # init banks
    print("init gallary...")
    body_bank = []  # gallery array of persons
    body_bank_dist = []  #
    # body_bank_bb = []
    body_im_array = []
    print("done!")

    if cfg.visualisation.key:
        print("start visualisation mode...")
        plt.ion()
        print("done!")

    curTime = 0
    with tf.Session() as sess:
        args_save = True
        if args_save:
            video_writer = cv2.VideoWriter(cfg.video.path, cv2.VideoWriter_fourcc(*'XVID'), 6, (cfg.image.width, cfg.image.height))

        while True:
            curTime += 1
            #print(curTime)

            #-----------------------------------------

            # pipe to get image from intel cam
            frameset = pipe.wait_for_frames()
            color_frame = frameset.get_color_frame()
            color = np.asanyarray(color_frame.get_data()).copy()
            #print(np.shape(color), type(color))
            # rgb_frame = make_rgb(color)

            frameL = cv2.resize(color, (cfg.image.width, cfg.image.height))


            # -----------------------------------------
            # choose start frame point
            if curTime < 10:
                continue
            # choose fps
            if curTime % 1 != 0:
                continue
            # -----------------------------------------
            if frameL is None:
                break
            # -----------------------------------------

            # -----------------------------------------
            # get bbox of objects
            detections, scores, classes, num = odapi.processFrame(frameL)

            # get bbox of persons
            mask = np.array([a & b for a, b in zip(np.array(classes) == 1, np.array(scores) > cfg.detector.threshold )])
            detections = np.array(detections)
            if np.sum(mask) != 0:
                detections = detections[mask]
            else:
                detections = []

            # apply ion filter
            detections_iou = []
            for slow_loop_idx in range(len(detections)):

                if detections[slow_loop_idx][0] == -1:
                    continue

                for fast_loop_idx in range(len(detections)):
                    if detections[fast_loop_idx][0] == -1:
                        continue

                    r = bb_intersection_over_union(detections[slow_loop_idx], detections[fast_loop_idx])
                    if r > cfg.detector.iou_threshold and slow_loop_idx != fast_loop_idx:
                        detections[fast_loop_idx][0] = -1

            for box in detections:
                if box[0] == -1:
                    continue
                detections_iou.append(box)

            # save detections after filter
            detections = detections_iou

            # -----------------------------------------


            # start tracking part
            if len(detections) != 0:

                # init body_bank
                result_arr = [] # ids of detected persons
                result_features = [] # features of detected persons respectively to result_arr idxes
                result_dist_arr = [] # distance of detected persons respectively to result_arr idxes
                body_bank_tmp = body_bank.copy() # body bank copy for next loop to exclude person after search
                len_body_bank_tmp = len(body_bank_tmp) # decreasing length of body_bank_tmp

                # search loop
                for i, body_detection_xy in enumerate(detections):
                    # if body bank (gallary) is not empty
                    if len(body_bank) != 0:

                        # if tmp body bank (gallary) is empty but we still have detected persons
                        if len_body_bank_tmp == 0:


                            # get coordinates of person
                            p = detections[i]
                            p = check_coords(p)

                            # do preparations for image to fit it into reid network
                            img = np.transpose(frameL[int(p[0]):int(p[2]), int(p[1]):int(p[3])]).astype('f') / 255.
                            img = np.expand_dims(img, axis=0)

                            # extract features
                            features_img_query = F.normalize(extract_features(torch.from_numpy(img).cuda())).cpu().numpy()[0]
                            # add features of new pearson to gallary

                            body_bank.append(features_img_query)
                            result = len(body_bank)
                            result_arr.append(result)
                            #body_bank_bb.append(detections[i])
                            # init distance of new person
                            body_bank_dist.append(0)
                            continue


                        # get distance of person to persons in gallery
                        distmat_arr, query_f = get_dist([np.array(body_detection_xy)], frameL, body_bank_tmp)

                        # get id of person in gallery
                        result = np.argmin(distmat_arr, 1)[0]
                        #print(distmat_arr[0][result])

                        # if minimum distance is lower cfg.model.threshold it is the same person
                        if (distmat_arr[0][result] < cfg.model.threshold):
                            # add id to result array
                            result_arr.append(result)

                            # exclude person from search
                            body_bank_tmp[result] = []

                            # decrease lenght of tmp body bank
                            len_body_bank_tmp -= 1

                            # save features and distance
                            result_features.append(query_f[result])
                            result_dist_arr.append(np.min(distmat_arr, 1)[0])
                        else:
                            # save features and distance
                            body_bank.append(query_f[result])
                            body_bank_dist.append(0)

                    else:
                        print("add new user")
                        #result = len(body_bank) + 1

                        # get coordinates of person
                        p = detections[i]
                        p = check_coords(p)

                        # do preparations for image to fit it into reid network
                        img = np.transpose(frameL[int(p[0]):int(p[2]), int(p[1]):int(p[3])]).astype('f') / 255.
                        img = np.expand_dims(img, axis=0)

                        # extract features
                        features_img_query = F.normalize(extract_features(torch.from_numpy(img).cuda())).cpu().numpy()[0]

                        # save features and distance
                        body_bank.append(features_img_query)
                        result = len(body_bank)
                        result_arr.append(result)
                        #body_bank_bb.append(detections[i])
                        body_bank_dist.append(0)

                # ax for visualisation of features
                ax = [0] * len(result_arr)
                if cfg.visualisation.key:
                    if len(body_bank) != 0:
                        fig, ax = plt.subplots(nrows=len(body_bank), ncols=3, gridspec_kw={'width_ratios': [2, 10, 10]}, figsize=(10, 2))

                        if len(body_bank) == 1:
                            ax = [ax]

                for i, pack in enumerate(zip(result_arr, result_dist_arr, ax)):
                    id, result, row = pack
                    # add new human if % is too low

                    # refresh bank
                    if cfg.visualisation.key:
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

                    # refresh image in gallery if image is good
                    if result < cfg.model.threshold:
                        p = detections[i]
                        p = check_coords(p)
                        body_bank_dist[id] = result# + body_bank_dist[id]) / 2
                        body_bank[id] = result_features[i] #+ body_bank[id]) / 2  #frameL[int(p[1]):int(p[3]), int(p[0]):int(p[2])]


                    text = "_ID_{} <{}>".format(str(id), str(round(result, 3)))
                    cv2.putText(frameL, text,
                                (int(detections[i][1]) - 10, int(detections[i][0]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            if cfg.visualisation.key:
                plt.show()
                plt.pause(4)
                plt.close()

            frameL = frameL / 255
            for p in detections:
                cv2.rectangle(frameL, (p[1], p[0]), (p[3], p[2]), (50, 50, 250), 2)

            cv2.imshow("L", frameL)
            #print("shape ", frameL.shape)

            if args_save:
                video_writer.write((frameL * 255).astype(np.uint8))
            cv2.waitKey(1)

if __name__ == '__main__':
    main()
