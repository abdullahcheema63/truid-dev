import cv2
from django.shortcuts import render
from django.http import JsonResponse
import base64
from django.core.files.base import ContentFile
from django.core.files.storage import default_storage
from django.conf import settings

import re
from datetime import datetime
from card_classifier.classifier_app.models import *
from easydict import EasyDict as edict
import yaml
from rest_framework.decorators import api_view
import numpy as np
import stringdist

from yolov5.utils.torch_utils import time_synchronized
from yolov5.utils.general import check_img_size, scale_coords
from yolov5.utils.datasets import letterbox
import torch
import pickle


@api_view(['POST'])
def get_config(request):
    # if request.method == "POST":
    f = request.FILES['sentFile']  # here you get the files needed
    response = {}
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    date_new = date.replace("-", "_")
    date_new = date_new.replace(" ", "_")
    date_new = date_new.replace(":", "_")
    file_name = 'reference_frame_' + date_new + '.png'
    file_name_2 = default_storage.save(file_name, f)
    print(file_name_2)
    file_url = default_storage.url(file_name_2)

    result = settings.OCR.ocr('.' + file_url, cls=True)
    result_new = []
    for rr in result:
        rr[0] = np.array(rr[0]).astype(np.int64).tolist()
        ss = (rr[1][0], np.float64(rr[1][1]))
        result_new.append([rr[0], ss])
    # boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    # print(txts)
    text = ' '.join(txts)
    # print(text)
    del_words = []
    for n in re.findall(r'[\u4e00-\u9fff]+', text):
        # print(n)
        del_words.append(n)
    for word in del_words:
        if word in text:
            # print(word)
            text = text.replace(word, "")

    predictions = settings.FASTTEXT_MODEL.predict(text)
    # predictions[0][0] = None
    # print(predictions[0])
    config_file = settings.CFG_SELECTION_DICT[predictions[0][0]]
    # config_file = None
    if config_file is not None:
        config = open(config_file)
        config = edict(yaml.full_load(config))
        # print(CFG_SELECTION_DICT)
        # print("results", result_new)
    else:
        config = None
    response['OCR_Results'] = result_new
    response['card_type'] = predictions[0][0]
    response['config_dict'] = config
    # response['session'] = Session
    session = Session()
    session.start_time = date
    session.reference_frame = file_name_2
    session.config_file = config
    session.OCR_results = result_new
    session.save()
    return JsonResponse(response)


@api_view(['POST'])
def extract_data(request):
    f = request.POST['session_id']
    response = {}
    session = Session.objects.get(pk=f)

    results = session.OCR_results
    config = session.config_file
    # print(type(config))
    OCR_params = config["keys"]
    # print(config)
    # print(OCR_params)
    # response['config'] = config
    data = extra_data_from_ocr(results, OCR_params)
    # print(data)
    response["Extracted_Data"] = data
    extracted_data = extractedData()
    extracted_data.data = data
    extracted_data.session = session
    extracted_data.save()
    return JsonResponse(response)


@api_view(['POST'])
def authenticate(request):
    image_file = request.FILES['rectified_image']  # here you get the files needed
    session_id = request.POST['session_id']
    response = {}
    # print(f)
    image = cv2.imdecode(np.fromstring(image_file.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    session = Session.objects.get(pk=session_id)
    config = session.config_file

    variance, flag_hologram_auth = flag_hologram_detection(image, config, session)
    circle_hologram_auth = detect_circle_hologram(image, config, session)
    # print(variance)
    # if flag_det:
    response["authenticate_flag_hologram"] = flag_hologram_auth
    response["flag_variance"] = variance
    response["authenticate_circle_hologram"] = circle_hologram_auth
    # print(type(file_name_2))
    # file_url = default_storage.url(file_name_2)
    return JsonResponse(response)


def flag_hologram_detection(image, config, session):
    Flag_hologram_params = config["security_parameters"]["flag_hologram"]
    detect_flag_hologram = Flag_hologram_params["detect_hologram"]
    variance_threshold = config["security_parameters"]["flag_hologram"]["variance_threshold"]
    if detect_flag_hologram:
        roi_left = Flag_hologram_params["roi_left"]
        roi_right = Flag_hologram_params["roi_right"]
        roi_top = Flag_hologram_params["roi_top"]
        roi_bottom = Flag_hologram_params["roi_bottom"]
        # variance_threshold = Flag_hologram_params["variance_threshold"]
        roi_image = image[roi_top:roi_bottom, roi_left:roi_right, :]
        detected_box, f_vect = detect(settings.YOLO_V5, roi_image)
        # flag_det = None
        if detected_box is not None:
            # boxed_image = cv2.rectangle(roi_image, (detected_box[0], detected_box[1]),
            #                             (detected_box[2], detected_box[3]), (0, 255, 0), 3)

            # file_name_2 = cv2.imwrite(settings.MEDIA_ROOT + "flag_detected.png", boxed_image)
            flag_det = roi_image[detected_box[1]:detected_box[3], detected_box[0]:detected_box[2], :]
            flag_det = cv2.resize(flag_det, (100, 70))
            flag_hologram_db = flagHologram.objects.filter(session=session)
            if len(flag_hologram_db) < 10:
                flag_det = pickle.dumps(flag_det)
                flag_det = base64.b64encode(flag_det)
                flag_hologram_obj = flagHologram()
                flag_hologram_obj.flag_image = flag_det
                flag_hologram_obj.session_variance = 0
                flag_hologram_obj.session = session
                flag_hologram_obj.save()

                return 0, False
            elif len(flag_hologram_db) >= 10:
                print(len(flag_hologram_db))
                flag_image_query = flag_hologram_db.values("flag_image")
                # print(flag_image_query)
                flag_images = [a_dict['flag_image'] for a_dict in flag_image_query]
                # print(flag_images)
                images = []
                for i,flag_image in enumerate(flag_images):
                    print(i)
                    flag_image = base64.b64decode(flag_image)
                    flag_image = pickle.loads(flag_image)
                    images.append(flag_image)

                images = np.array(images)
                mean_image = images.mean(axis=0).astype(np.uint8)

                distance_query = flag_hologram_db.values("distance")
                distances = [a_dict['distance'] for a_dict in distance_query]
                distances = np.array(distances)
                variances = flag_hologram_db.values("session_variance")
                variances = [a_dict['session_variance'] for a_dict in variances]
                variance = variances[-1]
                if variance >= variance_threshold:
                    return variance, True

                res = cv2.matchTemplate(flag_det, mean_image, cv2.TM_SQDIFF_NORMED)

                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
                print('distance ', min_val)
                variance = distances.var()
                flag_hologram_obj = flagHologram()
                flag_hologram_obj.session = session
                flag_det = pickle.dumps(flag_det)
                flag_det = base64.b64encode(flag_det)
                flag_hologram_obj.flag_image = flag_det
                flag_hologram_obj.distance = min_val
                flag_hologram_obj.session_variance = variance
                flag_hologram_obj.save()
                return variance, False


### Circular Hologram Detection
def detect_circle_hologram(image, config, session):
    security_params = config["security_parameters"]["circle_hologram"]
    cfg_num_detections = security_params["num_detections"]
    cfg_detection_ratio = security_params["detection_ratio"]
    cfg_min_radius = security_params["min_circle_radius"]
    cfg_max_radius = security_params["max_circle_radius"]
    cfg_circle_box_xmin = security_params["circle_box_xmin"]
    cfg_circle_box_ymin = security_params["circle_box_ymin"]
    cfg_circle_box_xmax = security_params["circle_box_xmax"]
    cfg_circle_box_ymax = security_params["circle_box_ymax"]
    cfg_circle_center_x = security_params["circle_centre_x"] - cfg_circle_box_xmin
    # cfg_circle_center_x = (556+832)/2 -556
    cfg_circle_center_y = security_params["circle_centre_y"] - cfg_circle_box_ymin
    # cfg_circle_center_y = (235 + 495) / 2 -235
    cfg_distance_thresh = security_params["distance_thresh"]
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])

    # convert the YUV image back to RGB format
    img_bgr = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    img_bgr = img_bgr[cfg_circle_box_ymin:cfg_circle_box_ymax, cfg_circle_box_xmin:cfg_circle_box_xmax]
    # img_bgr = cv2.resize(img_bgr, None, fx = (1/2.25), fy=(1/2.25), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.medianBlur(gray, 5)
    # cv2.imshow('', gray)
    rows = gray.shape[0]
    # cv2.imshow('gray', cv2.resize(gray, None, fx=0.25, fy=0.25))
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, rows / 8,
                               param1=50, param2=25,  ## param1=50, param2=25
                               minRadius=cfg_min_radius, maxRadius=cfg_max_radius)
    circle_detected = False
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            euclidean_distance = np.sqrt(((i[0] - cfg_circle_center_x) ** 2) + ((i[1] - cfg_circle_center_y) ** 2))
            if euclidean_distance < cfg_distance_thresh:
                # circle center
                # img_bgr = cv2.circle(img_bgr, center, 1, (0, 100, 100), 3)
                # circle outline
                # radius = i[2]
                # img_bgr = cv2.circle(img_bgr, center, radius, (255, 0, 255), 3)
                circle_detected = True
    circle_hologram_db = circleHologram.objects.filter(session=session)
    print(len(circle_hologram_db))
    if len(circle_hologram_db) == 0:
        circle_hologram = circleHologram()
        circle_hologram.session = session
        circle_hologram.frame_number = 1
        if circle_detected:
            circle_hologram.detection_number = 1
        else:
            circle_hologram.detection_number = 0
        circle_hologram.save()
    else:
        print(circle_hologram_db.latest('id'))
        circle_hologram = circleHologram()
        circle_hologram.session = session
        circle_hologram.frame_number = circle_hologram_db.latest('id').frame_number + 1
        if circle_detected:
            circle_hologram.detection_number = circle_hologram_db.latest('id').detection_number + 1
        else:
            circle_hologram.detection_number = circle_hologram_db.latest('id').detection_number
        circle_hologram.save()
        print(circle_hologram_db.latest('id').frame_number)
    if circle_hologram.detection_number < cfg_num_detections:
        return False
    else:
        ratio = circle_hologram.detection_number / circle_hologram.frame_number
        if ratio <= cfg_detection_ratio:
            return True
        else:
            return False
    # return circle_detected


def detect(model, img0, img_size=224, stride=32):
    t1 = time_synchronized()
    img = letterbox(img0, img_size, stride=stride)[0]

    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    device = 'cpu'

    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(img_size, s=stride)  # check img_size

    img = torch.from_numpy(img).to(device)
    img = img.float()  # uint8
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # Inference
    pred, fvect = model(img, augment=False)
    if pred is not None:
        det = pred[0]
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        t2 = time_synchronized()
        return det[0, :].numpy().astype(np.int), fvect
    else:
        return None, None


def get_point(box, string):
    if string == 'top_left':
        point = box[0]
    elif string == 'top_right':
        point = box[1]
    elif string == 'bottom_right':
        point = box[2]
    elif string == 'bottom_left':
        point = box[3]
    else:
        print("point string name is incorrect")
    return point


def bb_intersection(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    return interArea


def absolute_key_dict(keys, result):
    DB = []
    for key, value in keys.items():
        key_word = value['key']
        boxA = value['value_box']
        temp = []
        for idx, line in enumerate(result):
            x = []
            y = []
            txt = line[1][0]
            box = line[0]
            for loc in box:
                x.append(loc[0])
                y.append(loc[1])
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)
            boxB = (xmin, ymin, xmax, ymax)
            iou = bb_intersection(boxA, boxB)
            temp.append([key_word, box, txt, iou])
        temp = np.array(temp, dtype=object)
        max_iou_box_ind = np.argmax(temp[:, -1])
        DB.append(temp[max_iou_box_ind, :])
        print('')

    # rem_results = [v for i, v in enumerate(result) if i not in del_idx]

    missing_keys_dict = {}
    for key in DB:
        key_word = key[0]
        txt = key[2]
        missing_keys_dict[key_word] = txt
    return missing_keys_dict


def relative_keys_dict(keys, result):
    DB = []
    del_idx = []
    for key, value in keys.items():
        key_word = value['key']
        lev_dist = value['lev']
        ref_point = value['reference_point']
        value_search_point = value['value_point_search']
        dist_from_key_x = value['dist_from_key_x']
        dist_from_key_y = value['dist_from_key_y']
        for idx, line in enumerate(result):
            txt = line[1][0]
            if stringdist.levenshtein(txt, key_word) <= lev_dist:
                DB.append([key_word, line[0], ref_point, value_search_point, dist_from_key_y, dist_from_key_y])
                del_idx.append(idx)
                break
            else:
                continue

    rem_results = [v for i, v in enumerate(result) if i not in del_idx]

    avail_keys_dict = {}
    for item in DB:
        key_word = item[0]
        box = item[1]
        ref_point = item[2]  ## reference point of key box
        ref_point = get_point(box, ref_point)  ## reference point of key box

        dist_from_key_x = item[4]
        dist_from_key_y = item[5]
        for line in rem_results:
            value_search_point = item[3]  ## This point to be searched wrt reference point
            value_search_point = get_point(line[0],
                                           value_search_point)  ## This point to be searched wrt reference point
            if (value_search_point[1] >= ref_point[1] + dist_from_key_y - 10) and \
                    (value_search_point[1] <= ref_point[1] + dist_from_key_y + 10):
                if (value_search_point[0] >= ref_point[0] + dist_from_key_x - 30) and \
                        (value_search_point[0] <= ref_point[0] + dist_from_key_x + 30):
                    print(key_word)
                    print(line[1][0])
                    avail_keys_dict[key_word] = line[1][0]
                else:
                    continue
    return avail_keys_dict, rem_results


def extra_data_from_ocr(ocr_results, OCR_PARAMS):
    keys = OCR_PARAMS
    available_keys = {}
    missing_keys = {}
    for key, value in keys.items():
        key_available = value['key_available']
        if key_available:
            available_keys[key] = value
        else:
            missing_keys[key] = value
    print('')

    if available_keys is not None:
        available_keys_extracted, results = relative_keys_dict(available_keys, ocr_results)
    else:
        available_keys_extracted = {}
    if missing_keys is not None:
        missing_keys_extracted = absolute_key_dict(missing_keys, results)
        # for key, value in missing_keys.items():
        #     box = value['value_box']
        #     image = cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)

    else:
        missing_keys_extracted = {}

    dictionary = available_keys_extracted.copy()
    dictionary.update(missing_keys_extracted)
    print(available_keys_extracted)
    print(missing_keys_extracted)

    return dictionary
