import time
import os
import json
import cv2
import shutil
import numpy as np


def generate_palette(len):

    color_palette = []

    for i in range(len):
        color_palette.append(list(np.random.choice(range(256), size=3)))

    return color_palette



def color_picker(yolo_class_list, color_palette, object_name):

    if object_name == 'person':
        return [0, 255, 255]
    elif object_name == 'face':
        return [0, 255, 0]
    elif object_name == 'head':
        return [255, 0, 0]
    else:
        return color_palette[yolo_class_list.index(object_name)]



def draw_box(img, box, object_name, thick, color):

    color = [int(s) for s in color]

    bbox = box['bbox']
    score = box['score']
    [x1, y1, x2, y2] = list(map(float, bbox[0:4]))
    ## Draw bbox
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, thick)
    ## Draw text
    cv2.putText(img, '%.3f' % score, (int(x1), int(y1) + 30), 0, 1e-3 * img.shape[1], color, thick // 3)
    ## Draw object_name
    if object_name != 'person' and object_name != 'face':
        cv2.putText(img, object_name, (int(x1), int(y1) - 15), 0, 1e-3 * img.shape[1], color, thick // 3)



def visualize_json(opt):

    if os.path.isdir(opt.folder_detects):
        shutil.rmtree(opt.folder_detects)
    os.mkdir(opt.folder_detects)

    json_data = open(opt.folder_video + '.json').read()
    data = json.loads(json_data)

    frame_first_idx = 0
    frame_last_idx = int(list(data.keys())[-1])

    ## Read first image from inputs folder
    img = cv2.imread(opt.folder_inputs + '%05d.jpg' % (frame_first_idx))
    thick = int((img.shape[0]+img.shape[1])/300)
    color_palette = generate_palette(len(opt.yolo_class_list))

    for frame_idx in range(frame_first_idx, frame_last_idx+1):

        ## Log progress
        if frame_idx % opt.log_step == 0:
            time_visualize_start = time.time()

        ## read image from inputs folder
        img = cv2.imread(opt.folder_inputs + '%05d.jpg' % (frame_idx))

        if frame_idx in list(map(int, data.keys())):
            for object_name in data[str(frame_idx)]:
                for i, box in enumerate(data[str(frame_idx)][object_name]):
                    draw_box(img, box, object_name, thick, color_picker(opt.yolo_class_list, color_palette, object_name))

        cv2.imwrite(opt.folder_detects + '%05d.jpg' % (frame_idx), img)

        ## Log progress
        if (frame_idx+1) % opt.log_step == 0:
            print('#### FPS {:4.1f} -- visualize #{:4} - #{:4}'
                .format(opt.log_step/(time.time()-time_visualize_start), frame_idx-opt.log_step+1, frame_idx))

    ## Log progress
    if (frame_idx+1) % opt.log_step != 0:
        print('#### FPS {:4.1f} -- visualize #{:4} - #{:4}'
            .format((frame_idx % opt.log_step + 1)/(time.time()-time_visualize_start), frame_idx - frame_idx % opt.log_step, frame_idx))



def extract_face(opt):

    if os.path.isdir(opt.folder_faces):
        shutil.rmtree(opt.folder_faces)
    os.mkdir(opt.folder_faces)

    json_data = open(opt.folder_video + '.json').read()
    data = json.loads(json_data)

    frame_first_idx = 0
    frame_last_idx = int(list(data.keys())[-1])

    for frame_idx in range(frame_first_idx, frame_last_idx+1):

        ## Log progress
        if frame_idx % opt.log_step == 0:
            time_extract_face_start = time.time()

        ## read image from inputs folder
        img = cv2.imread(opt.folder_inputs + '%05d.jpg' % (frame_idx))

        if frame_idx in list(map(int, data.keys())):
            if 'face' in data[str(frame_idx)]:
                for i, box in enumerate(data[str(frame_idx)]['face']):
                    bbox = box['bbox']
                    img_face = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    cv2.imwrite(opt.folder_faces + '{:04d}_{:02d}.jpg'.format(frame_idx, i), img_face)

        ## Log progress
        if (frame_idx+1) % opt.log_step == 0:
            print('#### FPS {:5.2f} -- extract_face #{:4} - #{:4}'
                .format(opt.log_step/(time.time()-time_extract_face_start), frame_idx-opt.log_step+1, frame_idx))

    ## Log progress
    if (frame_idx+1) % opt.log_step != 0:
        print('#### FPS {:5.2f} -- extract_face #{:4} - #{:4}'
            .format((frame_idx % opt.log_step + 1)/(time.time()-time_extract_face_start), frame_idx - frame_idx % opt.log_step, frame_idx))
