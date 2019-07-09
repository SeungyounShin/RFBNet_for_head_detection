import os
import cv2
import time
import sys


def video_to_frame(opt):

    if os.path.isdir(opt.folder_inputs):
        return False

    os.mkdir(opt.folder_video)
    os.mkdir(opt.folder_inputs)

    vidcap = cv2.VideoCapture(opt.video_path)
    frame_idx = 0

    while True:
        ## Log progress
        if frame_idx % opt.log_step == 0:
            time_v2f_start = time.time()

        ret, frame = vidcap.read()
        if ret == False:
            break

        # rotate every frames
        if opt.rotate == 270:
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 0)
        elif opt.rotate == 90:
            frame = cv2.transpose(frame)
            frame = cv2.flip(frame, 1)
        elif opt.rotate == 180:
            frame = cv2.flip(frame, 0)
            frame = cv2.flip(frame, 1)
        elif opt.rotate == 0:
            frame = frame
        else:
            print('ERROR: ROTATE MUST BE 0/90/180/270')
            print('current rotate: {:3}'.format(opt.rotate))
            sys.exit(1)

        cv2.imwrite(opt.folder_inputs + '%05d'%frame_idx + '.jpg', frame)

        ## Log progress
        if (frame_idx+1) % opt.log_step == 0:
            print('#### FPS {:4.1f} -- v2f #{:4} - #{:4}'
                .format(opt.log_step/(time.time()-time_v2f_start), frame_idx-opt.log_step+1, frame_idx))

        frame_idx += 1

    ## Log progress
    if frame_idx % opt.log_step != 0:
        print('#### FPS {:4.1f} -- v2f #{:4} - #{:4}'
            .format((frame_idx % opt.log_step)/(time.time()-time_v2f_start), frame_idx - frame_idx % opt.log_step, frame_idx-1))

    return True



def frame_to_video(opt, targets):

    folder_targets = opt.folder_video + '/' + targets + '/'

    if os.path.isfile(opt.folder_video + '_' + targets + '.mp4'):
        os.remove(opt.folder_video + '_' + targets + '.mp4')

    files = [f for f in os.listdir(folder_targets) if os.path.isfile(os.path.join(folder_targets, f))]
    ## For sorting the file name properly
    files.sort(key = lambda x: int(x[:-4]))

    img = cv2.imread(folder_targets + files[0])
    out = cv2.VideoWriter(opt.folder_video + '_' + targets + '.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 30.0, (img.shape[1], img.shape[0]))

    for i in range(len(files)):

        ## Log progress
        if i % opt.log_step == 0:
            time_f2v_start = time.time()

        filename = folder_targets + files[i]
        ## Reading each files
        img = cv2.imread(filename)
        out.write(img)

        ## Log progress
        if (i+1) % opt.log_step == 0:
            print('#### FPS {:4.1f} -- f2v #{:4} - #{:4}'
                .format(opt.log_step/(time.time()-time_f2v_start), i-opt.log_step+1, i))

    ## Log progress
    if (i+1) % opt.log_step != 0:
        print('#### FPS {:4.1f} -- f2v #{:4} - #{:4}'
            .format((i % opt.log_step + 1)/(time.time()-time_f2v_start), i - i % opt.log_step, i))

    out.release()
