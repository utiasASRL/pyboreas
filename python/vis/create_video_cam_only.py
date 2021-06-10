import argparse
import cv2
import numpy as np
import os
from tqdm import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/data1/2020_12_01/', type=str,
                        help='location of root folder. Camera images are located under root+camera/')
    parser.add_argument('--extension', default='png', type=str, help='Extension of the image files.')
    args = parser.parse_args()

    files = os.listdir(args.root + 'camera/')
    img_files = [f for f in files if args.extension in f]
    img_files.sort()

    H = 1080
    W = 1920
    upper_crop = 240
    h_before_resize = 1377
    frame_rate = 30

    out = cv2.VideoWriter(args.root + 'video.mp4', cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), frame_rate, (W, H))

    for i in tqdm(range(len(img_files)), desc="Extracting"):
        frame = cv2.imread(args.root + 'camera/' + img_files[i])
        frame = frame[upper_crop:upper_crop+h_before_resize]
        frame = cv2.resize(frame, (W, H))
        out.write(frame)

    out.release()
    cv2.destroyAllWindows()
