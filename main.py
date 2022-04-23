#!/usr/bin/env python3

import cv2
import os
import sys
from imaging_interview import compare_frames_change_detection
from imaging_interview import preprocess_image_change_detection


def main():
    # Initialization
    folder_path = sys.argv[-1]
    files_in_folder = os.listdir(folder_path)
    score_threshold = 2000
    counter = 0
    new_sizes = (640, 480)  # (w, h)
    pivot_img, sample_img = None, None

    for file in files_in_folder:
        # Checking only png files
        if file.endswith('.png'):

            # In first iteration, take first file as pivot
            if pivot_img is None:
                pivot_img = cv2.imread(os.path.join(folder_path, file))
                pivot_img_preprocess = preprocess_image_change_detection(pivot_img, [5])
                pivot_img_preprocess_resize = cv2.resize(pivot_img_preprocess, new_sizes,
                                                         interpolation=cv2.INTER_LINEAR)
                continue

            # In next iterations, take another file as sample and compute score
            sample_img = cv2.imread(os.path.join(folder_path, file))
            if sample_img is None:
                continue
            sample_img_preprocess = preprocess_image_change_detection(sample_img, [5])
            sample_img_preprocess_resize = cv2.resize(sample_img_preprocess, new_sizes, interpolation=cv2.INTER_LINEAR)

            score, res_cnts, thresh = compare_frames_change_detection(pivot_img_preprocess_resize,
                                                                      sample_img_preprocess_resize, 100)

            counter += 1
            print("File number = {} , score = {}, file name = {}".format(counter, score, file))

            # If score is small, delete file, otherwise replace pivot image
            if score > score_threshold:
                pivot_img_preprocess_resize = sample_img_preprocess_resize
            else:
                os.remove(os.path.join(folder_path, file))


if __name__ == '__main__':
    main()
