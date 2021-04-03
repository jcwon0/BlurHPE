import argparse
import os
import glob

import numpy as np
import cv2
import mmcv
import torch

from mmpose.apis import single_gpu_test
from mmpose.apis import init_pose_model
from mmpose.apis import inference_bottom_up_pose_model
from mmpose.apis import vis_pose_result

def parse_args():
    parser = argparse.ArgumentParser(description='test for visualization')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--input_dir', help='input path')
    parser.add_argument('--output_dir', help='output result file')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    model = init_pose_model(args.config, args.checkpoint, device='cuda:0')

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    args.output_dir += 'vis/'
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # load test images
    listImages = glob.glob(os.path.normpath("%s/*/*.jpg"%(args.input_dir)))
    listImages.sort()
    lenImages = len(listImages)
    print('# of images : %d'%(lenImages))

    for i, path_image in enumerate(listImages):

        img_input = cv2.imread(path_image)

        pose_results, returned_outputs = inference_bottom_up_pose_model(
            model=model,
            img_or_path=img_input,
            return_heatmap=True,
            outputs=None
        )

        img_vis = vis_pose_result(
            model=model,
            img=img_input,
            result=pose_results,
            kpt_score_thr=0.9,
            dataset='BottomUpPoseTrack18Dataset',
            show=False,
            out_file=None)

        # path to save results
        path_out_dir, path_out_filename = os.path.split(path_image[len(args.input_dir):])
        path_out_dir = os.path.join(args.output_dir, path_out_dir)
        if not os.path.exists(path_out_dir):
            os.mkdir(path_out_dir)

        cv2.imwrite(os.path.join(path_out_dir, path_out_filename), img_vis)

        if i % (lenImages/5) == 0:
            print('[%5d/%5d] Visualizing keypoints and heatmaps...'%(i, lenImages))


if __name__ == '__main__':
    main()
