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
from mmpose.apis import vis_pose_result_no_limb

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

    name_kpts = [
        'nose', 'head_bottom', 'head_top', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

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

        img_vis = vis_pose_result_no_limb(
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

        for k in range(17):
            path_overlaid = '%s/%s_%s.jpg'%(path_out_dir, path_out_filename[:-4], name_kpts[k])

            heatmap = cv2.resize(returned_outputs[0]['heatmap'][0][k], dsize=(img_input.shape[1], img_input.shape[0]), interpolation=cv2.INTER_LINEAR)
            heatmap[heatmap<0]=0
            heatmap = cv2.cvtColor((heatmap*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)

            image_overlaid = cv2.addWeighted(heatmap, 0.5, img_input, 0.5, 0)
            cv2.imwrite(path_overlaid, image_overlaid)

        if i % (lenImages/5) == 0:
            print('[%5d/%5d] Visualizing keypoints and heatmaps...'%(i, lenImages))


if __name__ == '__main__':
    main()
