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
    parser.add_argument('--v_kpt', type=bool, default=False, help='visualization of keypoint')
    parser.add_argument('--v_limb', type=bool, default=False, help='visualization of limb')
    parser.add_argument('--v_hmap', type=bool, default=False, help='visualization of heatmap')
    parser.add_argument('--th_kpt', type=float, default=0.3, help='threshold for keypoint')
    parser.add_argument('--input_dir', help='input path')
    parser.add_argument('--output_dir', help='output result file')
    args = parser.parse_args()
    return args


def main():

    args = parse_args()

    print('vis. for keypoints : ', args.v_kpt)
    print('vis. for limbs : ', args.v_limb)
    print('vis. for heatmaps : ', args.v_hmap)
    print('keypoint threshold : ', args.th_kpt)

    if not (args.v_kpt or args.v_limb or args.v_hmap):
        print('At least one option for visualization should be True.')
        exit()

    # posetrack
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
            return_heatmap=args.v_hmap,
            outputs=None
        )

        if args.v_limb:
            img_vis_limb = vis_pose_result(
                model=model,
                img=img_input,
                result=pose_results,
                kpt_score_thr=args.th_kpt,
                dataset='BottomUpPoseTrack18Dataset',
                show=False,
                out_file=None)

        if args.v_kpt or args.v_hmap:
            img_vis = vis_pose_result_no_limb(
                model=model,
                img=img_input,
                result=pose_results,
                kpt_score_thr=args.th_kpt,
                dataset='BottomUpPoseTrack18Dataset',
                show=False,
                out_file=None)

        # path to save results
        path_out_dir, path_out_filename = os.path.split(path_image[len(args.input_dir):])
        path_out_dir = os.path.join(args.output_dir, path_out_dir)
        if not os.path.exists(path_out_dir):
            os.mkdir(path_out_dir)

        if args.v_kpt:
            path_kpt = os.path.join(path_out_dir, 'keypoint')
            if not os.path.exists(path_kpt):
                os.mkdir(path_kpt)

            cv2.imwrite(os.path.join(path_kpt, path_out_filename[:-4] + '_keypoint.jpg'), img_vis)

        if args.v_limb:
            path_limb = os.path.join(path_out_dir, 'limb')
            if not os.path.exists(path_limb):
                os.mkdir(path_limb)

            cv2.imwrite(os.path.join(path_limb, path_out_filename[:-4] + '_limb.jpg'), img_vis_limb)

        if args.v_hmap:
            path_hmap = os.path.join(path_out_dir, 'heatmap')
            if not os.path.exists(path_hmap):
                os.mkdir(path_hmap)

            for k in range(17):
                heatmap = cv2.resize(returned_outputs[0]['heatmap'][0][k], dsize=(img_input.shape[1], img_input.shape[0]), interpolation=cv2.INTER_LINEAR)
                heatmap[heatmap<0]=0
                heatmap = cv2.cvtColor((heatmap*255).astype(np.uint8),cv2.COLOR_GRAY2RGB)
                img_overlaid = cv2.addWeighted(heatmap, 0.5, img_input, 0.5, 0)

                path_overlaid = '%s/%s_%s.jpg'%(path_hmap, path_out_filename[:-4], name_kpts[k])
                cv2.imwrite(path_overlaid, img_overlaid)

        if i % int(lenImages/20) == 0:
            print('[%5d/%5d] Bottom-up inferencing for visualization...'%(i, lenImages))


if __name__ == '__main__':
    main()
