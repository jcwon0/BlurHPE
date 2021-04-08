import argparse
import os
import glob

import numpy as np
import cv2
import mmcv
import torch

# from mmpose.apis import single_gpu_test
from mmpose.apis import init_pose_model
from mmpose.apis import inference_top_down_pose_model
from mmpose.apis import vis_pose_result
from mmpose.apis import vis_pose_result_no_bbox
from mmdet.apis import init_detector
from mmdet.apis import inference_detector

def parse_args():
    parser = argparse.ArgumentParser(description='test for visualization')
    parser.add_argument('det_config', help='Config file for detection')
    parser.add_argument('det_checkpoint', help='Checkpoint file for detection')
    parser.add_argument('pose_config', help='Config file for pose')
    parser.add_argument('pose_checkpoint', help='Checkpoint file for pose')
    parser.add_argument('--v_bbox', default=False, help='visualization of person bounding box')
    parser.add_argument('--th_kpt', type=float, default=0.3, help='threshold for keypoint')
    parser.add_argument('--input_dir', help='input path')
    parser.add_argument('--output_dir', help='output result file')
    args = parser.parse_args()
    return args

def process_mmdet_results(mmdet_results, cat_id=0):
    """Process mmdet results, and return a list of bboxes.

    :param mmdet_results:
    :param cat_id: category id (default: 0 for human)
    :return: a list of detected bounding boxes
    """
    if isinstance(mmdet_results, tuple):
        det_results = mmdet_results[0]
    else:
        det_results = mmdet_results

    bboxes = det_results[cat_id]

    person_results = []
    for bbox in bboxes:
        person = {}
        person['bbox'] = bbox
        person_results.append(person)

    return person_results

def main():

    args = parse_args()

    model_bbox = init_detector(args.det_config, args.det_checkpoint, device='cuda:0')
    model_pose = init_pose_model(args.pose_config, args.pose_checkpoint, device='cuda:0')

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

        mmdet_results = inference_detector(model_bbox, img_input)
        person_results = process_mmdet_results(mmdet_results)

        pose_results, returned_outputs = inference_top_down_pose_model(
            model_pose,
            img_input,
            person_results,
            bbox_thr=0.4,
            format='xyxy',
            dataset='TopDownPoseTrack18dataset',
            return_heatmap=False,
            outputs=None)

        img_vis = vis_pose_result_no_bbox(
            model_pose,
            img_input,
            pose_results,
            dataset='TopDownPoseTrack18dataset',
            kpt_score_thr=args.th_kpt,
            show=False)

        if args.v_bbox:
            img_vis_bbox = vis_pose_result(
                model_pose,
                img_input,
                pose_results,
                dataset='TopDownPoseTrack18dataset',
                kpt_score_thr=args.th_kpt,
                show=False)

        # path to save results
        path_out_dir, path_out_filename = os.path.split(path_image[len(args.input_dir)-1:])
        path_out_dir = os.path.join(args.output_dir, path_out_dir)
        if not os.path.exists(path_out_dir):
            os.mkdir(path_out_dir)

        if args.v_bbox:
            path_bbox = os.path.join(path_out_dir, 'with_bbox')
            if not os.path.exists(path_bbox):
                os.mkdir(path_bbox)
            cv2.imwrite(os.path.join(path_bbox, path_out_filename[:-4] + '_bbox.jpg'), img_vis_bbox)

        path_nobox = os.path.join(path_out_dir, 'without_bbox')
        if not os.path.exists(path_nobox):
            os.mkdir(path_nobox)
        cv2.imwrite(os.path.join(path_nobox, path_out_filename), img_vis)

        if i % int(lenImages/20) == 0:
            print('[%5d/%5d] Top-down inferencing for visualization...'%(i, lenImages))


if __name__ == '__main__':
    main()
