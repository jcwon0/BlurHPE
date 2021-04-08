import argparse
import os
import numpy as np
import cv2
import mmcv

from xtcocotools.coco import COCO

def parse_args():
    parser = argparse.ArgumentParser(description='visualization of ground truth')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--output_dir', help='output result file')
    args = parser.parse_args()
    return args


def main():

    palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                        [230, 230, 0], [255, 153, 255], [153, 204, 255],
                        [255, 102, 255], [255, 51, 255], [102, 178, 255],
                        [51, 153, 255], [255, 153, 153], [255, 102, 102],
                        [255, 51, 51], [153, 255, 153], [102, 255, 102],
                        [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                        [255, 255, 255]])

    skeleton = [[16, 14], [14, 12], [17, 15], [15, 13],
                [12, 13], [6, 12], [7, 13],
                [6, 2], [2, 7],
                [6, 8], [8, 10], [7, 9], [9, 11],
                [1, 2], [1, 3], [1, 4], [1, 5]]

    pose_limb_color = palette[[
        0, 0, 0, 0,
        7, 7, 7,
        9, 9,
        9, 9, 9, 9,
        16, 16, 16, 16
    ]]

    pose_kpt_color = palette[[
        16, 16, 16, 16, 16, 9, 9, 9, 9, 9, 9, 0, 0, 0, 0, 0, 0
    ]]

    radius = 4

    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)

    data_root = cfg['data_root']

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    coco = COCO(cfg['data']['val']['ann_file'])

    img_ids = coco.getImgIds()
    img_ids = [
        img_id for img_id in img_ids
        if len(coco.getAnnIds(imgIds=img_id, iscrowd=None)) > 0
    ]
    num_images = len(img_ids)
    print('# of images : %d'%(num_images))

    for idx in range(num_images):

        img_id = img_ids[idx]

        ann_id = coco.getAnnIds(imgIds=img_id)
        ann_load = coco.loadAnns(ann_id)

        ann = list()
        for ann_k_list in ann_load:
            ann_k_np = np.ndarray([17, 3], dtype=np.float32)
            for i in range(17):
                for j in range(3):
                    ann_k_np[i, j] = ann_k_list['keypoints'][3*i + j]
            ann.append(ann_k_np)

        image_path = coco.loadImgs(img_id)[0]['file_name']
        path_input = os.path.join(data_root, image_path)

        image = mmcv.imread(path_input)
        img_h, img_w, _ = image.shape

        for kpts in ann:
            # draw each point on image
            for kid, kpt in enumerate(kpts):
                x_coord, y_coord, kpt_visible = int(kpt[0]), int(kpt[1]), kpt[2]
                if kpt_visible:
                    r, g, b = pose_kpt_color[kid]
                    cv2.circle(image, (int(x_coord), int(y_coord)), radius, (int(r), int(g), int(b)), -1)

            # draw limbs
            for sk_id, sk in enumerate(skeleton):
                pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                if (pos1[0] > 0 and pos1[0] < img_w and pos1[1] > 0
                        and pos1[1] < img_h and pos2[0] > 0
                        and pos2[0] < img_w and pos2[1] > 0
                        and pos2[1] < img_h
                        and kpts[sk[0] - 1, 2]
                        and kpts[sk[1] - 1, 2]):
                    r, g, b = pose_limb_color[sk_id]
                    cv2.line(
                        image,
                        pos1,
                        pos2, (int(r), int(g), int(b)),
                        thickness=1)

        image_folder, image_name = os.path.split(image_path)
        image_folder = os.path.split(image_folder)[1]
        path_output = os.path.join(args.output_dir, image_folder)
        if not os.path.exists(path_output):
            os.mkdir(path_output)

        cv2.imwrite(os.path.join(path_output, image_name), image)

        if idx % int(num_images/20) == 0:
            print('[%5d/%5d] Drawing ground-truth skeleton...'%(idx, num_images))


if __name__ == '__main__':
    main()
