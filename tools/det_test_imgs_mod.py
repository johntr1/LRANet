#!/usr/bin/env python
import os
import os.path as osp
import glob
from argparse import ArgumentParser

import mmcv
from mmcv.utils import ProgressBar
from mmdet.apis import inference_detector, init_detector

from mmocr.models import build_detector  # noqa: F401
from mmocr.utils import list_to_file


def gen_target_path(target_root_path, src_name, suffix):
    file_name = osp.split(src_name)[-1]
    name = osp.splitext(file_name)[0]
    return osp.join(target_root_path, name + suffix)


def save_results(result, out_dir, img_name, score_thr=0.3):
    assert 'boundary_result' in result
    assert 0 < score_thr < 1

    txt_file = gen_target_path(out_dir, img_name, '.txt')
    valid_boundary_res = [
        res for res in result['boundary_result'] if res[-1] > score_thr
    ]
    lines = [','.join([str(round(x)) for x in row]) for row in valid_boundary_res]
    list_to_file(txt_file, lines)


def main():
    parser = ArgumentParser()
    parser.add_argument('img_dir', type=str, help='Directory with images')
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('checkpoint', type=str, help='Checkpoint file')
    parser.add_argument('--score-thr', type=float, default=0.5, help='Bbox score threshold')
    parser.add_argument('--out-dir', type=str, default='./results', help='Output directory')
    parser.add_argument('--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()

    assert 0 < args.score_thr < 1

    # Build model
    model = init_detector(args.config, args.checkpoint, device=args.device)
    if hasattr(model, 'module'):
        model = model.module
    if model.cfg.data.test['type'] == 'ConcatDataset':
        model.cfg.data.test.pipeline = model.cfg.data.test['datasets'][0].pipeline

    # Prepare output directories
    out_vis_dir = osp.join(args.out_dir, 'out_vis_dir')
    mmcv.mkdir_or_exist(out_vis_dir)
    out_txt_dir = osp.join(args.out_dir, 'out_txt_dir')
    mmcv.mkdir_or_exist(out_txt_dir)

    # Get image list
    exts = ('*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG', '*.JPEG')
    img_paths = []
    for ext in exts:
        img_paths.extend(glob.glob(osp.join(args.img_dir, ext)))
    img_paths = sorted(img_paths)

    progressbar = ProgressBar(task_num=len(img_paths))
    for img_path in img_paths:
        progressbar.update()
        img_name = osp.basename(img_path)
        result = inference_detector(model, img_path)
        save_results(result, out_txt_dir, img_name, score_thr=args.score_thr)
        out_file = osp.join(out_vis_dir, img_name)
        model.show_result(
            img_path,
            result,
            score_thr=args.score_thr,
            show=False,
            out_file=out_file
        )

    print(f'\nInference complete. Results saved in: {args.out_dir}\n')


if __name__ == '__main__':
    main()
