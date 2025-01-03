import os
import argparse
import tifffile

import numpy as np
import cv2 as cv

from .modules import stitching
from typing import Union


def search_files(file_path, exts):
    files_ = list()
    for root, dirs, files in os.walk(file_path):
        if len(files) == 0:
            continue
        for f in files:
            fn, ext = os.path.splitext(f)
            if ext in exts: files_.append(os.path.join(root, f))

    return files_


def rc_key(row: int, col: int):
    return '{}_{}'.format(str(row).zfill(4), str(col).zfill(4))


def filename2index(file_name, style='motic', row_len=None):
    file_name = os.path.basename(file_name)
    if style.lower() in ['motic', 'cghd']:
        tags = os.path.splitext(file_name)[0].split('_')
        xy = list()
        for tag in tags:
            if (len(tag) == 4) and tag.isdigit():
                if int(tag) < 999:
                    xy.append(tag)
        x_str = xy[0]
        y_str = xy[1]
        return [int(y_str), int(x_str)]
    elif style.lower() == 'zeiss':
        line = os.path.splitext(file_name)[0].split('_')
        c = int(float(line[2]))
        r = int(float(line[1]))
        return [c, r]
    elif style.lower() == "leica dm6b":
        num = file_name.split("_")[1][1:]
        x = int(int(num) / row_len)
        y = int(int(num) % row_len)
        if x % 2 == 1:
            y = row_len - y - 1
        return [y, x]
    else:
        return None


def images_path2dict(images_path, style='motic', row_len=None):
    image_support = ['.jpg', '.png', '.tif', '.tiff', '.TIFF']
    fov_images = search_files(images_path, exts=image_support)
    src_fovs = dict()
    rows = cols = -1
    for it in fov_images:
        col, row = filename2index(it, style=style, row_len=row_len)
        if row > rows: rows = row
        if col > cols: cols = col
        src_fovs[rc_key(row, col)] = it

    return src_fovs, rows + 1, cols + 1


def create_loc(rows, cols, shape, overlap):
    height, width = shape
    fov_loc = np.zeros((rows, cols, 2), dtype=int)
    for i in range(rows):
        for j in range(cols):
            fov_loc[i, j, 0] = j * (width - int(width * overlap))
            fov_loc[i, j, 1] = i * (height - int(height * overlap))

    return fov_loc


def stitch_image(
        image_path: str = '',
        overlap: float = 0.1,
        name: str = '',
        scope_flag: bool = False,
        output_path: str = ''
) -> Union[None, np.ndarray]:
    """
    图像拼接函数
    小图排列格式如下：
    -------------------------
       0_0, 0_1, ... , 0_n
       1_0, 1_1, ... , 1_n
       ...
       m_0, m_1, ... , m_n
    -------------------------
    其中, m 和 n 分别表示 row 和 col

    Args:
        image_path:
        name: 图像命名 可不填
        overlap: 显微镜预设overlap
        scope_flag: 是否直接使用显微镜拼接
        output_path:

    Returns:

    """
    src_fovs, rows, cols = images_path2dict(image_path)  # , style="leica dm6b", row_len=5)

    stitch = stitching.Stitching()
    stitch.set_size(rows, cols)
    stitch.set_overlap(overlap)

    loc = None
    if scope_flag:
        if isinstance(list(src_fovs.items())[0][1], str):
            shape = cv.imread(list(src_fovs.items())[0][1], 0).shape
        else: shape = list(src_fovs.items())[0][1].shape
        loc = create_loc(rows, cols, shape, overlap)

    if loc is not None:
        stitch.set_global_location(loc)

    stitch.stitch(src_fovs)
    image = stitch.get_image()
    #
    if os.path.isdir(output_path):
        if len(name) == 0: name = 'image'
        tifffile.imwrite(os.path.join(output_path, f'{name}_fov_stitched.tif'), image)
    else:
        return image


def main(args, para):
    stitch_image(
        image_path = args.input,
        overlap = args.overlap,
        name = args.name,
        scope_flag = args.scope,
        output_path = args.output
    )


def arg_parser():

    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                        help="Tar file / Input image dir.")
    parser.add_argument("-overlap", "--overlap", action="store", dest="overlap", type=float, required=False,
                        default=0.1, help="Overlap.")
    parser.add_argument("-s", "--scope", action = "store", dest = "scope", type = bool, required = False,
                        default = False, help = "Scope stitch.")
    parser.add_argument("-n", "--name", action="store", dest="name", type=str, required=False,
                        default = '', help="Name.")
    parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=False,
                        default = '', help="Result output dir.")

    parser.set_defaults(func=main)
    (para, args) = parser.parse_known_args()
    para.func(para, args)


if __name__ == "__main__":
    arg_parser()


