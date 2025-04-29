# -*- coding: utf-8 -*-
# @Time    : 2024/12/24 17:47
# @Author  : unknow
# @File    : module_compare.py
import os.path

import numpy as np
from .common import wrapper_class
import cv2
from skimage import measure
import tifffile as tif
from glob import glob

import sys
from pathlib import Path
sys.path.append(Path(__file__).parents[2])
from cellbin2.utils.ipr import read

def subtract(a, b):
    if isinstance(a, (bool, np.bool_, np.bool)) or isinstance(b, (bool, np.bool_, np.bool)):
        a, b = map(lambda x: 1 if x else 0, (a,b))
    if isinstance(a, (int, float, np.int64, np.int32)) and isinstance(b, (int, float, np.int64, np.int32)):
        sub = a - b

    else:
        sub = -9999

    return sub


def decode(mask_rle, shape):
    """

    Args:
        mask_rle:
        shape:

    Returns:解析ipr里面的mask

    """
    starts = mask_rle[:, 0]
    lengths = mask_rle[:, 1]
    starts -= 1
    ends = starts + lengths
    binary_mask = np.zeros(int(shape[0]) * int(shape[1]), dtype=np.uint8)
    for lo, hi in zip(starts, ends):
        binary_mask[lo:hi] = 1
    return binary_mask.reshape(shape)


def iou(a, b, epsilon=1e-5):
    """
    add by jqc on 2023/04/10
    Args:
        a ():
        b ():
        epsilon ():

    Returns:

    """
    # 首先将a和b按照0/1的方式量化
    a = (a > 0).astype(int)
    b = (b > 0).astype(int)

    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)

    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)

    # 计算IoU
    iou = intersection / (union + epsilon)

    return iou


def count_iou(mask1: np.ndarray, mask2: np.ndarray):
    h1, w1 = mask1.shape[:2]
    mask2 = cv2.resize(mask2, (w1, h1))
    iou_result = iou(mask2, mask1)
    iou_result = round(iou_result, 6) * 100

    print(iou_result)
    return iou_result


def get_mask_area(mask):
    label = measure.label(mask, connectivity=1)
    props = measure.regionprops(label)
    area_lsit = [p['area'] for p in props]
    cellnum = len(area_lsit)
    mean_area = np.mean(area_lsit)
    median_area = np.median(area_lsit)
    max_area = np.max(area_lsit)
    return cellnum, mean_area, median_area, max_area

class CompareBase:
    comment = ''


class CompareQC(CompareBase):
    def __init__(self, ipr_dict, ipr_dict_):
        self.QcPassFlag_result = subtract(ipr_dict.QCInfo.QCPassFlag, ipr_dict_.QCInfo.QCPassFlag)
        self.TrackCrossQCFlag_result = subtract(ipr_dict.QCInfo.TrackCrossQCPassFlag, ipr_dict_.QCInfo.TrackCrossQCPassFlag)
        self.ChipDetectQCPassFlag_result = subtract(ipr_dict.QCInfo.ChipDetectQCPassFlag, ipr_dict_.QCInfo.ChipDetectQCPassFlag)
        self.TrackLineScore = subtract(ipr_dict.QCInfo.TrackLineScore, ipr_dict_.QCInfo.TrackLineScore)

        if self.ChipDetectQCPassFlag_result or self.TrackCrossQCFlag_result or self.ChipDetectQCPassFlag_result:
            self.comment = 'QC is different'


class CompareChip(CompareBase):
    def __init__(self, ipr_dict, ipr_dict_):
        pass


class CompareTemplate(CompareBase):
    def __init__(self, ipr_dict, ipr_dict_):
        pass
        self.comment = 'Template is different'
        self.Compare_template_score = -9999
        if all(isinstance(x, (int, float, np.int64)) for x in [ipr_dict.QCInfo.TemplateValidArea, ipr_dict.QCInfo.TemplateRecall,
                                                               ipr_dict_.QCInfo.TemplateValidArea, ipr_dict_.QCInfo.TemplateRecall]):
            template_score = max(ipr_dict.QCInfo.TemplateValidArea, ipr_dict.QCInfo.TemplateRecall)
            _template_score = max(ipr_dict_.QCInfo.TemplateValidArea, ipr_dict_.QCInfo.TemplateRecall)

            self.Compare_template_score = subtract(template_score, _template_score)

            if template_score > 0.2 and abs(self.Compare_template_score) < 0.05:
                self.comment = 'Template is same'


class CompareRegist(CompareBase):
    def __init__(self, ipr_dict, ipr_dict_):

        Compare_QCresult = subtract(ipr_dict.QCInfo.QCPassFlag, ipr_dict_.QCInfo.QCPassFlag)
        Compare_offsetX = subtract(ipr_dict.Register.OffsetX, ipr_dict_.Register.OffsetX)
        Compare_offsetY = subtract(ipr_dict.Register.OffsetY, ipr_dict_.Register.OffsetY)
        Compare_CounterRot90 = subtract(ipr_dict.Register.CounterRot90, ipr_dict_.Register.CounterRot90)

        if Compare_QCresult == 0 and (Compare_CounterRot90 == 0) and abs(Compare_offsetX) < 51 and abs(Compare_offsetY) < 51:
            self.comment = 'SAME'
        else:
            self.comment = 'DIFF'


class CompareTissue(CompareBase):
    def __init__(self, ipr_dict, ipr_dict_):
        tissue = decode(ipr_dict.TissueSeg.TissueMask, ipr_dict.TissueSeg.TissueSegShape)
        tissue_ = decode(ipr_dict_.TissueSeg.TissueMask, ipr_dict_.TissueSeg.TissueSegShape)

        self.iou = count_iou(tissue, tissue_)


class CompareTissue2(CompareBase):
    def __init__(self, tissue_dir: str, tissue_dir_: str):
        tissue = tif.imread(tissue_dir)
        tissue_ = tif.imread(tissue_dir_)
        self.iou = count_iou(tissue, tissue_)


class NoneAttribute:
    def __getattr__(self, name):
        return None


class CompareCell(CompareBase):
    def __init__(self, ipr_dict, ipr_dict_):
        mask = decode(ipr_dict.CellSeg.CellMask, ipr_dict.CellSeg.CellSegShape)
        mask_ = decode(ipr_dict_.CellSeg.CellMask, ipr_dict_.CellSeg.CellSegShape)
        self.cellnum, self.mean_area, self.median_area, self.max_area = get_mask_area(mask)
        self.cellnum_, self.mean_area_, self.median_area_, self.max_area_ = get_mask_area(mask_)


class CompareCell2(CompareBase):
    def __init__(self, mask_dir: str, mask_dir_: str):
        mask = tif.imread(mask_dir)
        mask_ = tif.imread(mask_dir_)
        self.cellnum, self.mean_area, self.median_area, self.max_area = get_mask_area(mask)
        self.cellnum_, self.mean_area_, self.median_area_, self.max_area_ = get_mask_area(mask_)


# class CompareModule:
#     def __init__(self, ipr_dict, ipr_dict_):
#         self.ipr_dict = dict2class(ipr_dict)
#         self.ipr_dict_ = dict2class(ipr_dict_)
#         self.CompareQC = CompareQC(self.ipr_dict, self.ipr_dict_)
#         self.CompareChip = CompareChip(self.ipr_dict, self.ipr_dict_)
#         self.CompareTemplate = CompareTemplate(self.ipr_dict, self.ipr_dict_)
#         self.CompareRegist = CompareRegist(self.ipr_dict, self.ipr_dict_)
#         self.CompareTissue = CompareTissue(self.ipr_dict, self.ipr_dict_)
#         self.CompareCell = CompareCell(self.ipr_dict, self.ipr_dict_)
#
#     def __getattr__(self, name):
#         return NoneAttribute()

class CompareModule2:
    stain_list = ['ssDNA', 'DAPI', 'HE']
    stain = ""

    def __init__(self, result1, result2):
        ipr_file = glob(os.path.join(result1, "*.ipr"))
        ipr_file2 = glob(os.path.join(result2, "*.ipr"))

        if len(ipr_file) == 1 and len(ipr_file2) == 1:
            _, image_dct = read(ipr_file[0])
            _, image_dct_ = read(ipr_file2[0])

            for stain in self.stain_list:
                if image_dct.get(stain) and image_dct_.get(stain):
                    self.stain = stain
                    print(f'Main stain is {self.stain}')

            if self.stain == '':
                raise "Without same main stain in two ipr "

            self.ipr_dict = wrapper_class(image_dct[self.stain])
            self.ipr_dict_ = wrapper_class(image_dct_[self.stain])
            self.CompareQC = CompareQC(self.ipr_dict, self.ipr_dict_)
            self.CompareChip = CompareChip(self.ipr_dict, self.ipr_dict_)
            self.CompareTemplate = CompareTemplate(self.ipr_dict, self.ipr_dict_)
            self.CompareRegist = CompareRegist(self.ipr_dict, self.ipr_dict_)

        tissue1 = os.path.join(result1, f'{self.ipr_dict.ImageInfo.STOmicsChipSN}_{self.ipr_dict.QCInfo.StainType}_tissue_cut.tif')
        tissue2 = os.path.join(result2, f'{self.ipr_dict_.ImageInfo.STOmicsChipSN}_{self.ipr_dict_.QCInfo.StainType}_tissue_cut.tif')

        if os.path.exists(tissue1) and os.path.exists(tissue2):
            self.CompareTissue = CompareTissue2(tissue1, tissue2)

        mask1 = os.path.join(result1, f'{self.ipr_dict.ImageInfo.STOmicsChipSN}_{self.ipr_dict.QCInfo.StainType}_mask.tif')
        mask2 = os.path.join(result2, f'{self.ipr_dict_.ImageInfo.STOmicsChipSN}_{self.ipr_dict_.QCInfo.StainType}_mask.tif')

        if os.path.exists(mask1) and os.path.exists(mask2):
            self.CompareCell = CompareCell2(mask1, mask2)

    def __getattr__(self, name):
        return NoneAttribute()


def module_compare_pipeline(result1, result2):
    cpm = CompareModule2(result1, result2)
    result_is_same = True
    result_dict = dict()

    if cpm.CompareQC.QcPassFlag_result == -1:
        result_is_same = False
        print('QC pass before, fail now')
        result_dict['QC'] = 'failed'

    elif cpm.CompareQC.QcPassFlag_result == 1:
        result_dict['QC'] = 'Better'

    elif cpm.CompareQC.QcPassFlag_result == 0:
        result_dict['QC'] = 'SAME'

    else:
        result_dict['QC'] = 'ERROR'

    result_dict['QC_score_change'] = cpm.CompareQC.TrackLineScore

    if cpm.CompareRegist.comment == 'DIFF':
        result_is_same = False
        print('Regist param is different')

    result_dict['Regist_use_method'] = f'{cpm.ipr_dict.Register.Method}_{cpm.ipr_dict_.Register.Method}'

    result_dict['Tissue_iou'] = cpm.CompareTissue.iou

    result_dict['cellnum'] = f'{cpm.CompareCell.cellnum}_{cpm.CompareCell.cellnum_}'
    result_dict['cell_area'] = f'{cpm.CompareCell.median_area}_{cpm.CompareCell.median_area_}'

    return result_is_same, result_dict



