# -*- coding: utf-8 -*-
# @Time    : 2024/12/24 17:47
# @Author  : unknow
# @File    : module_compare.py

import numpy as np
from .common import dict2class
import cv2
from skimage import measure


def subtract(a, b):
    if isinstance(a, (bool, np.bool_, np.bool)) or isinstance(b, (bool, np.bool_, np.bool)):
        a, b = map(lambda x: 1 if x else 0, (a,b))
    if isinstance(a, (int, float, np.int64)) and isinstance(b, (int, float, np.int64)):
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
    iou_result = round(iou_result, 4) * 100

    print(iou_result)
    return iou_result


def get_mask_area(mask):
    label = measure.label(mask, connectivity=4)
    props = measure.regionprops(label)
    area_lsit = [p['area'] for p in props]
    cellnum = len(area_lsit)
    mean_area = np.mean(area_lsit)
    median_area = np.median(area_lsit)
    max_area = np.max(area_lsit)
    return cellnum, mean_area, median_area, max_area


class CompareQC(object):
    def __init__(self, ipr_dict: dict2class, _ipr_dict: dict2class):
        self.comment = 'QC is same!'
        self.QcPassFlag_result = subtract(ipr_dict.QcPassFlag, _ipr_dict.QcPassFlag)  # 目前暂时还是先用这个
        self.TrackCrossQCFlag_result = subtract(ipr_dict.TrackCrossQCPassFlag, _ipr_dict.TrackCrossQCPassFlag)
        self.ChipDetectQCPassFlag_result = subtract(ipr_dict.ChipDetectQCPassFlag,_ipr_dict.ChipDetectQCPassFlag)
        self.TrackLineScore = subtract(ipr_dict.TrackLineScore, _ipr_dict.TrackLineScore)

        if self.ChipDetectQCPassFlag_result or self.TrackCrossQCFlag_result or self.ChipDetectQCPassFlag_result:
            self.comment = 'QC is different'


class CompareChip(object):
    def __init__(self, ipr_dict: dict2class, _ipr_dict: dict2class):
        self.comment = 'Chip detect is same'
        pass


class CompareTemplate(object):
    def __init__(self, ipr_dict: dict2class, _ipr_dict: dict2class):
        pass
        self.comment = 'Template is different'
        self.Compare_template_score = -9999
        if all(isinstance(x, (int, float, np.int64)) for x in [ipr_dict.TemplateValidArea, ipr_dict.TemplateRecall,
                                                               _ipr_dict.TemplateValidArea, _ipr_dict.TemplateRecall]):
            template_score = max(ipr_dict.TemplateValidArea, ipr_dict.TemplateRecall)
            _template_score = max(_ipr_dict.TemplateValidArea, _ipr_dict.TemplateRecall)

            self.Compare_template_score = subtract(template_score, _template_score)

            if template_score > 0.2 and abs(self.Compare_template_score) < 0.05:
                self.comment = 'Template is same'


class CompareRegist(object):
    def __init__(self, ipr_dict: dict2class, _ipr_dict: dict2class):
        self.comment = 'Register is Same'

        Compare_QCresult = subtract(ipr_dict.QcPassFlag, _ipr_dict.QcPassFlag)  # 这里的QC先用小c
        Compare_offsetX = subtract(ipr_dict.OffsetX, _ipr_dict.OffsetX)
        Compare_offsetY = subtract(ipr_dict.OffsetY, _ipr_dict.OffsetY)

        if Compare_QCresult ==0 and (subtract(ipr_dict.CounterRot90, ipr_dict.CounterRot90) == 0)\
                and (subtract(ipr_dict.CounterRot90, ipr_dict.CounterRot90) == 0):
            if abs(Compare_offsetX) < 51 and Compare_offsetY < 51:
                self.comment = 'Register is same'
        else:
            self.comment = 'Register is different'


class CompareTissue(object):
    def __init__(self, ipr_dict: dict2class, _ipr_dict: dict2class):
        tissue = decode(ipr_dict.TissueMask, ipr_dict.TissueSegShape)
        tissue_ = decode(_ipr_dict.TissueMask, _ipr_dict.TissueSegShape)

        self.iou = count_iou(tissue, tissue_)


class CompareCell(object):
    def __init__(self, ipr_dict: dict2class, _ipr_dict: dict2class):
        mask = decode(ipr_dict.CellMask, ipr_dict.CellSegShape)
        mask_ = decode(_ipr_dict.CellMask, _ipr_dict.CellSegShape)
        self.cellnum, self.mean_area, self.median_area, self.max_area = get_mask_area(mask)
        self.cellnum_, self.mean_area_, self.median_area_, self.max_area_ = get_mask_area(mask_)


class CompareModule(object):
    def __init__(self, ipr_dict, _ipr_dict):
        self.ipr_dict = dict2class(ipr_dict)
        self._ipr_dict = dict2class(_ipr_dict)
        self.CompareQC = CompareQC(self.ipr_dict, self._ipr_dict)
        self.CompareChip = CompareChip(self.ipr_dict, self._ipr_dict)
        self.CompareTemplate = CompareTemplate(self.ipr_dict, self._ipr_dict)
        self.CompareRegist = CompareRegist(self.ipr_dict, self._ipr_dict)
        self.CompareTissue = CompareTissue(self.ipr_dict, self._ipr_dict)
        self.CompareCell = CompareCell(self.ipr_dict, self._ipr_dict)



