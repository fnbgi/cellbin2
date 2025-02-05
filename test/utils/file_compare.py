# -*- coding: utf-8 -*-
# @Time    : 2024/12/23 11:23
# @Author  : unknow
# @File    : file_compare.py
import os
import sys
from pathlib import Path
from typing import Dict, Union, Tuple, List
import numpy as np
sys.path.append(Path(__file__).parents[1])
import config
from typing import List
from .common import create_crosspoint, compare_points

sys.path.append(Path(__file__).parents[2])
from cellbin2.utils.ipr import read

stain_list = ('ssDNA', 'DAPI', 'HE')

def deep_compare(
        obj1,
        obj2,
        path: str = '',
) -> Tuple[List[str], List[str]]:
    """

    Args:
        obj1:
        obj2:
        attrs:

    Returns:

    """
    type_diffs = []
    value_diffs = []
    miss_attrs = []

    # skip specify attributes
    if path.endswith("QCInfo.CrossPoints"):
        return type_diffs, value_diffs, miss_attrs

    # Judge whether the types are consistent
    if type(obj1) != type(obj2):
        attr_type = str(type(obj1)).replace("<", "<'")
        attr_type2 = str(type(obj2)).replace("<", "<'")
        type_diffs.append(f"the type of attribute {path} is {attr_type}, and comparison is {attr_type2} !")

        return type_diffs, value_diffs, miss_attrs

    # Judge whether the values are consistent
    if hasattr(obj1, "__dict__"):
        d1 = vars(obj1)
        d2 = vars(obj2)
        all_attrs = set(d2.keys())
        for attr in all_attrs:
            sub_path = f"{path}.{attr}" if path else attr

            if attr not in d1:
                # miss_attrs.append(f"[{sub_path}] - missing in now ipr!")
                miss_attrs.append(sub_path)
                continue

            td, vd, ma = deep_compare(d1[attr], d2[attr], sub_path)
            type_diffs.extend(td)
            value_diffs.extend(vd)
            miss_attrs.extend(ma)
        return type_diffs, value_diffs, miss_attrs

    if isinstance(obj1, dict):
        all_keys = set(obj2.keys())
        for key in all_keys:
            sub_path = f"{path}.{key}" if path else key

            if key not in obj1:
                # miss_attrs.append(f"[{sub_path}] - missing in now ipr!")
                miss_attrs.append(sub_path)
                continue
            td, vd, ma = deep_compare(obj1[key], obj2[key], sub_path)
            type_diffs.extend(td)
            value_diffs.extend(vd)
            miss_attrs.extend(ma)
        return type_diffs, value_diffs, miss_attrs

    # if isinstance(obj1, (list, tuple)):
    #     if obj1 != obj2:
    #         value_diffs.append(
    #             f"[{path}] - list length is different: {len(obj1)} vs. {len(obj2)}"
    #         )
    #     else:
    #         for i, (v1, v2) in enumerate(zip(obj1, obj2)):
    #             sub_path = f"{path}[{i}]"
    #             td, vd = deep_compare(v1, v2, sub_path)
    #             type_diffs.extend(td)
    #             value_diffs.extend(vd)
    #     return type_diffs, value_diffs

    if isinstance(obj1, np.ndarray) and isinstance(obj2, np.ndarray):
        if obj1.shape != obj2.shape:
            value_diffs.append(
                f"[{path}] - ndarray shape is different: {obj1.shape} vs. {obj2.shape}"
            )
        else:
            # 形状相同再比较内容
            if not np.array_equal(obj1, obj2):
                value_diffs.append(
                    f"[{path}] - ndarray is not equal"
                )
        return type_diffs, value_diffs, miss_attrs


    if obj1 != obj2:
        value_diffs.append(
            f"[{path}] - value different: {obj1} vs. {obj2}"
        )

    return type_diffs, value_diffs, miss_attrs


def compare_Crosspoint(image_dct: dict, image_dct_: dict):
    for stain in image_dct.keys():
        if stain in stain_list and stain in image_dct_.keys():
            if image_dct[stain].ImageInfo.StitchedImage == image_dct_[stain].ImageInfo.StitchedImage:
                point1 = create_crosspoint(image_dct[stain])
                point2 = create_crosspoint(image_dct_[stain])
                warnning = compare_points(point1, point2)
                diff = f"[{stain}.QCInfo.CrossPoints] - " + warnning

            else:
                diff = f"[{stain}.QCInfo.CrossPoints] - The input Images is different,one is fov folders ,Other is Stitched Image!"

            return diff
        else:
            diff = f"[{stain}.QCInfo.CrossPoints] - Stain is different! No Compare!"

    return diff


def ipr_compare(ipr_file: str, ipr_file_ = config.PRODUCT_IPR):
    _, image_dct = read(ipr_file)
    _, image_dct_ = read(ipr_file_)

    type_is_same = True
    value_is_same = True

    type_diffs, value_diffs, miss_attrs = deep_compare(image_dct, image_dct_)

    if len(type_diffs):
        type_is_same = False

    if len(value_diffs):
        value_is_same = False

    new_warning = compare_Crosspoint(image_dct, image_dct_)
    value_diffs.append(new_warning)

    # categories the values diff list
    categories = ["TissueSeg", "ImageInfo", "CellSeg", "QCInfo", "Stitch", "Register"]
    cat_map = {c: [] for c in categories}
    cat_map["Others"] = []

    for diff in value_diffs:
        if diff.startswith("[") and "]" in diff:
            path_part = diff[1 : diff.index("]")]
        else:
            path_part = diff

        split_path = path_part.split(".")
        if len(split_path) > 1 and split_path[1] in categories:
            cat_map[split_path[1]].append(diff)
        else:
            cat_map["Others"].append(diff)

    return type_is_same, value_is_same, type_diffs, cat_map,  miss_attrs

# def ipr_compare(ipr_file: str, _ipr_file = config.PRODUCT_IPR):
#     """
#     compare 2 iprs attrs and type
#     """
#     type_is_same = True
#     value_is_same = True
#     type_record = []
#     value_record = []
#
#     type_attributes, ipr_dict = parse_ipr(ipr_file)
#     _type_attributes, _ipr_dict = parse_ipr(_ipr_file)
#
#     # compare key and type of 2 ipr
#     de_attrs = set(_type_attributes.keys()).difference(type_attributes.keys())
#     if len(de_attrs):
#         print(f'The missing attributes are {de_attrs}')
#         type_is_same = False
#         value_is_same = False
#
#     if type_attributes == _type_attributes:
#         print('The property names and data types of the two files are the same')
#     else:
#         for attr_name in _type_attributes.keys():
#             _attr_type = str(_type_attributes.get(attr_name, None)).replace("<", "<`")
#             attr_type = str(type_attributes.get(attr_name, None)).replace("<", "<`")
#             if attr_type != _attr_type:
#                 warning = f"attribute {attr_name} is {attr_type}, and comparison is {_attr_type} !"
#                 print(warning)
#                 type_record.append(warning)
#
#
#     #compare values of 2 ipr
#     if ipr_dict == _ipr_dict:
#         print('The values of attribute of the two files are the same')
#
#     else:
#         value_is_same, value_record = dict_compare(ipr_dict, _ipr_dict)
#         new_warning = compare_Crosspoint(ipr_dict, _ipr_dict)
#         value_record.append(new_warning)
#
#     return type_is_same, value_is_same, type_record, value_record, de_attrs


def get_filelist(path: str, suffix_list: List[str]):
    file_list = []
    for root, dirs, files in os.walk(path):
        file_list.extend(files)
    file_list = [file for file in file_list if os.path.splitext(file)[-1] in suffix_list]

    return file_list


def file_compare(result_dir: str, compare_dir: str):
    """
    compare result files  with Product version
    """
    suffix_list = ['.tif', '.tiff', '.rpi', '.TIFF']
    need_files = get_filelist(compare_dir, suffix_list)
    result_files = get_filelist(result_dir, suffix_list)
    is_complete = True

    de_file = set(need_files).difference(result_files)

    if len(de_file):
        print(f'Compare with {compare_dir}. Missing file is {de_file}')
        is_complete = False

    return is_complete, de_file

