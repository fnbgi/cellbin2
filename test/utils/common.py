# -*- coding: utf-8 -*-
# @Time    : 2024/12/25 11:20
# @Author  : unknow
# @File    : common.py
import h5py
import numpy as np
import pandas as pd
from typing import Union, Literal
import sys
from pathlib import Path
sys.path.append(Path(__file__).parents[2])

from cellbin2.image.augmentation import  dapi_enhance, he_enhance
from cellbin2.contrib.template.inferencev1 import TemplateReferenceV1
from cellbin2.utils.ipr import read

pt_enhance_method = {
    'ssDNA': dapi_enhance,
    'DAPI': dapi_enhance,
    'HE': he_enhance
}

stain_list = ['ssDNA', 'DAPI', 'HE']

def xlsx2json(xlsx_path: str):
    df = pd.read_excel(xlsx_path, index_col="SN")
    df = df.fillna(value="/")
    data_json = df.to_dict(orient="index")

    return data_json


def get_data_info(
        xlsx_path: str,
        run_module: Union[list, str],
        use_path: str = 'ztron_path'
):
    basic_info_df = pd.read_excel(xlsx_path, sheet_name='basic_info')
    basic_info_df = basic_info_df.fillna(value="/")

    path_info_df = pd.read_excel(xlsx_path, sheet_name=use_path)
    path_info_df = path_info_df.fillna(value="/")

    if isinstance(run_module, str):
        run_info = pd.read_excel(xlsx_path, sheet_name=run_module)
        run_list = set(run_info[run_info["RUN"] == 1]['SN'].tolist())

    elif isinstance(run_module, list):
        run_list = set()
        for module in run_module:
            run_info = pd.read_excel(xlsx_path, sheet_name=module)
            run_list = run_list | set(run_info[run_info["RUN"] == 1]['SN'].tolist())

    result = []
    for sn in run_list:
        basic_info = basic_info_df[basic_info_df["SN"] == sn]
        path_info = path_info_df[path_info_df["SN"] == sn]

        if not basic_info.empty and not path_info.empty:
            merge_info = pd.merge(basic_info, path_info, on='SN').to_dict(orient='records')[0]
            tmp_tup = (sn, merge_info)
            result.append(tmp_tup)

    return result


class Sample:
    def __init__(self):
        pass

    def __getattr__(self, name):
        # If the property does not exist, return None
        return None


class Wrapper:
    def __init__(self, obj):
        self._obj = obj

    def __getattr__(self, name):
        if hasattr(self._obj, name):
            return getattr(self._obj, name)
        return None


def dict2class(dct):
    sample = Sample()
    for k, v in dct.items():
        if isinstance(v, dict):
            sample.__dict__[k] = dict2class(v)
        else:
            sample.__dict__[k] = v

    return sample


def wrapper_class(obj):
    wrapper = Wrapper(obj)
    for k in obj.__dict__:
        # print(k)
        if hasattr(obj.__dict__[k], "__dict__"):
            wrapper.__dict__[k] = wrapper_class(obj.__dict__[k])

        else:
            wrapper.__dict__[k] = obj.__dict__[k]
    return wrapper

# def get_attributes(file):
#     type_attributes = {}
#     ipr_dict = {}
#     CrossPoints_dict = {}
#
#     def visit_group(name, obj):
#         if isinstance(obj, h5py.Group):
#             # get all name , type, value of obj
#             attrs = obj.attrs.items()
#             parent = obj.parent.name.split('/')[-1]
#             for attr_name, attr_value in attrs:
#                 type_attributes[attr_name] = type(attr_value)
#                 if attr_name in ['Method', 'OffsetX', 'OffsetY', 'ScaleX', 'ScaleY'] and parent in ['Register', 'QCInfo']:
#                     # print(attr_name)
#                     continue
#                 ipr_dict[attr_name] = attr_value
#
#         if isinstance(obj, h5py.Dataset):
#             # set dataset of obj to array
#             new_name = name.split('/')[-1]
#             parent = obj.parent.name.split('/')[-1]
#             if parent == 'CrossPoints':
#                 CrossPoints_dict['fov_' + new_name] = obj[()]
#             else:
#                 ipr_dict[new_name] = obj[()]
#
#         ipr_dict['CrossPoints'] = CrossPoints_dict
#
#     file.visititems(visit_group)
#     return type_attributes, ipr_dict


# def parse_ipr(ipr_file):
#     ipr_if_dict = {}
#
#     with h5py.File(ipr_file, 'r') as f:
#         stain_list = ('ssDNA', 'DAPI', 'HE')
#         set_ = ('ManualState', 'StereoResepSwitch', 'Preview', 'Research')
#         if_group = list(set(f.keys()).difference(set_ + stain_list))
#
#         if len(set(f.keys()).intersection(stain_list)) == 0:
#             raise Exception('This ipr is not legal, break!')
#
#         for group in f.keys():
#             if group in stain_list:
#                 type_attributes, ipr_dict = get_attributes(f[group])
#
#         for fluo in if_group:
#             type_attributes_, ipr_dict_ = get_attributes(f[fluo])
#             ipr_if_dict[fluo] = ipr_dict_
#
#         ipr_dict['IF'] = ipr_if_dict
#
#         return type_attributes, ipr_dict


def dict_compare(ipr_dict: dict, ipr_dict2: dict):
    """

    Args:
        ipr_dict:
        ipr_dict2: 上一次版本的结果

    Returns:

    """
    record = []
    is_same = True

    for k, v in ipr_dict2.items():
        if k.startswith('fov_'):
            continue

        if isinstance(v, dict):
            _is_same, _record = dict_compare(ipr_dict.get(k), v)
            record.extend(_record)
        else:
            if np.any(v != ipr_dict.get(k)):
                if isinstance(v, np.ndarray):
                    warning = f'the value of attribute < {k} > is different '
                else:
                    warning = f'the value of attribute < {k} > is different, ' \
                              f'please check !!! new is {ipr_dict.get(k)} and comparison is {v} '
                # warning = f'the value of attribute < {k} > is different '
                print(warning)
                _is_same = False
                record.append(warning)
            else:
                _is_same = True

        is_same &= _is_same

    return is_same, record


def dict2mdtable(data_dict):
    headers = list(data_dict.keys())
    rows = [list(data_dict.values())]

    # 生成 Markdown 表格
    markdown_table = "| " + " | ".join(headers) + " |\n"
    markdown_table += "| " + " | ".join(["---"] * len(headers)) + " |\n"
    for row in rows:
        markdown_table += "| " + " | ".join(map(str, row)) + " |\n"

    return markdown_table

def create_crosspoint(image_dct):
    _points = np.ones([0, 4])
    loc = image_dct.Stitch.ScopeStitch.GlobalLoc
    if len(loc):
        for k,v in vars(image_dct.QCInfo.CrossPoints).items():
            if v.shape[1] == 3:
                v = np.concatenate((v[:, :2], np.zeros([v.shape[0], 2])), axis=1)
            r, c = map(int, k.split("_")[-2:])
            _loc = loc[r, c]
            v[:, :2] = v[:, :2] + _loc
            _points = np.concatenate((_points, v), axis=0)
    else:
        for k, v in vars(image_dct.QCInfo.CrossPoints).items():
            _points = np.concatenate((_points, v), axis=0)

    return _points

def compare_points(
        points1: np.ndarray,
        points2: np.ndarray,
        k: int = 10
):
    check_result = 'YES'
    _temp, _track = TemplateReferenceV1.pair_to_template(
        points1[:, :2], points1[:, :2], threshold = k
    )
    warning = f"New DATA QC detects {len(points1)} points, " \
              f" compared to {len(points2)} points, " \
              f"{len(_temp)} pairs of points within 10pixels "
    if len(points1) != len(points2) or len(points2) != len(_temp):
        print(warning)
        check_result = 'NO'

    return warning, check_result

if __name__ == '__main__':
    ipr, image_dct = read(r"D:\temp\cellbin_test\GOLD\SS200000135TL_D1\SS200000135TL_D1_SC_20241030_151054_4.1.0.ipr")
    wrapper_class(image_dct['ssDNA'])
