# -*- coding: utf-8 -*-
# @Time    : 2024/12/25 11:20
# @Author  : unknow
# @File    : common.py
import h5py
import numpy as np


class Sample:
    def __init__(self):
        pass

    def __getattr__(self, name):
        # 如果属性不存在，返回 None
        return None


def dict2class(dct):
    sample = Sample()
    for k, v in dct.items():
        if isinstance(v, dict):
            sample.__dict__[k] = dict2class(v)
        else:
            sample.__dict__[k] = v

    return sample


def get_attributes(file):
    type_attributes = {}
    ipr_dict = {}
    CrossPoints_dict = {}

    def visit_group(name, obj):
        if isinstance(obj, h5py.Group):
            # get all name , type, value of obj
            attrs = obj.attrs.items()
            for attr_name, attr_value in attrs:
                type_attributes[attr_name] = type(attr_value)
                ipr_dict[attr_name] = attr_value

        if isinstance(obj, h5py.Dataset):
            # set dataset of obj to array
            new_name = name.split('/')[-1]
            parent = obj.parent.name.split('/')[-1]
            if parent == 'CrossPoints':
                CrossPoints_dict['fov_' + new_name] = obj[()]
            else:
                ipr_dict[new_name] = obj[()]

        ipr_dict['CrossPoints'] = CrossPoints_dict

    file.visititems(visit_group)
    return type_attributes, ipr_dict


def parse_ipr(ipr_file):
    ipr_if_dict = {}

    with h5py.File(ipr_file, 'r') as f:
        stain_list = ('ssDNA', 'DAPI', 'HE')
        set_ = ('ManualState', 'StereoResepSwitch', 'Preview', 'Research')
        if_group = list(set(f.keys()).difference(set_ + stain_list))

        if len(set(f.keys()).intersection(stain_list)) == 0:
            raise Exception('This ipr is not legal, break!')

        for group in f.keys():
            if group in stain_list:
                type_attributes, ipr_dict = get_attributes(f[group])

        for fluo in if_group:
            type_attributes_, ipr_dict_ = get_attributes(f[fluo])
            ipr_if_dict[fluo] = ipr_dict_

        ipr_dict['IF'] = ipr_if_dict

        return type_attributes, ipr_dict


def dict_compare(ipr_dict: dict, ipr_dict2: dict):
    """

    Args:
        ipr_dict:
        ipr_dict2: 上一次版本的结果

    Returns:

    """
    is_same = True
    for k,v in ipr_dict2.items():
        if isinstance(v, dict):
            dict_compare(ipr_dict.get(k), v)
        else:
            if v != ipr_dict.get(k):
                print(f'the value of attribute < {k} > is different, please check !!!')
                is_same = False
    return is_same



