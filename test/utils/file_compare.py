# -*- coding: utf-8 -*-
# @Time    : 2024/12/23 11:23
# @Author  : unknow
# @File    : file_compare.py
import os
from glob import glob
from typing import Union
import h5py

from .common import parse_ipr, dict_compare
from .. import config


def ipr_compare(ipr_file: str, _ipr_file = config.PRODUCT_IPR):
    """
    compare 2 iprs attrs and type
    """
    type_is_same = True
    value_is_same = True
    type_record = []
    value_record = []

    type_attributes, ipr_dict = parse_ipr(ipr_file)
    _type_attributes, _ipr_dict = parse_ipr(_ipr_file)

    # compare key and type of 2 ipr
    de_attrs = set(_type_attributes.keys()).difference(type_attributes.keys())
    if len(de_attrs):
        print(f'The missing attributes are {de_attrs}')
        type_is_same = False
        value_is_same = False

    else:
        if type_attributes == _type_attributes:
            print('The property names and data types of the two files are the same')

        else:
            for attr_name in _type_attributes.keys():
                _attr_type = _type_attributes.get(attr_name, None)
                attr_type = type_attributes.get(attr_name, None)
                if attr_type != _attr_type:
                    warning = f'attribute < {attr_name} > is {attr_type}, and comparison is {_attr_type} !'
                    print(warning)
                    type_record.append(warning)


    #compare values of 2 ipr
    if ipr_dict == _ipr_dict:
        print('The values of attribute of the two files are the same')

    else:
        value_is_same, value_record = dict_compare(ipr_dict, _ipr_dict)

    return type_is_same, value_is_same, type_record, value_record


def get_filelist(path: str, suffix_list: list[str]):
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

    de_file = set(need_files).difference(result_files)

    if len(de_file):
        print(f'Compare with {compare_dir}. Missing file is {de_file}')






