# -*- coding: utf-8 -*-
"""
🌟 Create Time  : 2025/4/24 16:35
🌟 Author  : CB🐂🐎 - lizepeng
🌟 File  : stereo_chip_name.py
🌟 Description  : 
🌟 Key Words  :
"""
import os
import json


def dict2class(dct, cls = None):
    if cls is None:
        scn = StereoChipName()
    else:
        scn = cls
    for k, v in dct.items():
        if isinstance(v, dict):
            scn.__dict__[k] = dict2class(v)
        else:
            scn.__dict__[k] = v

    return scn


class StereoChipName(object):
    """

    """
    def __init__(
            self,
            prefix: str,
            chip_name: str = None
    ):
        """

        Args:
            prefix:

        """
        self.prefix = prefix
        self.chip_name = chip_name

        self.state = ''
        self.pitch = -1
        self.track_line = []
        self.long = False
        self.short = False
        self.origin = ''
        self.corr_long = ''
        self.corr_long_prefix = ''
        self.sequence = ''
        self.s1_length = -1
        self.un_s1_length = -1

    def parse_info(self, ):
        """
        用于部分芯片的特殊处理

        Returns:

        """
        if self.chip_name is None:
            return

        # 注: SS2/FP2/SS1长码加TL/TR/BL/BR对应T10，不加对应T10L
        if self.prefix in ["SS2", "FP2", "SS1"]:
            for s in ["TL", "TR", "BL", "BR"]:
                if s in self.chip_name:
                    self.sequence = "T10"
                    break
            else:
                self.sequence = "T10L"


def get_chip_prefix_info(
        prefix: str,
        chip_name: str = None
):
    """

    Args:
        prefix:
        chip_name:

    Returns:

    """
    curr_path = os.path.dirname(os.path.realpath(__file__))
    chip_mask_file = os.path.join(curr_path, r'../config/chip_name.json')

    with open(chip_mask_file, 'r') as fd:
        chip_dict = json.load(fd)

    name_dict = chip_dict[prefix]
    scn = StereoChipName(prefix = prefix, chip_name = chip_name)
    scn = dict2class(name_dict, scn)

    scn.parse_info()

    return scn


if __name__ == '__main__':
    get_chip_prefix_info(prefix='SS2', chip_name = 'SS200000TL_D1')






