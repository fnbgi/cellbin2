# -*- coding: utf-8 -*-
# @Time    : 2025/1/5 21:06
# @Author  : unknow
# @File    : conftest.py

import pytest
from utils.common import get_data_info

def pytest_addoption(parser):
    parser.addoption(
        "--datafile",
        action='store',
        default=r"D:\StereoImage-cellbin\Cellbin_devTest\data\solutions_data.xlsx"  # 设置默认值
    )

# @pytest.fixture()
# def datafile(request):
#     datafile_path = request.config.getoption("--datafile")
#     if not datafile_path:
#         pytest.skip("No --datafile provided")
#
#     # 进行数据处理，比如读取文件内容
#     processed_data = get_data_info(datafile_path)
#     return processed_data
