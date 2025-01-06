import shutil
import sys
import os
import pytest
import traceback

CURR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# CB_PATH = os.path.join(CURR_PATH,)
print(CURR_PATH)
sys.path.append(CURR_PATH)
from cellbin2.cellbin_pipeline import pipeline
import cellbin2
from glob import glob
from utils.file_compare import ipr_compare, file_compare
from utils.module_compare import module_compare_pipeline
from utils.common import dict2mdtable, get_data_info
from datetime import datetime
from pathlib import Path
import collections
import pandas as pd


WEIGHTS_ROOT = r"D:\StereoImage-cellbin\fengning_presonal_gitlab\no_matrix_17\weights"
TEST_OUTPUT_DIR = r"D:\temp\cellbin_test\testdata"
DEMO_DATA_DIR = "/media/Data1/user/dengzhonghan/data/cellbin2/demo_data"
table_dict = collections.defaultdict()

# TEST_DATA = [
#     (
#         # ssDNA
#         "SS200000135TL_D1",
#         "SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif",
#         "",
#         "ssDNA",
#         "SS200000135TL_D1/SS200000135TL_D1.raw.gef",
#         "",
#         "Stereo-seq T FF V1.2 R",
#         r"D:\temp\cellbin_test\GOLD\SS200000135TL_D1"
#     )
# ]
datafile_path = r"D:\temp\cellbin_test\solutions_data.xlsx"
processed_data = get_data_info(datafile_path)

def fold(record_md, list):
    record_md.write('<details>\n')
    record_md.write('<summary>Type different detail </summary> \n')
    for i in list:
        record_md.write(f'* {i}<br> \n')
    record_md.write('</details> \n\n')


def pytest_addoption(parser):
    parser.addoption(
        "--datafile",
        action='store',
        default=None  #放一个固定的数据在这里
    )

class TestCompare:
    @pytest.fixture(scope="class")
    def record_md(self):
        formatted_datetime = datetime.now().strftime("%Y%m%d%H%M")
        self.f = open(os.path.join(TEST_OUTPUT_DIR, f"Testreport_{formatted_datetime}.md"), "a")
        yield self.f  # 允许测试函数使用
        self.f.close()  # 测试完成后关闭文件

    @pytest.fixture(scope="class")
    def table(self):
        formatted_datetime = datetime.now().strftime("%Y%m%d%H%M")
        yield table_dict
        df = pd.DataFrame(table_dict).T
        df.to_excel(os.path.join(TEST_OUTPUT_DIR, f"Testreport_{formatted_datetime}.xlsx"))

    # test script mode
    @pytest.mark.parametrize("sn, info_dict", processed_data)
    def test_run(self, sn, info_dict, record_md, table):
        print(sn)

        record_md.write(f"# Test case {sn} <br>\n")

        cur_test_out = TEST_OUTPUT_DIR #tmp_dir_without version record
        print(f"Test results had saved at {cur_test_out}")
        cur_out = os.path.join(cur_test_out, sn)

        ipr_file = glob(os.path.join(cur_out, '*.ipr'))

        if len(ipr_file) == 0:
            os.makedirs(cur_test_out, exist_ok=True)
            im_path = info_dict.get("Stitched_Image")
            if_path = info_dict.get("IF_image")
            if if_path != "/":
                pps = if_path.split(",")
                pps_ = [os.path.join(DEMO_DATA_DIR, i) for i in pps]
                if_path = ",".join(pps_)
            else:
                if_path = None
            trans_gef = info_dict.get("matrix_path")

            if info_dict.get("pgef_path") != "/":
                p_gef = info_dict.get("pgef_path")
            else:
                p_gef = None

            s_type = info_dict.get("track_stain_type")
            kit_type = info_dict.get("kit_type")
            print(sn, im_path, if_path, s_type, trans_gef, p_gef, kit_type)
            pipeline(
                chip_no=sn,
                input_image=im_path,
                if_image=if_path,
                stain_type=s_type,
                param_file=None,
                output_path=cur_out,
                matrix_path=trans_gef,
                protein_matrix_path=p_gef,
                kit=kit_type,
                if_report=True,
                weights_root=WEIGHTS_ROOT,
            )

            ipr_file = glob(os.path.join(cur_out, '*.ipr'))
            if len(ipr_file) == 0:
                pytest.fail(" New ipr failed to created, ERROR!.")

        saw_result = info_dict.get("SAW_V8.1")
        ipr_file_product = glob(os.path.join(saw_result, '*.ipr'))


        record_md.write('## Ipr compare: <br>\n')
        if len(ipr_file_product) == 0:
            pytest.fail(" Can not find ipr in SAW V8 result dir, forcing test to fail.")

        else:
            # Compare Ipr
            type_is_same, value_is_same, type_record, value_record, de_attrs = ipr_compare(ipr_file[0], ipr_file_product[0])
            if len(de_attrs):
                record_md.write('* The following attrs are missing: <br>\n')
                record_md.write(f'{de_attrs} <br> \n')

            if len(type_record):
                fold(record_md, type_record)

            if len(value_record):
                fold(record_md, value_record)

            record_md.write('--- \n')

            # Compare File
            is_complete, de_file = file_compare(cur_out, saw_result)
            record_md.write('## Number of file: <br>\n')
            if len(de_file):
                record_md.write('* The following files are missing: <br>\n')
                record_md.write(f'{de_file} <br> \n')

            else:
                record_md.write(f'* All documents required for the product are complete. <br>\n')
            record_md.write('--- \n')

            # Compare Module
            result_is_same, result_dict = module_compare_pipeline(cur_out, saw_result)
            record_md.write('## Module compare: <br>\n')

            # 补充上面的内容
            result_dict['type_is_same'] = type_is_same
            result_dict['file_is_complete'] = is_complete
            result_dict['result_is_same'] = result_is_same

            markdown_table = dict2mdtable(result_dict)
            record_md.write(f"{markdown_table}")
            table[sn] = result_dict

            assert type_is_same & is_complete & result_is_same, "Test Failed~ The IPR file does not meet the product requirements."
