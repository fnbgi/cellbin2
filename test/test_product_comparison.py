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
from datetime import datetime
from pathlib import Path


WEIGHTS_ROOT = r"D:\StereoImage-cellbin\fengning_presonal_gitlab\no_matrix_17\weights"
TEST_OUTPUT_DIR = r"D:\temp\cellbin_test\testdata"
DEMO_DATA_DIR = "/media/Data1/user/dengzhonghan/data/cellbin2/demo_data"


TEST_DATA = [
    (
        # ssDNA
        "SS200000135TL_D1",
        "SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif",
        "",
        "ssDNA",
        "SS200000135TL_D1/SS200000135TL_D1.raw.gef",
        "",
        "Stereo-seq T FF V1.2 R",
        r"D:\temp\cellbin_test\GOLD\SS200000135TL_D1"
    )
]

class TestProductVs:
    @pytest.fixture(scope="class")
    def record_md(self):
        formatted_datetime = datetime.now().strftime("%Y%m%d%H%M")
        self.f = open(os.path.join(TEST_OUTPUT_DIR ,f"Testreport_{formatted_datetime}.md"), "a")
        yield self.f  # 允许测试函数使用
        self.f.close()  # 测试完成后关闭文件

    # test script mode
    @pytest.mark.parametrize("sn, im_path, if_path, s_type, trans_gef, p_gef, kit_type, saw_result", TEST_DATA)
    def test_run(self, sn, im_path, if_path, s_type, trans_gef, p_gef, kit_type, saw_result, record_md):
        record_md.write(f"# Test case {sn} <br>\n")

        # git_commit = os.getenv('GITHUB_SHA')
        # if git_commit is not None:
        #     cur_test_out = os.path.join(TEST_OUTPUT_DIR, git_commit)
        # else:
        #     cur_test_out = os.path.join(TEST_OUTPUT_DIR, cellbin2.__version__)
        cur_test_out = TEST_OUTPUT_DIR #tmp_dir_without version record
        print(f"Test results had saved at {cur_test_out}")
        cur_out = os.path.join(cur_test_out, sn)

        ipr_file = glob(os.path.join(cur_out, '*.ipr'))
        if len(ipr_file) == 0:
            os.makedirs(cur_test_out, exist_ok=True)
            im_path = os.path.join(DEMO_DATA_DIR, im_path)
            if if_path != "":
                pps = if_path.split(",")
                pps_ = [os.path.join(DEMO_DATA_DIR, i) for i in pps]
                if_path = ",".join(pps_)
            else:
                if_path = None
            trans_gef = os.path.join(DEMO_DATA_DIR, trans_gef)
            if p_gef != "":
                p_gef = os.path.join(DEMO_DATA_DIR, p_gef)
            else:
                p_gef = None
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

        ipr_file_product = glob(os.path.join(saw_result, '*.ipr'))


        record_md.write('## Ipr compare: <br>\n')
        if len(ipr_file_product) == 0:
            pytest.fail(" Can not find ipr in SAW V8 result dir, forcing test to fail.")

        else:
            type_is_same, value_is_same, type_record, value_record = ipr_compare(ipr_file[0], ipr_file_product[0])
            for i in type_record:
                record_md.write(f'* {i} \n')

            for i in value_record:
                record_md.write(f'* {i} \n')

            record_md.write('--- \n')


            is_complete, de_file = file_compare(cur_out, saw_result)
            record_md.write('## Number of file: \n')
            if len(de_file):
                record_md.write('* The following files are missing: <br>\n')
                record_md.write(f'{de_file} <br> \n')

            else:
                record_md.write(f'* All documents required for the product are complete. <br>\n')

            assert type_is_same & is_complete, "Test Failed~ The IPR file does not meet the product requirements."