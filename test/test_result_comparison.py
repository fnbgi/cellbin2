import shutil
import sys
import os
import pytest
import traceback

from test.utils.module_compare import module_compare_pipeline

CURR_PATH = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# CB_PATH = os.path.join(CURR_PATH,)
print(CURR_PATH)
sys.path.append(CURR_PATH)
from cellbin2.cellbin_pipeline import pipeline
import cellbin2
from glob import glob
from utils.file_compare import ipr_compare
from utils.common import dict2mdtable
from datetime import datetime


WEIGHTS_ROOT = "/media/Data1/user/dengzhonghan/data/cellbin2/weights"
TEST_OUTPUT_DIR = "/media/Data1/user/dengzhonghan/data/cellbin2/auto_test"
DEMO_DATA_DIR = "/media/Data1/user/dengzhonghan/data/cellbin2/demo_data"
TEST_DATA = [
    (
        # DAPI + mIF
        "SS200000045_M5",  # sn
        "SS200000045_M5/SS200000045_M5_fov_stitched.tif",  # DAPI, HE, ssDNA path
        "SS200000045_M5/SS200000045_M5_ATP_IF_fov_stitched.tif,"
        "SS200000045_M5/SS200000045_M5_CD31_IF_fov_stitched.tif,"
        "SS200000045_M5/SS200000045_M5_NeuN_IF_fov_stitched.tif",  # IF path
        "DAPI",  # stain_type (DAPI, HE, ssDNA)
        "SS200000045_M5/SS200000045_M5.raw.gef",  # transcriptomics gef path
        "",  # protein gef path
        "Stereo-seq T FF V1.2 R",
        "" # Product version SAW v8 dir
    ),
    (
        # FF H&E
        "C04042E3",
        "C04042E3/C04042E3_fov_stitched.tif",
        "",
        "HE",
        "C04042E3/C04042E3.raw.gef",
        "",
        "Stereo-seq T FF V1.3 R",
        ""
     ),
    (
        # ssDNA
        "SS200000135TL_D1",
        "SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif",
        "",
        "ssDNA",
        "SS200000135TL_D1/SS200000135TL_D1.raw.gef",
        "",
        "Stereo-seq T FF V1.2 R",
        ""
    ),
    (
        # DAPI + IF
        "A03599D1",
        "A03599D1/A03599D1_DAPI_fov_stitched.tif",
        "A03599D1/A03599D1_IF_fov_stitched.tif",
        "DAPI",
        "A03599D1/A03599D1.raw.gef",
        "A03599D1/A03599D1.protein.raw.gef",
        "Stereo-CITE T FF V1.0 R",
        ""
    )
]



class TestResult:
    @pytest.fixture(scope="class")
    def record_md(self):
        formatted_datetime = datetime.now().strftime("%Y%m%d%H%M")
        self.f = open(f"Testreport_{formatted_datetime}.md", "a")
        yield self.f  # 允许测试函数使用
        self.f.close()  # 测试完成后关闭文件

    # test script mode
    @pytest.mark.parametrize("sn, im_path, if_path, s_type, trans_gef, p_gef, kit_type, saw_result", TEST_DATA)
    def test_run(self, sn, im_path, if_path, s_type, trans_gef, p_gef, kit_type, saw_result, record_md):
        record_md.write(f"# Test case {sn}")

        git_commit = os.getenv('GITHUB_SHA')
        if git_commit is not None:
            cur_test_out = os.path.join(TEST_OUTPUT_DIR, git_commit)
        else:
            cur_test_out = os.path.join(TEST_OUTPUT_DIR, cellbin2.__version__)
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

        ipr_file_product = glob(os.path.join(saw_result, '*.ipr'))
        ipr_file = glob(os.path.join(cur_out, '*.ipr'))


        record_md.write('## Ipr compare: \n')
        if len(ipr_file_product) == 0:
            pytest.fail(" Can not find ipr in SAW V8 result dir, forcing test to fail.")

        else:
            # Compare ipr
            type_is_same, value_is_same, type_record, value_record = ipr_compare(ipr_file[0], ipr_file_product[0])
            for i in type_record:
                record_md.write(f'* {i} \n')

            for i in value_record:
                record_md.write(f'* {i} \n')

            record_md.write('--- \n')

            assert type_is_same

            # Compare Module
            result_is_same, result_dict = module_compare_pipeline(cur_out, saw_result)
            record_md.write('## Module compare: \n')

            markdown_table = dict2mdtable(result_dict)
            record_md.write(f"{markdown_table}")

            assert result_is_same
