from typing import Dict, Union, Optional
import numpy as np
import os
from pydantic import BaseModel, Field

from cellbin2.modules.metadata import ProcFile
from cellbin2.modules import naming
from cellbin2.contrib.alignment import AlignMode, registration, RegistrationOutput
from cellbin2.utils.ipr import IFChannel, ImageChannel
from cellbin2.utils import clog
from cellbin2.contrib.alignment.basic import ChipFeature
from cellbin2.modules.extract.matrix_extract import MatrixFeatureExtract
from cellbin2.utils.stereo_chip import StereoChip
from cellbin2.utils.config import Config
from cellbin2.contrib.alignment import ChipBoxInfo


class RegisterInput(BaseModel):
    image_file: ProcFile
    cur_f_name: naming.DumpImageFileNaming
    files: Dict[int, ProcFile]
    channel_images: Dict[str, Union[IFChannel, ImageChannel]]
    output_path: str
    param_chip: StereoChip
    config: Config

    class Config:
        arbitrary_types_allowed = True


class RegisterOutput(BaseModel):
    gene_chip_box: Optional[ChipBoxInfo] = None
    MatrixTemplate: Optional[np.ndarray] = None
    info: RegistrationOutput

    class Config:
        arbitrary_types_allowed = True


def run_register(
        reg_in: RegisterInput
) -> RegisterOutput:
    """
    这个模块的任务就是对配准整体逻辑的整合，返回一个下游要用的配准参数
    这里有以下几种情况：
    1. if图，返回reuse图的配准参数
    2. 影像图+矩阵：前置配准、重心法、芯片框配准
    3. 影像图+影像图：暂不支持

    返回（RegisterOutput）：配准参数
    """
    param_chip = reg_in.param_chip
    image_file = reg_in.image_file
    files = reg_in.files
    channel_images = reg_in.channel_images
    cur_f_name = reg_in.cur_f_name
    output_path = reg_in.output_path
    config = reg_in.config
    sn = param_chip.chip_name

    reg_out_dct = {}
    if image_file.registration.reuse != -1:
        f_name = files[image_file.registration.reuse].get_group_name(sn=sn)
        info = channel_images[f_name].get_registration()
        clog.info('Get registration param from ipr')
        reg_out_dct['info'] = info
    else:
        # TODO 配准前置暂关
        #  11/22 by lizepeng
        # if self._channel_images[file_tag].Register.Method == AlignMode.Template00Pt.name:  # 先前做了前置配准
        #     # 从ipr获取配准参数
        #     info = self._channel_images[file_tag].get_registration()
        #     clog.info('Get registration param from ipr')
        # else:
        fixed = files[image_file.registration.fixed_image]
        # 动图参数构建
        g_name = image_file.get_group_name(sn=sn)
        param1 = channel_images[g_name]

        moving_image = ChipFeature(
            tech_type=image_file.tech,
        )
        moving_image.tech_type = image_file.tech
        moving_image.set_mat(cur_f_name.transformed_image)
        # 这里建议不要去ipr读，而是
        if param1.QCInfo.TrackCrossQCPassFlag:
            moving_image.set_template(param1.Stitch.TransformTemplate)  # param1.transform_template_info
        if param1.QCInfo.ChipDetectQCPassFlag:
            moving_image.set_chip_box(param1.Stitch.TransformChipBBox.get())

        # 静图参数构建
        if fixed.is_matrix:
            # 场景1：静图是矩阵
            mfe = MatrixFeatureExtract(
                output_path=output_path,
                image_file=fixed,
                m_naming=naming.DumpMatrixFileNaming(
                    sn=sn,
                    m_type=fixed.tech.name,
                    save_dir=output_path
                )
            )
            mfe.set_chip_param(param_chip)
            mfe.set_config(config)
            mfe.extract4stitched()

            fixed_image = ChipFeature(
                tech_type=fixed.tech,
                template=mfe.template,
                chip_box=mfe.chip_box,
            )
            fixed_image.set_mat(mfe.mat)
            # channel_images[g_name].Register.MatrixTemplate = mfe.template.template_points
            reg_out_dct['MatrixTemplate'] = mfe.template.template_points
            # channel_images[g_name].Register.GeneChipBBox.update(fixed_image.chip_box)
            reg_out_dct['gene_chip_box'] = fixed_image.chip_box
        else:
            raise Exception("Not supported yet")

        info, temp_info = registration(
            moving_image=moving_image,
            fixed_image=fixed_image,
            ref=param_chip.fov_template,
            from_stitched=False,
            qc_info=(param1.QCInfo.TrackCrossQCPassFlag, param1.QCInfo.ChipDetectQCPassFlag)
        )
        if info is not None:
            reg_out_dct['info'] = info
        # TODO: 那这里temp_info不记录到ipr吗，这里需要改下，外面现在默认info一定存在的
        if temp_info is not None:
            temp_info.register_mat.write(
                os.path.join(output_path, f"{sn}_chip_box_register.tif")
            )
            np.savetxt(
                os.path.join(output_path, f"{sn}_chip_box_register.txt"),
                temp_info.offset
            )
    reg_out = RegisterOutput(**reg_out_dct)
    return reg_out
