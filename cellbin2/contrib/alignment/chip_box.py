# -*- coding: utf-8 -*-
import numpy as np
from typing import Union

from cellbin2.contrib.param import ChipFeature
from cellbin2.contrib.alignment.basic import Alignment, RegistrationInfo, AlignMode
from cellbin2.utils import clog
from cellbin2.image import CBImage


class ChipAlignment(Alignment):
    """
    满足TissueBin需求：利用2模态芯片角为特征点，计算变换参数，实现配准。误差约100pix
    """
    def __init__(self,):
        super(ChipAlignment, self).__init__()

        self.register_matrix: np.matrix = None
        self.transforms_matrix: np.matrix = None

        self.rot90_flag = True
        self.no_trans_flag = False

    def registration_image(self,
                           file: Union[str, np.ndarray, CBImage]):
        """ 对待变换的图像，调用图像处理库按照对齐参数，返回变换后的图 """

        if not isinstance(file, CBImage):
            image = cbimread(file)
        else:
            image = file

        if self.no_trans_flag:
            # TODO
            result = None
        else:
            result = image.trans_image(
                scale=[1 / self._scale_x, 1 / self._scale_y],
                rotate=self._rotation,
                rot90=self.rot90,
                offset=self.offset,
                dst_size=self._fixed_image.mat.shape,
                flip_lr=self.hflip
            )

        return result

    def align_stitched(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        """

        Args:
            fixed_image:
            moving_image:

        Returns:

        """
        self._rotation = -moving_image.chip_box.Rotation
        self._scale_x = moving_image.chip_box.ScaleX
        self._scale_y = moving_image.chip_box.ScaleY
        self._fixed_image = fixed_image

        if self.no_trans_flag:
            self.align_transformed(fixed_image, moving_image)
        else:
            transformed_image = self.transform_image(file=moving_image.mat)

            transformed_feature = ChipFeature()
            transformed_feature.set_mat(transformed_image)

            trans_mat = self.get_coordinate_transformation_matrix(
                moving_image.mat.shape,
                [1 / self._scale_x, 1 / self._scale_y],
                self._rotation
            )

            trans_points = self.get_points_by_matrix(moving_image.chip_box.chip_box, trans_mat)
            transformed_feature.chip_box.set_chip_box(trans_points)

            self.align_transformed(fixed_image, transformed_feature)

    def align_transformed(
            self,
            fixed_image: ChipFeature,
            moving_image: ChipFeature
    ):
        """

        Args:
            fixed_image:
            moving_image:

        Returns:

        """
        self._fixed_image = fixed_image
        coord_index = [0, 1, 2, 3]
        if self.rot90_flag: range_num = 4
        else: range_num = 1

        if self.hflip:
            new_box = self.transform_points(points = moving_image.chip_box.chip_box,
                                            shape = moving_image.mat.shape, flip=0)
            new_mi = np.fliplr(moving_image.mat.image)

        register_info = dict()
        for index in range(range_num):
            register_image, M = self.get_matrix_by_points(
                new_box[coord_index, :], fixed_image.chip_box.chip_box,
                True, new_mi, fixed_image.mat.shape
            )

            lu_x, lu_y = map(int, fixed_image.chip_box.chip_box[0])
            rd_x, rd_y = map(int, fixed_image.chip_box.chip_box[2])

            _wsi_image = register_image[lu_y: rd_y, lu_x:rd_x]
            _gene_image = fixed_image.mat.image[lu_y: rd_y, lu_x:rd_x]

            ms = self.multiply_sum(_wsi_image, _gene_image)
            # _, res = self.dft_align(_gene_image, _wsi_image, method = "sim")

            clog.info(f"Rot{index * 90}, Score: {ms}")
            register_info[index] = {"score": ms, "mat": M}  # , "res": res}

            coord_index.append(coord_index.pop(0))

        best_info = sorted(register_info.items(), key = lambda x: x[1]["score"], reverse = True)[0]
        if self.rot90_flag: self._rot90 = range_num - best_info[0]
        _mat = self.get_coordinate_transformation_matrix(
            moving_image.mat.shape, [1, 1], 90 * best_info[0]
        )
        _box = self.get_points_by_matrix(new_box, _mat)
        _box = self.check_border(_box)
        # self._offset = (fixed_image.chip_box.chip_box[0, 0] - (_box[0, 0] + _box[1, 0]) / 2,
        #                 fixed_image.chip_box.chip_box[0, 1] - (_box[0, 1] + _box[3, 1]) / 2)

        self._offset = (fixed_image.chip_box.chip_box[0, 0] - _box[0, 0],
                        fixed_image.chip_box.chip_box[0, 1] - _box[0, 1])


def chip_align(
        moving_image: ChipFeature,
        fixed_image: ChipFeature,
        from_stitched: bool = True
) -> RegistrationInfo:
    """
    :param moving_image: 待配准图，通常是染色图（如ssDNA、HE）
    :param fixed_image: 固定图
    :param from_stitched: 从拼接图开始

    Returns:

    """
    ca = ChipAlignment()
    if from_stitched: ca.align_stitched(fixed_image=fixed_image, moving_image=moving_image)
    else: ca.align_transformed(fixed_image=fixed_image, moving_image=moving_image)

    info = RegistrationInfo(**{
            'offset': tuple(list(ca.offset)),
            'counter_rot90': ca.rot90,
            'flip': ca.hflip,
            'register_score': ca.score,
            'register_mat': ca.registration_image(moving_image.mat),
            'method': AlignMode.TemplateCentroid
        }
    )

    return info


if __name__ == '__main__':
    from cellbin2.image import cbimread, cbimwrite
    from cellbin2.contrib.param import TemplateInfo
    from cellbin2.utils.common import TechType
    from cellbin2.contrib.chip_detector import ChipParam, detect_chip
    from cellbin2.matrix.box_detect import detect_chip_box

    # 移动图像信息
    moving_image = ChipFeature()
    moving_image.tech_type = TechType.DAPI
    moving_mat = cbimread(r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif")
    moving_image.set_mat(moving_mat)

    cfg = ChipParam(
        **{"DAPI_stage1_weights_path":
            r"E:/03.users/liuhuanlin/01.data/cellbin2/weights\chip_detect_obb8n_640_SD_202409_pytorch.onnx",
            "DAPI_stage2_weights_path":
            r"E:/03.users/liuhuanlin/01.data/cellbin2/weights\chip_detect_yolo8m_1024_SD_202409_pytorch.onnx"})

    # file_path = r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI.tif"
    m_info = detect_chip(moving_mat.image, cfg=cfg, stain_type=TechType.DAPI, actual_size=(19992, 19992))
    moving_image.set_chip_box(m_info)

    # 固定对象信息
    fixed_image = ChipFeature()
    fixed_image.tech_type = TechType.Transcriptomics
    fixed_image.set_mat(r"E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_gene.tif")

    f_info = detect_chip_box(fixed_image.mat.image)
    fixed_image.set_chip_box(f_info)

    result = chip_align(moving_image, fixed_image)
    print(result)
    cbimwrite(r'E:\03.users\liuhuanlin\01.data\cellbin2\stitch\A03599D1_DAPI_registbox.tif', result.register_mat)
