import os
from typing import List, Any, Tuple, Union
import numpy as np
import shutil

from PIL import ImageFile, Image

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from cellbin2.utils import clog
from cellbin2.contrib.alignment.basic import transform_points
from cellbin2.image import cbimwrite
from cellbin2.utils.common import TechType
from cellbin2.contrib.alignment import ChipFeature, RegistrationInput, get_alignment_00, ChipBoxInfo, RegistrationOutput
from cellbin2.modules.metadata import ProcFile
from cellbin2.modules import naming, run_state

from cellbin2.utils import ipr
from cellbin2.modules.extract.base import FeatureExtract
from cellbin2.contrib.mask_manager import BestTissueCellMask, MaskManagerInfo
from cellbin2.contrib.tissue_segmentor import TissueSegInputInfo
from cellbin2.utils.pro_monitor import process_decorator
from cellbin2.image import CBImage, cbimread
from cellbin2.modules.extract.transform import run_transform
from cellbin2.utils.common import iPlaceHolder


class ImageFeatureExtract(FeatureExtract):
    def __init__(self, image_file: ProcFile, output_path: str, sn: str, im_naming: naming.DumpImageFileNaming):
        super(ImageFeatureExtract, self).__init__(image_file, output_path)
        self._fov_wh = (2000, 2000)  # 初始裁图尺寸
        self._channel_image = ipr.IFChannel()  # Union[ipr.ImageChannel, ipr.IFChannel]
        # self._naming = naming.DumpImageFileNaming(image_file.file_path)
        self._naming: naming.DumpImageFileNaming = im_naming
        # self._state = run_state.ImageRunState(image_file.file_path, dst_dir=output_path)
        # self._state = state
        self.sn = sn

    def set_channel_image(self, ci: Union[ipr.ImageChannel, ipr.IFChannel]):
        self._channel_image = ci

    def transform2regist(self, info: RegistrationOutput):
        """
        将transform图（染色图/mask）变成配准图:模板变成regist模板
        """
        self._channel_image.update_registration(info)
        dct = {
            self._naming.transformed_image: self._naming.registration_image,
            self._naming.transform_cell_mask: self._naming.cell_mask,
            self._naming.transform_cell_mask_raw: self._naming.cell_mask_raw,
            self._naming.transform_tissue_mask: self._naming.tissue_mask,
            self._naming.transform_tissue_mask_raw: self._naming.tissue_mask_raw,
            # self._naming.transform_cell_correct_mask: self._naming.cell_correct_mask,
            self._naming.transformed_template: self._naming.register_template,
            self._naming.transformed_track_template: self._naming.register_track_template
        }
        for src, dst in dct.items():
            src_path = src
            dst_path = dst
            # if os.path.exists(dst_path):
            #     continue
            if os.path.exists(src_path):
                if os.path.splitext(src_path)[1] == ".txt":  # 或其他判断
                    points, _ = transform_points(
                        src_shape=self._channel_image.Stitch.TransformShape,
                        points=np.loadtxt(src_path),
                        rotation=(4 - info.counter_rot90) * 90,
                        flip=0 if info.flip else -1,
                        offset=info.offset
                    )
                    np.savetxt(dst_path, points)
                    if dst == self._naming.register_template:
                        self._channel_image.Register.RegisterTemplate = points
                    if dst == self._naming.register_track_template:
                        self._channel_image.Register.RegisterTrackTemplate = points
                else:
                    dst_image = cbimread(src_path).trans_image(
                        flip_lr=info.flip, rot90=info.counter_rot90, offset=info.offset, dst_size=info.dst_shape)
                    cbimwrite(dst_path, dst_image)
        # for img in [self._naming.transform_cell_mask, self._naming.transform_tissue_mask,
        #             self._naming.transform_cell_correct_mask]:
        #     img_path = os.path.join(self.output_path, img)
        #     if os.path.exists(img_path):
        #         os.remove(img_path)

    def extract4transform(self,):


        # tissue & cell seg
        final_tissue_mask = None
        final_cell_mask = None
        tissue_mask = None
        cell_mask = None
        if self._image_file.tissue_segmentation:
            if not self._naming.transform_tissue_mask_raw.exists():
                tissue_mask = self._tissue_segmentation(
                    image_path=self._naming.transformed_image,
                    res_path=self._naming.transform_tissue_mask_raw
                )
            else:
                tissue_mask_p = self._naming.transform_tissue_mask_raw
                tissue_mask = cbimread(tissue_mask_p, only_np=True)
            final_tissue_mask = tissue_mask
        if self._image_file.cell_segmentation:
            if not self._naming.transform_cell_mask_raw.exists():
                cell_mask = self._cell_segmentation(
                    image_path=self._naming.transformed_image,
                    res_path=self._naming.transform_cell_mask_raw
                )
            else:
                cell_mask_p = self._naming.transform_cell_mask_raw
                cell_mask = cbimread(cell_mask_p, only_np=True)
            final_cell_mask = cell_mask
        if tissue_mask is not None and cell_mask is not None:
            input_data = MaskManagerInfo(
                tissue_mask=tissue_mask,
                cell_mask=cell_mask,
                chip_box=self._channel_image.Stitch.TransformChipBBox.get(),
                method=1,
                stain_type=self._image_file.tech
            )
            btcm = BestTissueCellMask.get_best_tissue_cell_mask(input_data=input_data)
            final_tissue_mask = btcm.best_tissue_mask
            final_cell_mask = btcm.best_cell_mask
        if final_cell_mask is not None:
            cbimwrite(
                output_path=self._naming.transform_cell_mask,
                files=final_cell_mask
            )
        if final_tissue_mask is not None:
            cbimwrite(
                output_path=self._naming.transform_tissue_mask,
                files=final_tissue_mask
            )

    @process_decorator('GiB')
    def _tissue_segmentation(self, image_path: str, res_path: str):
        from cellbin2.contrib.tissue_segmentor import segment4tissue
        tissue_input = TissueSegInputInfo(
            weight_path_cfg=self._config.tissue_segmentation,
            input_path=image_path,
            stain_type=self._image_file.tech,
        )
        tissue_mask_output = segment4tissue(tissue_input)
        tissue_mask = tissue_mask_output.tissue_mask
        cbimwrite(res_path, tissue_mask)
        return tissue_mask

    @process_decorator('GiB')
    def _cell_segmentation(self, image_path: str, res_path: str):

        if self._image_file.tech == TechType.IF:
            from cellbin2.contrib import cellpose_segmentor

            cell_mask = cellpose_segmentor.segment4cell(
                input_path=image_path,
                cfg=self._config.cell_segmentation,
            )
        else:
            from cellbin2.contrib import cell_segmentor

            cell_mask, fast_mask = cell_segmentor.segment4cell(
                input_path=image_path,
                cfg=self._config.cell_segmentation,
                s_type=self._image_file.tech,
                fast=False, gpu=0)
        cbimwrite(res_path, cell_mask)
        return cell_mask

    def _adjust_mask(self, ):
        # TODO：更优的细胞分割结果输出
        clog.info('Adjust mask need update, please wait a moment..')
        pass

    @staticmethod
    def _expand_mask(mask: np.typing.NDArray, distance: int, process: int) -> np.typing.NDArray[np.uint8]:
        from cellbin2.contrib.fast_correct import run_fast_correct

        if distance < 1:
            clog.info(f"distance is: {distance} which is less than 0, return mask as it is")
            return mask

        fast_mask = run_fast_correct(
            mask=mask,
            dis=distance,
            process=process
        )
        return fast_mask

    def export_record(self, tag='record.ipr'):
        self._channel_image.write(os.path.join(self.output_path, tag), extra={})

    @property
    def channel_image(self, ):
        return self._channel_image


def main():
    from cellbin.modules.metadata import default_image
    from cellbin.utils.config import Config

    config_file = r'D:\01.code\cellbin2\dev\cellbin2\cellbin\config\cellbin.yaml'
    weights_root = r'D:\01.code\cellbin2\weights'

    image_file = default_image(file_path=r"D:\02.data\temp\A03599D1\fov_stitched_DAPI.tif",
                               tech_type='DAPI', clarity=False)
    ife = ImageFeatureExtract(image_file=image_file, output_path=r'D:\02.data\temp\A03599D1\cellbin2')
    ife.set_config(Config(config_file, weights_root))
    ife.extract4stitched()
    param = ife.channel_image.Register
    ife.extract4transform(scale=(1 / param.ScaleX, 1 / param.ScaleY),
                          rotation=param.Rotation, offset=(param.OffsetX, param.OffsetY))
    ife.export_record()


if __name__ == '__main__':
    main()
