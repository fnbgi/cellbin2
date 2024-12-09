import numpy as np
from pydantic import BaseModel, Field
from typing import List, Any, Tuple, Union, Optional

from cellbin2.image import cbimread, CBImage
from cellbin2.utils.ipr import ImageChannel, IFChannel
from cellbin2.contrib.alignment.basic import transform_points
from cellbin2.contrib.alignment.basic import ChipBoxInfo


class TransformInput(BaseModel):
    stitch_image: str
    image_info: Union[IFChannel, ImageChannel]
    scale: Tuple[float, float]
    rotation: float
    offset: Tuple[float, float]
    if_track: bool

    class Config:
        arbitrary_types_allowed = True


class TransformOutput(BaseModel):
    transform_image: CBImage
    TransformShape: Tuple[int, int]
    TrackPoint: Optional[np.ndarray] = None
    TransformTemplate: Optional[np.ndarray] = None
    TransformTrackPoint: Optional[np.ndarray] = None
    chip_box_info: Optional[ChipBoxInfo] = None

    class Config:
        arbitrary_types_allowed = True


def run_transform(inputs: TransformInput):
    scale, rotation, offset = inputs.scale, inputs.rotation, inputs.offset
    transform_image = cbimread(inputs.stitch_image).trans_image(
        scale=scale, rotate=rotation, offset=offset
    )
    trans_im_shape = transform_image.shape
    if inputs.if_track:
        info = inputs.image_info.Stitch.ScopeStitch
        stitch_template = inputs.image_info.Stitch.TemplatePoint
        qc_template = inputs.image_info.QCInfo.CrossPoints.stack_points
        stitch_trans_template, _ = transform_points(
            src_shape=(info.GlobalHeight, info.GlobalWidth),
            scale=scale, rotation=-rotation,
            points=stitch_template,
            offset=offset
        )

        qc_trans_template, _ = transform_points(
            src_shape=(info.GlobalHeight, info.GlobalWidth),
            scale=scale, rotation=-rotation,
            points=qc_template,
            offset=offset
        )
        stitch_chip_box = inputs.image_info.QCInfo.ChipBBox.get().chip_box
        trans_chip_box, _ = transform_points(
            src_shape=(info.GlobalHeight, info.GlobalWidth),
            scale=(1 / inputs.image_info.QCInfo.ChipBBox.ScaleX,
                   1 / inputs.image_info.QCInfo.ChipBBox.ScaleY),
            rotation=-inputs.image_info.QCInfo.ChipBBox.Rotation,
            points=stitch_chip_box,
            offset=offset
        )
        trans_chip_box_info = ChipBoxInfo()
        trans_chip_box_info.set_chip_box(trans_chip_box)
        trans_chip_box_info.ScaleX, trans_chip_box_info.ScaleY = 1.0, 1.0
        trans_chip_box_info.Rotation = 0.0
        outs = TransformOutput(
            transform_image=transform_image,
            TransformShape=trans_im_shape,
            TrackPoint=qc_template,
            TransformTemplate=stitch_trans_template,
            TransformTrackPoint=qc_trans_template,
            chip_box_info=trans_chip_box_info
        )
    else:
        outs = TransformOutput(
            transform_image=transform_image,
            TransformShape=trans_im_shape
        )
    return outs

