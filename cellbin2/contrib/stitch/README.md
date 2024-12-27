<div align="center">
  <h1 align="center">
    Stitch: Prepared by cell bin research group 
  </h1>
</div>

## Installation
As an independent module, stitch modules can be directly referenced by cellbin-v2 or installed by wheel

referenced by cellbin-v2
```shell
from cellbin2.contrib.stitch.stitch import stitch_image

# 
def stitch_image(
        image_path: str = '',
        overlap: float = 0.1,
        name: str = '',
        scope_flag: bool = False,
        output_path: str = ''
) -> Union[None, np.ndarray]:
    """
    图像拼接函数
    小图排列格式如下：
    -------------------------
       0_0, 0_1, ... , 0_n
       1_0, 1_1, ... , 1_n
       ...
       m_0, m_1, ... , m_n
    -------------------------
    其中, m 和 n 分别表示 row 和 col

    Args:
        image_path: {'0000_0000': '*.tif', ...}
        name: 图像命名 可不填
        overlap: 显微镜预设overlap
        scope_flag: 是否直接使用显微镜拼接
        output_path:

    Returns:

    """
```
used wheel
```shell
from stitch import stitch_image

# call method as above
```

## Example
```shell
image_path_dict = {
  '0000_0000': '1.tif',
  '0000_0001': '2.tif',
  '0001_0000': '3.tif',
  '0001_0001': '4.tif',
}

# if scope_flag is False, will using cellbin-stitch modules, otherwise using microscope stitch
# Overlap -- Please fill in according to the preset parameters of the microscope
image = stitch_image(
    image_path = image_path_dict,
    overlap = 0.1,
    scope_flag = False
)

# if want to save image and custom image name 
stitch_image(
    image_path = image_path_dict,
    overlap = 0.1,
    name = 'image',
    scope_flag = False,
    output_path = "*.tif"
)
```