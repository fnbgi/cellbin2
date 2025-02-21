<div align="center">
  <h1 align="center">
    Stitch: Prepared by cell bin research group 
  </h1>
</div>

## Installation
As an independent module, stitch modules can be directly referenced by cellbin-v2 or installed by wheel

### Referenced by cellbin-v2
```python
from cellbin2.contrib.stitch.mfws import stitch_image

# 
def stitch_image(
        image_path: str = '',
        process_rule: dict = None,
        overlap: str = '0.1',
        name: str = '',
        fuse_flag: bool = True,
        scope_flag: bool = False,
        down_size: int = 1,
        row_slice: str = '-1_-1',
        col_slice: str = '-1_-1',
        output_path: str = '',
        stereo_data: str = 'cellbin',
        file_type: str = '',
        debug: bool = False
) -> Union[None, np.ndarray]:
    """
    Image stitch function
    The format of the small image is as follows：
    -------------------------
       0_0, 0_1, ... , 0_n
       1_0, 1_1, ... , 1_n
       ...
       m_0, m_1, ... , m_n
    -------------------------
    Of which, m and n denote row and col

    Args:
        image_path:

        name: image name

        process_rule:

        overlap: scope overlap

        fuse_flag: whether or not fuse image

        scope_flag: scope stitch | algorithm stitch

        down_size: down-simpling size

        row_slice: means stitch start row and end row,
             if image has 20 rows and 20 cols, row = '0_10' express only stitch row == 0 -> row == 9,
             same as numpy slice, and other area will not stitch

        col_slice: same as 'row'

        output_path:

        stereo_data:
            - V3:
            - dolphin:
            - T1:
            - cellbin:

        file_type: re lambda, like '*.A.*.tif'

        debug:

    Returns:

    Examples:
        >>>

    """
```
### Wheel used 
```python
pip install MFWS-0.0.1-py3-none-any.whl

from mfws.stitch import stitch_image

# call method as above
```

### Console used 

#### command description
```python
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--input", action="store", dest="input", type=str, required=True,
                    help="Tar file / Input image dir.")

# scope overlap
parser.add_argument("-overlap", "--overlap", action="store", dest="overlap", type=str, required=False,
                    default='0.1', help="Overlap - 0.1 or 0.1_0.1 .")

# scope stitch or algorithm stitch
parser.add_argument("-s", "--scope", action = "store_true", dest = "scope",
                    required = False, help = "Scope stitch.")

# fuse
parser.add_argument("-f", "--fuse", action = "store_true", dest = "fuse", required = False, help = "Fuse.")

# down-sampling
parser.add_argument("-d", "--down", action = "store", dest = "down", type = float, required = False,
                    default = 1, help = "Down-sampling.")

# block selection
parser.add_argument("-row", "--row", action = "store", dest = "row", type = str, required = False,
                    default = '-1_-1', help = "Image select block - row.")
parser.add_argument("-col", "--col", action = "store", dest = "col", type = str, required = False,
                    default = '-1_-1', help = "Image select block - col.")

parser.add_argument("-n", "--name", action="store", dest="name", type=str, required=False,
                    default = '', help="Name.")
parser.add_argument("-o", "--output", action="store", dest="output", type=str, required=False,
                    default = '', help="Result output dir.")

parser.add_argument("-debug", "--debug", action = "store_true", dest = "debug", required = False, help = "debug.")

"""
interface by stereo data --
   V3: 
   dolphin:
   T1:
   cellbin:
any case is fine. 
"""
parser.add_argument("-id", "--id", action = "store", dest = "id", type = str, required = False,
                    default = 'cellbin', help = "Stereo data id.")
parser.add_argument("-file_type", "--file_type", action = "store", dest = "file_type", type = str,
                    required = False, default = '', help = "File name -- such as '*.A.*.tif'.")
```


## Example

### Using cellbin-v2 or wheel import
```python
# Data format -- dict
image_path = {
    '0000_0000': '1.tif', '0000_0001': '2.tif',
    '0001_0000': '3.tif', '0001_0001': '4.tif',
    '0002_0000': '5.tif', '0002_0001': '6.tif',
    '0003_0000': '7.tif', '0003_0001': '8.tif',
}

# Data format -- absolute path
image_path = '/data/image_path'
# the directory format is as follows
'''
    - 0000_0000.tif
    - 0000_0001.tif
    - 0001_0000.tif
    - 0001_0001.tif
    - 0002_0000.tif
    - 0002_0001.tif
    - 0003_0000.tif
    - 0003_0001.tif
'''

# if scope_flag is False, will using cellbin-stitch modules, otherwise using microscope stitch
# Overlap -- Please fill in according to the preset parameters of the microscope
image = stitch_image(
    image_path = image_path,
    overlap = 0.1,
    scope_flag = False
)

# if want to save image and custom image name 
stitch_image(
    image_path = image_path,
    overlap = 0.1,
    name = 'image',
    scope_flag = False,
    output_path = "*.tif"
)

# if want to slice image, only using scope coordinate stitch 1_0, 1_1, 2_2, 2_1, 
# and not need fuse image.
stitch_image(
    image_path = image_path,
    overlap = 0.1,
    name = 'image',
    scope_flag = True,
    row = '1_3',
    col = '0_2',
    fuse = False,
    output_path = "*.tif"
)
```

### Using console
Stereo data
```python
# if want using console stitch image

# Only supports image path input

stitch 
-i '/data/image_path' 
-o '/data/output_path' 
-overlap 0.1
-s  # if not need scope stitch, just not need to fill in
-f  # if not need fuse image, just not need to fill in
-n image 
-row 1_3 
-col 0_2
-d 2
```

V3 data
```python
stitch 
-i '/data/image_path' 
-o '/data/output_path' 
-n V3 
-overlap 0 
-row 15_17 
-col 15_17 
-file_type *.A.*.tif 
-s 
-id V3
```

Dolphin data
```python
stitch 
-i '/data/image_path'  
-o '/data/output_path'  
-n dolphin 
-overlap 0.1_0 
-row 1_10 
-col 1_10 
-file_type *.A.*.tif 
-id dolphin 
```

T1 data
```python
stitch 
-i '/data/image_path'  
-o '/data/output_path'  
-n T1 
-overlap 0.07925_0.07925 
-row 1_10 
-col 1_10 
-file_type *.A.*.tif 
-id T1 
```
