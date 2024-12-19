case1: only qc <br>
If kit type is Stereo-CITE T FF V1.0, you find Stereo-CITE T FF.json under cellbin2/config <br>
change run like follows
   
```shell
"run": {
    "qc": true,
    "alignment": false,
    "matrix_extract": false,
    "report": false,
    "annotation": false
  }
```
```shell
-c
A03599D1
-i
/media/Data1/user/dengzhonghan/data/cellbin2/demo_data/A03599D1/A03599D1_DAPI_fov_stitched.tif
-if
/media/Data1/user/dengzhonghan/data/cellbin2/demo_data/A03599D1/A03599D1_IF_fov_stitched.tif
-s
DAPI
-o
/media/Data1/user/dengzhonghan/data/cellbin2/random_test/A03599D1
-k
"Stereo-CITE T FF V1.0"
```

case2: only alignment & matrix extract <br>
This mode assumes the input image is aligned, so only run tissue and cell seg <br>
If kit type is Stereo-CITE T FF V1.0, you find Stereo-CITE T FF.json under cellbin2/config <br>
change run like follows
```shell
"run": {
    "qc": false,
    "alignment": true,
    "matrix_extract": true,
    "report": false,
    "annotation": false
  }
```

case 3: No matrix <br>
Stereo-seq T FF
ssDNA
```shell
-c SS200000135TL_D1 
-i /media/Data/dzh/data/cellbin2/demo_data/product_demo/SS200000135TL_D1/SS200000135TL_D1_fov_stitched_ssDNA.tif 
-s ssDNA 
-o /media/Data/dzh/data/cellbin2/test/SS200000135TL_D1 
-k "Stereo-seq T FF V1.2"
```