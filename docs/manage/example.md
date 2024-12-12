case 1:
Stereo-seq T FF
ssDNA
```shell
PYTHONPATH=/media/Data/dzh/code/github_cellbin2 CUDA_VISIBLE_DEVICES=0 python cellbin2/cellbin_pipeline.py \
-c SS200000135TL_D1 \
-i /media/Data/dzh/data/cellbin2/demo_data/product_demo/SS200000135TL_D1/SS200000135TL_D1.tif \
-s ssDNA \
-o /media/Data/dzh/data/cellbin2/test/SS200000135TL_D1 \
-k "Stereo-seq T FF V1.2"
```