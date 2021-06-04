#!/bin/bash

echo "Begun generating explanations."

MODELFOLDER=/content/drive/My\ Drive/Research\ Stuff/RML2021/code_demo/saved_models/vgg16/1215_vgg16_retrain
MODELNAME=20_9push0.7337.pth

for FILENAME in Horned_Puffin_0003_101045.JPEG
do
python3 local_analysis.py -test_img_name "$FILENAME" \
                          -test_img_dir '/content/drive/My Drive/Research Stuff/RML2021/code_demo/data/CUB_200_2011/datasets/cub200_cropped/test_cropped/106.Horned_Puffin/' \
                               -test_img_label 105 \
                               -test_model_dir "$MODELFOLDER/" \
                               -test_model_name "$MODELNAME"

python3 local_analysis_vis.py -local_analysis_directory "$MODELFOLDER/$FILENAME/" \
                              -test_image_directory '/content/drive/My Drive/Research Stuff/RML2021/code_demo/data/CUB_200_2011/datasets/cub200_cropped/test_cropped/'
done

echo "End."
