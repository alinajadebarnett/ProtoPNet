#!/bin/bash
#SBATCH --gres=gpu:1 -p compsci-gpu --job-name=combo

source /home/home2/ct214/virtual_envs/ml/bin/activate

nvidia-smi

echo "Begun generating explanations."

MODELFOLDER=/usr/xtmp/ProtoPNet_retrain/ProtoPNet/saved_models/vgg16/1215_vgg16_retrain/
MODELNAME=20_9push0.7337.pth

for FILENAME in Horned_Puffin_0003_101045.JPEG
do
srun -u python local_analysis.py -test_img_name "$FILENAME" \
                               -test_img_dir '/usr/xtmp/ProtoPNet_retrain/CUB_200_2011/datasets/cub200_cropped/test_cropped/106.Horned_Puffin/' \
                               -test_img_label 105 \
                               -test_model_dir "$MODELFOLDER/" \
                               -test_model_name "$MODELNAME"

srun -u python3 local_analysis_vis.py -local_analysis_directory "$MODELFOLDER/$FILENAME/"
done

echo "End."
