!git clone https://github.com/Amin-Golden/MOR-SfMLearner.git
!cd MOR-SfMLearner
!pip install -r requirements.txt
!pip3 install pebble


DATASET_DIR= "../../input/kitti-odometry/sequences/"
OUTPUT_DIR= "vo_results/"

POSE_NET="models/exp_pose_model_best40.pth.tar"

!python test_mor_vo.py \
--img-height 256 --img-width 832 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR --classes 0 1 2 3 4 5

!python test_mor_vo.py \
--img-height 256 --img-width 832 \
--sequence 10 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR --classes 0 1 2 3 4 5



!python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'


!git pull

import shutil
shutil.make_archive("vo_results", 'zip',"./vo_results" )

<a href=./MOR-SfMLearner/checkpoints.zip> Download File </a>

import utils
display = utils.notebook_init()  # checks


!mkdir Dataset/
!mkdir Dataset/kitti_vo_256/

DATASET="../../input/kitti-odometry-woo"
TRAIN_SET="Dataset/kitti_vo_256/"
!python3 data/prepare_train_data.py $DATASET --dataset-format 'kitti_odom' --dump-root $TRAIN_SET --width 832 --height 256 --num-threads 4

!git pull
!sh scripts/train_resnet50_pose_256.sh
