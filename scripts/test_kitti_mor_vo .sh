DATASET_DIR=/root/Dataset/kitti_odom_test/sequences/09/images_2
OUTPUT_DIR=vo_results/

POSE_NET=models/exp_pose_model_MorVo.pth.tar

python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR --classes 0 1 2 3 4 5

python test_vo.py \
--img-height 256 --img-width 832 \
--sequence 10 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR --classes 0 1 2 3 4 5

python ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'