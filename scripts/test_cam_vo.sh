OUTPUT_DIR=vo_results/
DATASET_DIR=/media/bjw/Disk/Dataset/kitti_odom_test/sequences/

POSE_NET=models/exp_pose_model_best.pth.tar

python3 test_vo.py \
--img-height 256 --img-width 832 \
--sequence 09 \
--pretrained-posenet $POSE_NET --dataset-dir $DATASET_DIR --output-dir $OUTPUT_DIR


python3 ./kitti_eval/eval_odom.py --result=$OUTPUT_DIR --align='7dof'