DATA_ROOT=/media/bjw/Disk
TRAIN_SET=Dataset/kitti_vo_256/
python train.py $TRAIN_SET \
--resnet-layers 50 \
--num-scales 1 \
-b4 -s0.1 -c0.5 --epochs 10 --epoch-size 10 --sequence-length 3 \
--with-ssim 1 \
--with-mask 0 \
--with-auto-mask 1 \
--with-pretrain 1 \
--pretrained-pose models/exp_pose_model_best40.pth.tar \
--log-output --with-gt \
--name resnet50_pose_256