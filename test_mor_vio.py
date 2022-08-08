import torch
import torch.backends.cudnn as cudnn

from imageio import imread, imsave
from skimage.transform import resize as imresize
import numpy as np
from path import Path
import argparse
from tqdm import tqdm


from inverse_warp import pose_vec2mat
from scipy.ndimage.interpolation import zoom

from inverse_warp import *

import models
import cv2
import os
import sys

# yolov5 
from pathlib import Path as Path1
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, time_sync

# EKF 

import pandas as pd
from numpy.linalg import inv
import matplotlib.pyplot as plt
import random

from rotations import Quaternion, skew_symmetric
from vis_tools import *

FILE = Path1(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path1(os.path.relpath(ROOT, Path1.cwd()))  # relative

parser = argparse.ArgumentParser(description='Script for visualizing depth map and masks',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--pretrained-posenet", required=True, type=str, help="pretrained PoseNet path")
parser.add_argument('--weights_obj', nargs='+', type=str, default=ROOT / 'models/YOLOv5m6-Argoverse.pt', help=' object detection model path(s)')

parser.add_argument("--img-height", default=256, type=int, help="Image height")
parser.add_argument("--img-width", default=832, type=int, help="Image width")
parser.add_argument("--no-resize", action='store_true', help="no resizing is done")

parser.add_argument("--dataset-dir", type=str, help="Dataset directory")
parser.add_argument("--output-dir", type=str, help="Output directory for saving predictions in a big 3D numpy file")
parser.add_argument("--img-exts", default=['png', 'jpg', 'bmp'], nargs='*', type=str, help="images extensions to glob")
parser.add_argument("--rotation-mode", default='euler', choices=['euler', 'quat'], type=str)
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--view-img', action='store_true', default=False, help='show results')

parser.add_argument("--sequence", default='09', type=str, help="sequence to test")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# vid = cv2.VideoCapture(2)

def load_tensor_image(img_r, args):
    
    img=img_r.astype(np.float32)
    h, w, _ = img.shape
    if (not args.no_resize) and (h != args.img_height or w != args.img_width):
        img = imresize(img, (args.img_height, args.img_width)).astype(np.float32)
    img = np.transpose(img, (2, 0, 1))
    

    tensor_img = ((torch.from_numpy(img).unsqueeze(0)/255-0.45)/0.225).to(device)
    return tensor_img,img


@torch.no_grad()

def main():
    
    args = parser.parse_args()


    # Load object detection model 
    model_obj = DetectMultiBackend(args.weights_obj, device=device, dnn=False, data='data/Argoverse.yaml', fp16=False)
    stride, names, pt = model_obj.stride, model_obj.names, model_obj.pt
    imgsz=(args.img_height, args.img_width)
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    #load posenet model
    weights_pose = torch.load(args.pretrained_posenet)
    pose_net = models.PoseResNet().to(device)
    pose_net.load_state_dict(weights_pose['state_dict'], strict=False)
    pose_net.eval()



    
    image_dir = Path(args.dataset_dir + args.sequence + "/image_2/")
    output_dir = Path(args.output_dir)
    output_dir.makedirs_p()

    test_files = sum([image_dir.files('*.{}'.format(ext)) for ext in args.img_exts], [])
    test_files.sort()


    ############################################################################################################
    ################# EKF init #################################################################################
    
    imu_dir="datasets/oxts" + args.sequence +".txt"
    imu=pd.read_csv(imu_dir, sep=" ", header=None, names=["lat", "lon","alt","roll","pitch","yaw","vn","ve","vf","vl","vu","ax","ay","az","af","al","au","wx","wy","wz","wf","wl","wu","pos_accuracy","vel_accuracy","navstat","numsats","posmode","velmode","orimode"])
    feature=["vf","ay","ax","az","wy","wx","wz","roll","pitch","yaw"]
    imu_f=imu[feature]
    imu_f = imu_f.values.tolist()
    imu_f = np.array(imu_f).T

    for k in range(1, imu_f[0,:].shape[0]): 
        imu_f[1, k - 1]=imu_f[1, k - 1] * -1
        imu_f[2, k - 1]=imu_f[2, k - 1] * -1
        imu_f[4, k - 1]=imu_f[4, k - 1] * -1
        imu_f[5, k - 1]=imu_f[5, k - 1] * -1
    

    # Covariance errors of the Acceleronmeter, Gyroscome and Camera
    var_imu_f = 0.1
    var_imu_w = 1.0
    var_cam = 1.0

    # Jacobian matrices
    g = np.array([0, -9.81,0 ])  # gravity
    l_jac = np.zeros([9, 6])
    l_jac[3:, :] = np.eye(6)  # motion model noise jacobian
    h_jac = np.zeros([3, 9])
    h_jac[:, :3] = np.eye(3)  # measurement model jacobian

    #### 3. Initial Values #########################################################################

    ################################################################################################
    # Let's set up some initial values for our ES-EKF solver.
    ################################################################################################
    p_imu_est = np.zeros([imu_f[0,:].shape[0], 3])
    p_est = np.zeros([imu_f[0,:].shape[0], 3])  # position estimates
    v_est = np.zeros([imu_f[0,:].shape[0], 3])  # velocity estimates
    q_est = np.zeros([imu_f[0,:].shape[0], 4])  # orientation estimates as quaternions
    p_cov = np.zeros([imu_f[0,:].shape[0], 9, 9])  # covariance matrices at each timestep
    
    # Set initial values
    p_imu_est[0] = [0,0,0]
    p_est[0] = [0,0,0] # Start the position at the first known orientation provided by the ground truth
    v_est[0] = np.zeros(3) # Start velocity stimes at cero
    q_est[0] = Quaternion(w=1, x=0, y=0, z=0).to_numpy()
    p_cov[0] = np.eye(9)  # covariance of estimate
    cam_i = 0 # Count camera updates
    def scale_lse_solver(X, Y):
        """Least-sqaure-error solver
        Compute optimal scaling factor so that s(X)-Y is minimum
        Args:
            X (KxN array): current data
            Y (KxN array): reference data
        Returns:
            scale (float): scaling factor
        """
        scale = np.sum(X * Y)/np.sum(X ** 2)
        return scale

    #### 4. Measurement Update #####################################################################

    def measurement_update(sensor_var, p_cov_check, y_k, p_check, v_check, q_check):
        # Compute Kalman Gain
        R = np.diag([sensor_var, sensor_var, sensor_var]) # Measurement covariance matrix calculation

        K = p_cov_check.dot(h_jac.T).dot(inv(h_jac.dot(p_cov_check).dot(h_jac.T) + R)) #Kalman gain calculation

        # Compute error state
        error_x = K.dot(y_k - p_check) # Error state computation

        # Correct predicted state
        p_check = p_check + error_x[0:3] 
        v_check = v_check + error_x[3:6]
        q_check = Quaternion(axis_angle = error_x[6:9]).quat_mult(q_check)

        # Compute corrected covariance
        p_cov_check = (np.eye(9) - K.dot(h_jac)).dot(p_cov_check)

        return p_check, v_check, q_check, p_cov_check


    #### 5. Main Filter Loop #######################################################################

    p_check = p_est[0] # Position check
    p_imu = p_est[0] # Position check
    v_check = v_est[0] # Velocity check
    q_check = q_est[0] # Orientation check
    p_cov_check = p_cov[0]

    f_jac = np.eye(9) # Jacobian matrix initialization
    Q_imu = np.diag([var_imu_f, var_imu_f, var_imu_f, var_imu_w, var_imu_w, var_imu_w]) # Q variance matrix

    xyz_pred = [[0,0,0],[0,0,0]]
    xyz_ref = [[0,0,0],[0,0,0]]

    scale=1



   ################### end EKF init ###########################################################################
    print('{} files to test'.format(len(test_files)))
    
    
    # object detection Dataloader
    webcam=False
    
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(image_dir, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(image_dir, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model_obj.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

    # load GT and imu data
   
    global_pose = np.eye(4)
    poses = [global_pose[0:3, :].reshape(1, 12)]
    k=0
    for path, im, im0s, vid_cap, s in dataset:
        t1 = time_sync()
        im = torch.from_numpy(im).to(device)
        im = im.half() if model_obj.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        t2 = time_sync()
        dt[0] += t2 - t1

        # Inference
        pred = model_obj(im, augment=False, visualize=False)
        t3 = time_sync()
        dt[1] += t3 - t2

        # NMS
        pred = non_max_suppression(pred, args.conf_thres, args.iou_thres, args.classes, args.agnostic_nms, max_det=args.max_det)
        dt[2] += time_sync() - t3

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0  # for save_crop
            annotator = Annotator(im0, line_width=-1, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                
                for *xyxy, conf, cls in reversed(det):

                    c = int(cls)  # integer class
                    label = None 
                    annotator.box_label(xyxy, label, color=colors(c, True))

            # Stream results
            im0 = annotator.result()
            if args.view_img:
                if p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

        if k==0:
            tensor_img1,img = load_tensor_image(im0, args)
            k=k+1
            cv2.imwrite("test.jpg", im0)
            print("first image")
        else:
            tensor_img2,img = load_tensor_image(im0, args)
            
            pose = pose_net(tensor_img1, tensor_img2)

            pose_mat = pose_vec2mat(pose).squeeze(0).cpu().numpy()
            pose_mat = np.vstack([pose_mat, np.array([0, 0, 0, 1])])
           
            trajectory = [np.array([0, 0, 0])]
            trajectory = pose_mat[3,0:3].T
            delta_t = 0.1
            print("pose_mat first",pose_mat[0:3,3])

            # Update state with IMU inputs

            C_ni = Quaternion(*q_check).to_mat() #Rotation matrix associated with the current vehicle pose (Computed from the quaternion)
            p_imu = p_imu + (delta_t * v_check) + (((delta_t**2) / 2) * (C_ni.dot(imu_f[1:4, k ]) + g))
            p_check = p_check + (delta_t * v_check) + (((delta_t**2) / 2) * (C_ni.dot(imu_f[1:4, k ]) + g)) # Position calculation
            v_check = v_check + (delta_t * (C_ni.dot(imu_f[1:4, k ]) + g)) #velocity calculation
            q_check = Quaternion(axis_angle = imu_f[4:7, k ] * delta_t).quat_mult(q_check) #Quaternion calculation (Current orientation)

            # Linearize Motion Model
            F = f_jac # F matrix value assignation
            F[0:3,3:6] = np.eye(3) * delta_t 
            #F[3:6,6:9] = -1 * skew_symmetric(C_ni.dot(imu[1:4, k - 1])) * delta_t 
            F[3:6,6:9] = -1 * C_ni.dot(skew_symmetric(imu_f[1:4, k ])) * delta_t # This line is the forum suggestion and works much better
            F[6:9,6:9] = Quaternion(axis_angle =(imu_f[4:7, 0]+ imu_f[4:7, k ] )* delta_t).to_mat().T # This line is the forum suggestion and works much better

            Q = Q_imu * (delta_t**2) # Variance calculation in discrete time

            # Propagate uncertainty
            p_cov_check = F.dot(p_cov_check).dot(F.T) + l_jac.dot(Q).dot(l_jac.T) #Variance calculation

            # Check availability of Cam measurements

            
                
            p_check, v_check, q_check, p_cov_check = measurement_update(var_cam, p_cov_check, trajectory, p_check, v_check, q_check)
                    
            # Save current states
            p_imu_est[k] = p_imu
            p_est[k] = p_check
            v_est[k] = v_check
            q_est[k] = q_check
            p_cov[k] = p_cov_check

            Rot = Quaternion(*q_check).to_mat() #Rotation matrix associated with the current vehicle pose (Computed from the quaternion)

            pose_mat[0:3,0:3]=Rot
            pose_mat[3,0:3]=p_check.T
            
            print("pose_mat",pose_mat)
            global_pose = global_pose @  np.linalg.inv(pose_mat)

            poses.append(global_pose[0:3, :].reshape(1, 12))

            # update
            tensor_img1 = tensor_img2
            print("image k:", k)
            k=k+1
                
        # Print time (inference-only)
        LOGGER.info(f'{s}Done. ({t3 - t2:.3f}s)')
    
    # 
    poses = np.concatenate(poses, axis=0)
    filename = Path(args.output_dir + args.sequence + ".txt")
    np.savetxt(filename, poses, delimiter=' ', fmt='%1.8e')
        # Print results
    tt = tuple(x / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % tt)
    if args.update:
        strip_optimizer(args.weights_obj)  # update model (to fix SourceChangeWarning)



if __name__ == '__main__':
    main()
