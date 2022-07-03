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
