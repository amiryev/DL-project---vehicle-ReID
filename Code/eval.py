#@title libraries imports for all cells
import numpy as np

import torch
import torchvision

import utils

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Description:
Computes intersection over union of query image against all detection in base image

inputs:
query_box - GT Bounding box of query image.
bboxes - bounding boxes of all detected targets from GroundingDINO

outputs:
iou - List containing IoU of every detection with GT (max non-zero values should be 1)
'''
def compute_iou(query_box, bboxes):
  if len(bboxes)==0:
    iou= torch.tensor(0)
  else:
    iou = torchvision.ops.box_iou(bboxes, query_box.unsqueeze(0))
  return iou.squeeze()

'''
Description:
Get the  rank of the traget vehicle. Compute IoU with all detections, highest
IoU score is the target vehicle.

inputs:
scores - Dictionary containing similarity score, the cropped target image and its bounding box values.
img_name - File name for current image (suppose to contain image number in it)
target_bbox_gt_list - List of GT bboxes for all frames in the current sequence

outputs:
target_iou_ind - Integer. The rank of the target vehicle [0:#num_of_vehicles-1]
or -1 if vehicle not detected by GroundingDINO.
'''
def get_target_index(scores, img_name, target_bbox_gt_list):
  # Get index for GT bbox list from image filename
  img_num = utils.get_num_from_filename(img_name) - 1

  # Get the relevant GT bbox from list of all frames GT
  target_bbox_gt = torch.tensor(target_bbox_gt_list[img_num])

  # Convert from (x,y,w,h) to (xmin,ymin,xmax,ymax)
  target_bbox_gt[2] += target_bbox_gt[0]
  target_bbox_gt[3] += target_bbox_gt[1]

  # convert the dict struct to a list of all bboxes
  bboxes = torch.tensor(np.array([s[2] for s in scores]))

  # compute IoU of all targets with GT bbox and use it to find the target correct vehicle
  iou = compute_iou(target_bbox_gt, bboxes)
  if (iou == 0).all():
    # in case the correct target vehicle was not detected
    target_iou_ind = -1
  else:
    target_iou_ind = iou.argmax()

  return torch.tensor(target_iou_ind)

'''
Description:
Compute Average Precision for single sequence of images.

inputs:
target_indices - List of target's ranks from every frame in the sequence.

outputs:
AP - Average Precision.
'''
def compute_AP(target_indices):
  # change range from [0:N-1] to [1:N]
  p= np.array(target_indices) + 1
  # avoid deviding by 0
  p[p == 0] = -1
  # higher rank = lower score
  p = 1 / p
  # index -1 is score 0
  p[p == -1] = 0
  # score mean
  AP = p.mean()
  return AP


