#@title libraries imports for all cells
import numpy as np

from PIL import Image

import os
import re
import warnings
warnings.filterwarnings('ignore')
import time

import torch
from torchvision.transforms.functional import crop

import load_models
import utils
import eval
import reid

device = "cuda" if torch.cuda.is_available() else "cpu"

'''
Description:
Get path to data and returns a list of all sequences names and the paths to annotations and sequences directories.

inputs:
data_dir - Path to data directory


outputs:
seq_dir - Path to sequences directory in data directory. Each sequence holds frames, first frame is used for query image.
seq_list - List of all sequences names.
anno_dir - Path to annotations directory in data directory. Each file in this directory holds the GT bboxes for the sequence with the same name.
'''
def init_data_paths(data_dir):
  seq_dir = os.path.join(data_dir, 'sequences')
  anno_dir = os.path.join(data_dir, 'annotations')

  seq_list = os.listdir(seq_dir)
  return seq_dir, seq_list, anno_dir


'''
Description:
Wrapper to run framework on VisDrone dataset.

inputs:
data_dir - Path to data directory
box_thresh - Thershold for GroundingDINO boxes detection.
query_prompt - Single word to use as prompt for GroundingDINO, Default is 'vehicles'.
num_ious - Number of rank-k values to hold.

outputs:
 mAP - Dictionary of mean Average Precision for each feature extraction model.
 acc - Dictionary of accuracy in range num_ious for each feature extraction model.
 tp - True positives.
 fp - False positives.
'''
def run_all_sequences(data_dir, models, log, box_thresh = 0.3, query_prompt = 'vehicles', num_ious = 10):
  # get sequences and annotations paths
  seq_dir, seq_list, anno_dir = init_data_paths(data_dir)
  log['number of images'] = 0 # for log file
  tp, fp = 0, 0
  acc = {'clip': np.zeros(num_ious), 'siglip': np.zeros(num_ious), 'dino': np.zeros(num_ious)}
  AP_list = {key:[] for key in acc.keys()}

  # run over all sequences
  for seq in seq_list:
    print(f'Working on sequence {seq}')
    print(f'Processed {log['number of images']} images so far')

    # init values for log file
    log['Sequance'].update({seq:{}})
    log['Sequance'][seq]['AP']= {}
    log['Sequance'][seq]['Images']={}

    # get list of GT bboxes for current sequence
    anno_file_path = os.path.join(anno_dir, seq + '.txt')
    target_bbox_gt_list = np.loadtxt(anno_file_path, delimiter=',')

    # list all the frames in current sequence
    cur_seq_dir = os.path.join(seq_dir, seq)
    images_list = sorted(os.listdir(cur_seq_dir))

    # get query image from the first frame in th sequence
    query_img_path = os.path.join(cur_seq_dir, images_list[0])
    query_img = Image.open(query_img_path)
    query_num = utils.get_num_from_filename(images_list[0]) - 1
    query_bbox = target_bbox_gt_list[query_num]
    query_image = crop(query_img, query_bbox[1], query_bbox[0], query_bbox[3], query_bbox[2])

    # run over all images in the sequence except the first one (query image)
    target_index_list = {key:[] for key in acc.keys()}
    for img_name in images_list[1:]:
      # update log file
      log['Sequance'][seq]['Images'].update({img_name:{}})  # for log file
      log['number of images'] +=1 # for log file

      # load base image to search for query
      base_img_path = os.path.join(cur_seq_dir, img_name)

      # run framework
      print(f'Processing image {img_name}')
      scores = reid.process_single_image(models, base_img_path, query_image, box_thresh, query_prompt= query_prompt)

      # evaluate the results of each model for the current image
      print(f'Evaluating results')
      for key, score in scores.items():
        # get target rank by IoU with GT bbox
        target_iou_rank = eval.get_target_index(score, img_name, target_bbox_gt_list)
        target_index_list[key].append(target_iou_rank.squeeze())
        log['Sequance'][seq]['Images'][img_name].update({key:target_iou_rank.item()+1})   # for log file
        if target_iou_rank < num_ious:
          acc[key][target_iou_rank] += 1
      if target_iou_rank == 0:
        tp += 1
      else:
        fp += 1

    # compute Average Precision for current sequence
    for key in target_index_list:
      AP = eval.compute_AP(target_index_list[key])
      AP_list[key].append(AP)
      log['Sequance'][seq]['AP'][key]= AP  # for log file

  print("AP_list: ", AP_list)
  mAP = {key: np.mean(value) for key,value in AP_list.items() if key in scores.keys()}

  return mAP, acc, tp, fp



def main():
    # choose configurations
    use_clip = True
    use_siglip = False
    use_dinov2 = True
    use_sam = False
    dino_box_thresh = 0.3 #threshold for GroudningDINO detection
    query_prompt = 'vehicles' # prompt word for GroundingDINO, only supports 'vehicles' or 'people' for now
    
    output_file = f'LOG Good {query_prompt} ' + ('CLIP ' if use_clip else "") + ('SigLip ' if use_siglip else "") + ('DinoV2 ' if use_dinov2 else "")
    output_file = output_file + ('Sam_masked ' if use_sam else '')+ "Thresholed_" + '{:.2f}'.format(dino_box_thresh)+".json"

    # dict for logs
    log = {'Configurations': {'Grounding Dino Threshold': dino_box_thresh,
                            'Clip embedding':use_clip,
                            'SigLip embedding': use_siglip,
                            'Dino v2 embbeding': use_dinov2,
                            'Sam2 masking': use_sam},
        'Start run time': time.strftime("%d.%m %H:%M", time.localtime()),
        'End run time': "",
        'mAP':{},
        'rank':{},
        'number of images':0,
        'Sequance':{}
    }

    # Load wanted models
    models = load_models.load_models(load_clip= use_clip,
                                load_siglip= use_siglip,
                                load_dino= use_dinov2,
                                load_sam= use_sam)


    # head data directory
    data_dir = '/data'

    log['Start run time'] = time.strftime("%d.%m %H:%M", time.localtime())

    # run algorithm on all sequences in data directory and get results metrics
    mAP, acc, tp, fp = run_all_sequences(data_dir, models, log, box_thresh= dino_box_thresh, query_prompt= query_prompt)
    total_queries = tp + fp
    rank = {key: value / total_queries for key, value in acc.items() if key in mAP}
    cum_rank = {key: np.cumsum(value) / total_queries for key, value in acc.items() if key in mAP}

    utils.save_results(log, mAP, cum_rank, output_file)

    print(f'mAP is {mAP}')
    print(f'rank is {rank}')
    print(f'cum_rank is {cum_rank}')

    
if __name__ == '__main__':
  main()