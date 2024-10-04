#@title libraries imports for all cells
import numpy as np
import matplotlib.pyplot as plt

import os
import re
import time
import json

from GroundingDINO.groundingdino.util.inference import annotate

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


'''
Description:
Gets a string of image file name, returns the number in the filename

inputs:
filename - String of image name. Name of file should be in format img<image_num>.jpg


outputs:
__ - Image number.
'''
def get_num_from_filename(filename):
  regex = re.compile(r'\d+')
  return int(regex.findall(filename)[0])


'''
Description:
Display an image with its objects marked with bounding boxes and scores.

inputs:
image_source - Input image
boxes - Bounding boxes of each detection in the image.
logits - Score of each detection
phrases - List of string with token matching each detection

outputs:
'''
def display_annotated_frame(image_source, boxes, logits, phrases):
  annotated_frame = annotate(image_source=np.array(image_source), boxes=boxes, logits=logits, phrases=phrases)

  plt.figure()
  plt.title('Base image with detection annotated')
  plt.imshow(annotated_frame)
  plt.show()




def display_reid_results(query_image, scores, num_results=5):
  if num_results > len(scores):
    num_results = len(scores)

  print("Original image:")
  plt.figure()
  plt.imshow(query_image)
  plt.show()

  print(f'Top {num_results} Results:')
  for i in range(num_results):
    print(f'Object #{i+1} Score: {scores[i][0]}')
    plt.figure()
    plt.imshow(scores[i][1])
    plt.show()





'''
Description:
Saves results log to a file. Avoids overwriting files by adding date and time to filename.

inputs:
mAP - mean Average Precision score.
rank - List with rank-k scores (default is 10 rankings)
filename - Name of log file.

outputs:
--
'''
def save_results(log, mAP, rank, filename='untitled'):
  # add and edit data to log
  log['mAP'] = mAP
  for key in rank:
    str_key = rank[key].tolist()
    rank[key]=", ".join(['{:.4f}'.format(x) for x in str_key])
  log['rank'] = rank
  for seq in log['Sequance']:
    for img in log['Sequance'][seq]['Images']:
       log['Sequance'][seq]['Images'][img]= str(log['Sequance'][seq]['Images'][img])
  log['End run time']= time.strftime("%d.%m %H:%M", time.localtime())

  # mange file name
  suffix = '.json'
  if not(filename.endswith(suffix)):
    os.path.join(filename, suffix)

  res_dir = '/Results'
  file_list = os.listdir(res_dir)

  if filename in file_list:
    now = time.strftime("%d_%m_%H_%M", time.localtime())
    filename = os.path.splitext(filename)[0]
    filename = f'{filename}_{now}{suffix}'

  path = os.path.join(res_dir, filename)

  # Convert and write JSON object to file
  with open(path, 'w') as outfile:
      json.dump(log, outfile, indent=4)



