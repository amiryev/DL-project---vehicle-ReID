import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
import matplotlib.patches as patches
from PIL import Image

import time

import torch
from torchvision.transforms.functional import crop

import load_models
import reid


device = "cuda" if torch.cuda.is_available() else "cpu"


def show_boxed_image(img, boxes=[], box_color=[], labels= [], size=(9,7)):
  fig, ax = plt.subplots(1, figsize=size)
  ax.imshow(img)
  rect=[]
  txt_offset=0
  for label, box in zip(labels, boxes):
    temp_rect=patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=1,
                        facecolor="none")
    for loop_rect in rect:
      if temp_rect.get_xy() == loop_rect.get_xy():
        txt_offset +=1
    rect.append(temp_rect)
    ax.text(box[0] ,box[1]-txt_offset * 35, label ,color = "white")

  ax.add_collection(PatchCollection((rect), facecolor='none', edgecolor = box_color, linewidth=2))
  ax.axis('off')
  plt.show()

def find_target(models, query_path, target_path, dino_box_thresh):
  # unzip models
  gd_model = models[0]
  
  # load images
  query_image = Image.open(query_path)
  target_image = Image.open(target_path)

  # entering prompt, get dino result, and choose object to detect
  choise=0
  while choise ==0:
    show_boxed_image(query_image)
    time.sleep(.5)
    prompt = input("Enter your target Prompt: \n")
    boxes, _, _ = reid.detect_objects(query_path, gd_model, box_thr=0.3, query=prompt)
    num_boxes=len(boxes)
    show_boxed_image(query_image, boxes, ['m']* num_boxes, range(1,num_boxes+1))
    time.sleep(.5)
    choise = int(input("Choose target number (0 to change prompt if there is no box on target): \n"))
    if choise<0 or choise>len(boxes):
      print("wrong input, try again")
      choise=0

  # crop image
  query_bbox = boxes[choise-1]
  croped_query_image = crop(query_image, query_bbox[1].item(), query_bbox[0].item(), query_bbox[3].item()-query_bbox[1].item(), query_bbox[2].item()-query_bbox[0].item())

  # look for target
  scores = reid.process_single_image(models, target_path, croped_query_image, dino_box_thresh, query_prompt=prompt)

  #show target boxed image
  trgt_labels=[]
  trgt_box=[]
  for key in scores:
          trgt_labels.append(key+ ": "+ '{:.3f}'.format(scores[key][0][0].item()))
          trgt_box.append(scores[key][0][2].numpy().astype(int))

  show_boxed_image(target_image, trgt_box, ['g']*len(trgt_box), trgt_labels)


def main():
    # insert your query and base images paths here
    query_file = "query_image_example.jpg"
    target_file = "base_image_example.jpg"

    # choose configurations
    use_clip = True
    use_siglip = False
    use_dinov2 = True
    use_sam = False
    dino_box_thresh = 0.3 #threshold for GroudningDINO detection
    
    # Load wanted models
    models = load_models.load_models(load_clip= use_clip,
                                load_siglip= use_siglip,
                                load_dino= use_dinov2,
                                load_sam= use_sam)

    find_target(models, query_file, target_file, dino_box_thresh)


if __name__ == '__main__':
  main()