#@title libraries imports for all cells
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import torch
import torchvision
import torchvision.transforms as T
from torchvision.transforms.functional import crop
from torchvision.transforms import InterpolationMode

from GroundingDINO.groundingdino.util import box_ops
from GroundingDINO.groundingdino.util.inference import load_image, predict

import utils

device = "cuda" if torch.cuda.is_available() else "cpu"


'''
Description:
Gets an image and GroundingDINO model, returns the bounding boxes detected by GD.

inputs:
img - Input image, must be string, tensor or numpy array.
gd_model - GroundingDINO model.

outputs:
bboxes - Bounding box for each detected vehicle in image. Format is [left, top, width, height]
boxes_score - Confidence in detection
phrases - List of string with token matching each detection
'''
def detect_objects(base_img, gd_model, box_thr=0.5, query='vehicles'):
  #Validate inputs
  if type(base_img) is str:
    image_source, img_tensor = load_image(base_img)
  elif type(base_img)==torch.tensor or type(base_img)==np.ndarray:
    img_tensor = torch.tensor(base_img)
  else:
    raise AttributeError('img must be string, tensor or numpy array')

  # apply GD on image, default query is 'vehicles'
  boxes, boxes_score, phrases = predict(
      model=gd_model,
      image=img_tensor,
      caption=query,
      box_threshold=box_thr,
      text_threshold=0.9,
      device=device
  )

  # convert normalized boxes to pixel values
  H, W, _ = image_source.shape
  bboxes = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])

  return bboxes, boxes_score, phrases


'''
Description:
Gets a binary image (mask) and reutrn the smallest rectangle containing pixels valued 1.

inputs:
mask - Binary image, must by numpy array.

outputs:
left - Left most boundry of box
right - Right most boundry of box
top - Top boundry of box
bottom - Bottom boundry of box
'''
def get_bbox_from_binary_image(mask):
  ind = np.argwhere(mask)
  if ind.shape.count(0) != 0:
    top, bottom, left, right = -1, -1, -1, -1
  else:
    top = ind[:, 0].min()
    bottom = ind[:, 0].max()
    left = ind[:, 1].min()
    right = ind[:, 1].max()
  return left, right, top, bottom


'''
Description:
Takes an image and a mask (binary image) of the same size, It returns the masked
image in PIL format and crops the image around the object.

inputs:
img - Input image.
mask - Binary image of same size as img.

outputs:
sam_img - PIL image with all pixels 0 except for the object
'''
def apply_mask(img, mask, thr=0.1):
  np_img = np.array(img)
  mask = np.array(mask)
  if mask.shape != np_img.shape[:2]:
    raise ValueError('img and mask must be the same size')

  # apply mask to all 3 channels
  masked_img = np_img * np.repeat(mask[:, :, np.newaxis], 3, axis=2)

  # Find minimal bounding box over object and crop image
  sam_img = masked_img

  if sam_img.shape[1] < 2 or sam_img.shape[2] < 2:
    sam_img = np_img

  H, W = mask.shape
  if mask.sum() < (thr*H*W):
    sam_img = np_img

  return Image.fromarray(sam_img.astype('uint8'), 'RGB')


'''
Description:
Method to convert an image to a feature vector. First, it is optional to mask
the object using SAM predictor, then crop to give a minimal image containing the object.
Second use CLIP to extract the features from the masked image.

inputs:
image - Input image. Should be PIL format but not a must.
clip_model - Pretrained CLIP model.
preprocess - Image preprocess transforms for CLIP.
mask - Optional. Mask input images with same mask.

outputs:
img_features - Essential features extracted from image.
'''
def clip_image_to_features(image, clip_model, preprocess=None, mask=None):
  if mask:
    masked_img = apply_mask(image, mask)
  else:
    masked_img = image

  # Encode image to get features and normalize them
  if preprocess is not None:
    processed_img = preprocess(masked_img).to(device)
  else:
    BICUBIC = InterpolationMode.BICUBIC
    transform = T.Compose([T.Resize((224, 224), interpolation=BICUBIC), T.ToTensor(),
      T.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    processed_img = transform(masked_img).to(device)

  with torch.no_grad():
    img_features = clip_model.encode_image(processed_img.unsqueeze(0))

  img_features = img_features / img_features.norm(dim=1, keepdim=True)
  return img_features.cpu().numpy()


'''
Description:
Method to convert an image to a feature vector. First, it is optional to mask
the object using SAM predictor, then crop to give a minimal image containing the object.
Second use CLIP to extract the features from the masked image.

inputs:
img - Input images. Should be PIL format but not a must.
siglip_model - Pretrained SigLIP model.
mask - Optional. Mask input images with same mask.

outputs:
img_features - Essential features extracted from image.
'''
def siglip_image_to_features(images, siglip_model, siglip_processor=None,  mask=None):
  if mask:
    masked_imgs = apply_mask(images, mask)
  else:
    masked_imgs = images

  # Encode image to get features and normalize them
  inputs = siglip_processor(images=masked_imgs, return_tensors="pt")
  with torch.no_grad():
    img_features = siglip_model.get_image_features(**inputs)

  img_features = img_features / img_features.norm(dim=1, keepdim=True)

  return img_features.cpu().numpy()


'''
Description:
Method to convert an image to a feature vector. First, it is optional to mask
the object using SAM predictor, then crop to give a minimal image containing the object.
Second use Dino_V2 to extract the features from the masked image.

inputs:
image - Input images. Should be PIL format but not a must.
dinov2_model - Pretrained Dino_v2 model.
mask - Optional. Mask input images with same mask.

outputs:
img_features - Essential features extracted from image.
'''
def dino2_image_to_features(image, dino_model , mask=None):
  if mask:
    #Apply mask on image
    masked_img = apply_mask(image, mask)
  else:
    masked_img = image

  transform = T.Compose([T.ToTensor(), T.Resize(244), T.CenterCrop(224), T.Normalize([0.5], [0.5])])
  transformed_imgs = np.array([transform(img)[:3] for img in masked_img])

  img_features = dino_model(torch.tensor(transformed_imgs).to(device))

  img_features = img_features / img_features.norm(dim=1, keepdim=True)
  return img_features.cpu().detach().numpy()


#@title Object Reidentification

'''
Description:
Takes an image and a SAM predictor. Its generates 3 SAM masks, and
returns the last one which empirically is the object.

inputs:
img - Input image.
predictor - SAM predictor for segmentaion.

outputs:
mask - Mask (binary image) of the segmented object.
score - Confidence in segmentation.
'''
def sam_mask_predictor(img, predictor):
  img = np.array(img)

  # Load image to SAM predictor
  predictor.set_image(img)

  # Set mid-point of image as point for object segmentation (computes seg around
  # this poing), and the label is 1 for foreground
  H, W, _ = img.shape
  input_point = np.array([[W/2, H/2]])
  input_label = np.array([1])

  # predict 3 segmentation masks
  with torch.no_grad():
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

  return masks[-1], scores[-1]


'''
Description:
Predict SAM2 segmentation mask and apply on each image.

inputs:
images - Input images (query and targets). a List tensors.
predictor - SAM predictor for segmentaion.

outputs:
masked_images - Masked cropped images.
'''
def apply_sam_mask(images, predictor):
  masked_images = []
  with torch.no_grad():
    for img in images:
      #generate SAM mask
      mask, _ = sam_mask_predictor(np.array(img), predictor)

      #Apply mask on image
      masked_images.append(apply_mask(img, mask))

    return masked_images

'''
Description:
compare simililarity of query image fetures and each target features. Use cosine
similarity metric.

inputs:
image_features - Numpy array containing features extracted from query image and targets.
images - Input images (query and targets), a List tensors. Used only to create
output list with score-image pair.


outputs:
scores - List of tuples pairs containing each target image and its similarity to
query score.
'''
def compute_features_similarity(image_features, images, bboxes, sim_type='cosine'):
  if sim_type == 'cosine':
    similarity = torch.nn.functional.cosine_similarity
    rev = True
  elif sim_type == 'norm':
    similarity = torch.norm
    rev = False
  image_features = torch.tensor(image_features)
  sim = [similarity(image_features[0], z)[0] for z in image_features[1:]]
  scores = zip(sim, images[1:], bboxes)

  scores = sorted(scores, key=lambda x: x[0], reverse=rev)

  return scores

'''
Description:
Crop targets to separate images from base image. Arrange query image and all targets
in a list (query first, then targets by GroundingDINO score)

inputs:
query_image - Input query image (object to find in base image). Should be string or PIL format
base_image - Input base image (extract targets from this image). Should be string or PIL format
bboxes - Bounding boxes to mark targets

outputs:
images - List containig query image and cropped targets images
'''
def get_image_list(query_image, base_image, bboxes):
  if type(query_image) is str:
    query_image = Image.open(query_image)
  if type(base_image) is str:
    base_image = Image.open(base_image)

  # extract features from all images
  images = [query_image]
  for bbox in bboxes.numpy():
    # cropped_img = crop(base_image, bbox[1], bbox[0], bbox[3], bbox[2])
    cropped_img = crop(base_image, bbox[1], bbox[0], bbox[3]-bbox[1], bbox[2]-bbox[0])
    images.append(cropped_img)
  return images



'''
Description:
Runs the entire pipeline on a single base image with specific query

inputs:
base_img_path - Path to base image (where we want to find the query object)
query_image - Query image. Holding one object.
box_thresh - Thershold for GroundingDINO boxes detection.
query_prompt - Single word to use as prompt for GroundingDINO, Default is 'vehicles'.
display_on - If True show ReID results on screen by similarity score. Default is False.

outputs:
scores - Data structure that holds results of ReID. A dictionary of the feature extraction models used. Each value in the dict
is a lsit of tuples, where each member of the list corresponds to a single target in the base image (cropped object). Each tuple
is (<similarity_score>, <cropped_image>, <bbox>).
'''
def process_single_image(models, base_img_path, query_image, box_thresh, query_prompt = 'vehicles',  display_on=False):
    # unzip models
    gd_model, clip_model, clip_preprocess, siglip_model, siglip_processor, dino_model, sam_predictor = models
  
    # detect targets on base image using GroundingDINO
    bboxes, _, _ = detect_objects(base_img_path, gd_model, box_thresh, query=query_prompt)

    # load base image to find the query in
    base_image = Image.open(base_img_path)

    # Make a list containing query image and cropped targets
    images = get_image_list(query_image, base_image, bboxes)

    # apply segmentation masks using sam2 if turned on
    if sam_predictor is None:
        masked_images = images
    else:
        masked_images = apply_sam_mask(images, sam_predictor)

    # Extract features from all images in list by model. CLIP works on a single image at a time, SIGLIP and DINO-V2 works aon bathces.
    scores = {}
    if clip_model is not None:
        clip_features = np.array([clip_image_to_features(img, clip_model) for img in masked_images])
        scores['clip'] = compute_features_similarity(clip_features, images, bboxes)
    if siglip_model is not None:
        siglip_features = siglip_image_to_features(masked_images, siglip_model)
        siglip_features = torch.tensor(siglip_features).unsqueeze(1)
        scores['siglip'] = compute_features_similarity(siglip_features, images, bboxes)
    if dino_model is not None:
        dino_features = dino2_image_to_features(masked_images, dino_model)
        dino_features = torch.tensor(dino_features).unsqueeze(1)
        scores['dino'] = compute_features_similarity(dino_features, images, bboxes)


    if display_on:
        utils.display_reid_results(query_image, scores, num_results=5)

    return scores