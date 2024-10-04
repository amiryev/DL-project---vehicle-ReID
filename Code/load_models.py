import torch
from transformers import AutoProcessor, AutoModel

from huggingface_hub import hf_hub_download

from GroundingDINO.groundingdino.models import build_model
from GroundingDINO.groundingdino.util.slconfig import SLConfig
from GroundingDINO.groundingdino.util.utils import clean_state_dict

import clip

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

device = "cuda" if torch.cuda.is_available() else "cpu"


'''
This method was copied from
https://huggingface.co/Andyrasika/GroundingDINO
'''
def load_model_hf(repo_id, filename, ckpt_config_filename, device='cpu'):
    cache_config_file = hf_hub_download(repo_id=repo_id, filename=ckpt_config_filename)

    args = SLConfig.fromfile(cache_config_file)
    model = build_model(args)
    args.device = device

    cache_file = hf_hub_download(repo_id=repo_id, filename=filename)
    checkpoint = torch.load(cache_file, map_location='cpu')
    log = model.load_state_dict(clean_state_dict(checkpoint['model']), strict=False)
    print("Model loaded from {} \n => {}".format(cache_file, log))
    _ = model.eval()
    return model


'''
Description:
Load SAM2 model

inputs:
checkpoints - Path to pretrained checkpoint

outputs:
predictor - SAM predictor for segmentation.
'''
def load_sam_model(checkpoint=None):
  if checkpoint is None:
    checkpoint="/content/checkpoints/sam2_hiera_small.pt"
  model_cfg = "sam2_hiera_s.yaml"

  sam2_model = build_sam2(model_cfg, checkpoint, device=device)
  sam2_model.eval()
  predictor = SAM2ImagePredictor(sam2_model)
  return predictor


#@title Load model

'''
Description:
Loads the relevant models.

inputs:
load_clip - If True load CLIP model
load_siglip - If True load SIGLIP model
load_dino - If True load DINO-V2 model
load_sam - If True load SAM2 model

outputs:
gd_model - GroundingDINO model (always loads)
clip_model - CLIP model.
clip_preprocess - Preprocessing for CLIP model
siglip_model - SIGLIP model.
siglip_processor - Preprocessing for SIGLIP model.
dinov2_model - DINO-V2 model.
sam_predictor - SAM2 predictor model.
'''
def load_models(load_clip, load_siglip, load_dino, load_sam):

  #load GroundingDINO (Always on)
  ckpt_repo_id = "ShilongLiu/GroundingDINO"
  ckpt_filenmae = "groundingdino_swinb_cogcoor.pth"
  ckpt_config_filename = "GroundingDINO_SwinB.cfg.py"
  gd_model = load_model_hf(ckpt_repo_id, ckpt_filenmae, ckpt_config_filename, device)
  gd_model.eval()

  if load_clip:
    # load CLIP
    clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
    clip_model.eval()
  else:
    clip_model = None
    clip_preprocess = None

  if load_siglip:
    model_name = "google/siglip-base-patch16-384"
    siglip_model = AutoModel.from_pretrained(model_name)
    siglip_processor = AutoProcessor.from_pretrained(model_name)
    siglip_model.eval()
  else:
    siglip_model = None
    siglip_processor = None

  if load_dino:
    dinov2_model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14").to(device)
    dinov2_model.eval()
  else:
    dinov2_model = None

  if load_sam:
    # load SAM predictor
    sam_predictor = load_sam_model()
  else:
    sam_predictor = None

  return gd_model, clip_model, clip_preprocess, siglip_model, siglip_processor, dinov2_model, sam_predictor