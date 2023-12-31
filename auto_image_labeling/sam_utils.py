import logging
import numpy as np
import torch
import cv2
from segment_anything import sam_model_registry, SamPredictor

from . import utils

logger = logging.getLogger('sam_utils')
logger.setLevel(logging.INFO)


def setup_sam(
        model_type="vit_b",
        sam_checkpoint="models/sam_vit_b_01ec64.pth",
    ):
    logger.info('setup_sam()')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    logger.info('sam mdoel is loaded')
    sam.to(device=device)

    predictor = SamPredictor(sam)
    logger.info('predictor ready')
    return predictor


def run_predictor(predictor, points):
    points_array = np.array(points)
    label_array = np.array([i for i in range(1, len(points)+1)])
    masks, _, _ = predictor.predict(
        point_coords=points_array,
        point_labels=label_array,
        #ask_input=mask_input[None, :, :],
        multimask_output=False,
    )
    print(masks.shape)
    polygons = utils.polygonize(masks[0].astype('uint8') * 255)
    print(polygons)
    return polygons


def show_mask(mask, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    #ax.imshow(mask_image)
    return mask_image

    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))    


