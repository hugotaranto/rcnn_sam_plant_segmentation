from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.sam import Sam
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
import gc

DPI = 100 # dots per inch for matplot

# creates an array of all of the bounding boxes in format [[x_min, y_min, x_max, y_max]]
# formatted for rcnn bounding box .txt output
def get_sam_bounding_boxes(bbox_path:Path, im_height:int) -> np.ndarray:
    bbox_data = np.loadtxt(bbox_path)
    bbox_data[:, 1:5] *= im_height # bounding box percentage poses were scaled only using the image height

    x_mins = bbox_data[:, 1] - bbox_data[:, 3] / 2
    y_mins = bbox_data[:, 2] - bbox_data[:, 4] / 2
    x_maxs = bbox_data[:, 1] + bbox_data[:, 3] / 2
    y_maxs = bbox_data[:, 2] + bbox_data[:, 4] / 2

    corners = np.stack([x_mins, y_mins, x_maxs, y_maxs], axis=1)
    return corners

# load image into numpy array
def load_image(image_path:Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could Not Read Image")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

# visualise the segmentations given an image and a mask
def viualise_segmentations(mask:np.ndarray, image:np.ndarray, output_dir:str, image_name:str) -> None:
    width, height = image.shape[:2]

    combined_mask = np.ma.masked_where(mask == 0, mask) # mask out the zero values

    # === Visualise with matplot === This is quite slow (takes majority of the runtime), could definitely be done faster using cv2 (or equivalent) directly
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    plt.imshow(image)
    plt.imshow(combined_mask, alpha=0.5, cmap='tab10')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
    plt.close()

# segment all the plants in a given image return mask of segments
def segment_plants(image:np.ndarray, bbox_path:Path, predictor:SamPredictor) -> np.ndarray:

    height, width = image.shape[:2]

    # === Load the bboxes ===
    bboxes = get_sam_bounding_boxes(bbox_path, height)

    # === Set Predictor ===
    predictor.set_image(image)      # Load the image into the predictor

    # === Segment each box individually using SAM ===
    masks = []
    for box in bboxes:
        mask, scores, _ = predictor.predict(
            box=box[None, :],
            multimask_output=False
        )
        masks.append(mask[0])

    # === Remove the predictor/embeddings we are done with them ===
    predictor.reset_image()
    torch.cuda.empty_cache()

    # === combine the masks ===
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for i, mask in enumerate(masks, 1):
        combined_mask[mask] = i

    return combined_mask # combined mask of all individual plant segmentations

# segment images given bounding boxes
def segment_images(data_path:str, bbox_path:str, output_dir:str, sam_checkpoint:str, visualise:bool=True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # === Load Sam Model ===
    sam_checkpoint = './sam_base_checkpoint.pth'
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda GPU Found")

    sam.to(device)
    predictor = SamPredictor(sam)

    # === Loop through and Segment Images ===
    # build a dictionary of image/bbox files by stem
    bbox_files = {f.stem: f for f in Path(bbox_path).glob('*.txt')}
    img_files = {f.stem: f for f in Path(data_path).glob('*.png')}

    # get the common stems
    common_stems = bbox_files.keys() & img_files.keys()

    # loop through the common files and segment them
    for stem in sorted(common_stems):
        bbox_dir = bbox_files[stem]
        img_path = img_files[stem]

        # === Load the image ===
        image = load_image(img_path)

        # === segment the image ===
        segmentation_mask = segment_plants(image, bbox_dir, predictor)

        if visualise:
            # === visualise the segmentations and save them as a png ===
            viualise_segmentations(segmentation_mask, image, output_dir, stem)

    # === Clean up ===
    del predictor
    del sam
    torch.cuda.empty_cache()
    gc.collect()

