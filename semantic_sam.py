from segment_anything import sam_model_registry, SamPredictor
from segment_anything.modeling.sam import Sam
import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
import os
import gc

from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed


DPI = 100 # dots per inch for matplot

# visualise the segmentations given an image and a mask
def visualise_segmentations(mask:np.ndarray, image:np.ndarray, output_dir:str, image_name:str) -> None:
    height, width = image.shape[:2]

    combined_mask = np.ma.masked_where(mask == 0, mask) # mask out the zero values

    # === Visualise with matplot === This is quite slow (takes majority of the runtime), could definitely be done faster using cv2 (or equivalent) directly
    fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    plt.imshow(image)
    plt.imshow(combined_mask, alpha=0.5, cmap='tab10')
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
    plt.close()


def get_leaf_contours_from_segmentation_mask(segmentation_mask):
    bounding_boxes = []
    centroids = []

    unique_labels = np.unique(segmentation_mask)
    for label_id in unique_labels:
        if label_id == 0:
            continue  # skip background

        binary_mask = (segmentation_mask == label_id).astype(np.uint8) * 255

        # Find contours
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            bounding_boxes.append([x, y, x + w, y + h])
            # cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))
                # cv2.circle(output_image, (cx, cy), 4, (255, 0, 0), -1)

    # Save visualisation
    # outpath = os.path.join(output_dir, f"{stem}_contours_from_mask.png")
    # cv2.imwrite(outpath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    return bounding_boxes, centroids

# def segment_leaves_watershed(segmentation_mask, image, output_dir, stem):
#     output_image = image.copy()
#     centroids = []
#     bboxs = []
#
#     # pre-process the mask
#     segmentation_mask = cv2.GaussianBlur(segmentation_mask, (3, 3), 0)
#     kernel = np.ones((3, 3), np.uint8)
#     segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)
#
#     for plant_id in np.unique(segmentation_mask):
#         if plant_id == 0:
#             continue
#
#         plant_mask = (segmentation_mask == plant_id).astype(np.uint8)
#
#         distance = ndimage.distance_transform_edt(plant_mask)
#         local_maxi = peak_local_max(
#             distance,
#             indices=False,
#             labels=plant_mask,
#             footprint=np.ones((100, 100))  # try larger footprints
#         )
#         markers = ndimage.label(local_maxi)[0]
#         leaf_labels = watershed(-distance, markers, mask=plant_mask)
#
#         # then do contour detect on the labels:
#         bbox, cnt = get_leaf_contours_from_segmentation_mask(leaf_labels)
#         centroids.append(cnt)
#         bboxs.append(bbox)
#         
#         for centroid in cnt:
#             cv2.circle(output_image, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
#
#                 # centroids.append((cx, cy))
#                 # # cv2.circle(output_image, (cx, cy), 4, (255, 0, 0), -1)
#
#         for bbox_ind in bbox:
#             cv2.rectangle(output_image, (bbox_ind[0], bbox_ind[1]), (bbox_ind[2], bbox_ind[3]), (0, 255, 0), 2)
#     
#     outpath = os.path.join(output_dir, f"{stem}_contours_from_mask.png")
#     cv2.imwrite(outpath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

def segment_leaves_watershed(segmentation_mask, image, output_dir, stem):
    output_image = image.copy()
    centroids = []
    bboxs = []

    # Preprocess mask: smooth and clean
    segmentation_mask = cv2.GaussianBlur(segmentation_mask, (3, 3), 0)
    kernel = np.ones((3, 3), np.uint8)
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)

    for plant_id in np.unique(segmentation_mask):
        if plant_id == 0:
            continue

        # Isolate plant area
        plant_mask = (segmentation_mask == plant_id).astype(np.uint8)

        # --- ✨ Mask original image to preserve boundary details ---
        # masked_image = cv2.bitwise_and(image, image, mask=plant_mask)
        masked_image = image.copy()
        masked_image[plant_mask == 0] = 0

        # Convert to grayscale for watershed
        gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)

        # Compute distance map
        distance = ndimage.distance_transform_edt(plant_mask)

        # Detect local maxima as seeds
        local_maxi = peak_local_max(
            distance,
            indices=False,
            labels=plant_mask,
            footprint=np.ones((20, 20))  # tweak as needed
        )
        markers = ndimage.label(local_maxi)[0]

        gradient = cv2.Laplacian(gray, cv2.CV_64F)
        gradient = np.uint8(np.absolute(gradient))

        # Apply watershed using negative distance (basins grow from seeds)
        # leaf_labels = watershed(-distance, markers, mask=plant_mask)
        leaf_labels = watershed(gradient, markers, mask=plant_mask)

        # Find contours from leaf segmentation
        bbox, cnt = get_leaf_contours_from_segmentation_mask(leaf_labels)
        centroids.append(cnt)
        bboxs.append(bbox)

        # Draw centroids
        for centroid in cnt:
            cv2.circle(output_image, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

        # Draw bounding boxes
        for bbox_ind in bbox:
            cv2.rectangle(output_image, (bbox_ind[0], bbox_ind[1]), (bbox_ind[2], bbox_ind[3]), (0, 255, 0), 2)

    # Save result
    outpath = os.path.join(output_dir, f"{stem}_contours_from_mask.png")
    cv2.imwrite(outpath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))


# creates an array of all of the bounding boxes in format [[x_min, y_min, x_max, y_max]]
# formatted for rcnn bounding box .txt output
def get_sam_bounding_boxes(bbox_path:Path, im_height:int) -> np.ndarray:
    bbox_data = np.loadtxt(bbox_path)
    if len(bbox_data.shape) == 1:
        bbox_data = np.array([bbox_data])
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
    model_type = "vit_b"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
        print("Cuda GPU Found")

    sam.to(device)
    predictor = SamPredictor(sam)

    # === Loop through and Segment Images ==
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

        # TEMP
        # print("before:", np.unique(segmentation_mask))
        # # segmentation_mask = segmentation_mask == 3
        # segmentation_mask = np.where(segmentation_mask == 3, 3, 0)
        # print("after:", np.unique(segmentation_mask))

        # visualise_segmentations(segmentation_mask, image, "./output/semantics", "single_plant")

        if visualise:
            # === visualise the segmentations and save them as a png ===
            visualise_segmentations(segmentation_mask, image, output_dir, stem)

        # leaf_seg = segment_leaves_watershed(segmentation_mask)
        # visualise_segmentations(leaf_seg, image, output_dir, stem)

        # bboxes, centroids = get_leaf_contours_from_segmentation_mask(segmentation_mask, image, output_dir, stem)
        # segment_leaves_watershed(segmentation_mask, image, output_dir, stem)



    # === Clean up ===
    del predictor
    del sam
    torch.cuda.empty_cache()
    gc.collect()



