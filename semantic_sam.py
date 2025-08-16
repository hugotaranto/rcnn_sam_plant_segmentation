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
from scipy.ndimage.measurements import label

from plantcv import plantcv as pcv
from typing import Tuple

from sklearn.cluster import DBSCAN
from skimage import color
import math

import random

DPI = 100 # dots per inch for matplot

def visualise_segmentation_random(mask:np.ndarray, image:np.ndarray, output_dir:str, image_name:str) -> None:
    height, width = image.shape[:2]
    num_labels = mask.max()

    # Seed for reproducibility
    random.seed(0)

    # Generate random colors for each label (label 0 is background = transparent)
    colors = [(0, 0, 0, 0)]  # RGBA for background (transparent)
    for _ in range(num_labels):
        # Generate random RGB with alpha=255
        colors.append((random.randint(0,255), random.randint(0,255), random.randint(0,255), 150))  # semi-transparent

    # Create an RGBA overlay image to color labels
    overlay = np.zeros((height, width, 4), dtype=np.uint8)

    for label in range(1, num_labels + 1):
        overlay[mask == label] = colors[label]

    # Convert original image to RGBA for blending
    if image.shape[2] == 3:  # if RGB
        image_rgba = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    else:
        image_rgba = image.copy()

    # Blend the overlay with the original image
    alpha_overlay = overlay[..., 3:] / 255.0
    alpha_image = 1.0 - alpha_overlay

    for c in range(3):  # for R, G, B channels
        image_rgba[..., c] = (alpha_image[..., 0] * image_rgba[..., c] + alpha_overlay[..., 0] * overlay[..., c]).astype(np.uint8)

    # Save with matplotlib
    plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
    plt.imshow(image_rgba)
    plt.axis('off')
    plt.tight_layout(pad=0)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, image_name), bbox_inches='tight', pad_inches=0)
    plt.close()

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


# def get_leaf_contours_from_segmentation_mask(segmentation_mask, output_dir, stem, image):
#     bounding_boxes = []
#     centroids = []
#     output_image = image.copy()
#
#     unique_labels = np.unique(segmentation_mask)
#     for label_id in unique_labels:
#         if label_id == 0:
#             continue  # skip background
#
#         binary_mask = (segmentation_mask == label_id).astype(np.uint8) * 255
#
#         # Find contours
#         contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#         for cnt in contours:
#             area = cv2.contourArea(cnt)
#             if area < 200:
#                 continue
#
#             x, y, w, h = cv2.boundingRect(cnt)
#             bounding_boxes.append([x, y, x + w, y + h])
#             cv2.rectangle(output_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#             M = cv2.moments(cnt)
#             if M["m00"] != 0:
#                 cx = int(M["m10"] / M["m00"])
#                 cy = int(M["m01"] / M["m00"])
#                 centroids.append((cx, cy))
#                 cv2.circle(output_image, (cx, cy), 4, (255, 0, 0), -1)
#
#     # Save visualisation
#     outpath = os.path.join(output_dir, f"{stem}_contours_from_mask.png")
#     cv2.imwrite(outpath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))
#
#     return bounding_boxes, centroids

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

            M = cv2.moments(cnt)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append((cx, cy))

    return bounding_boxes, centroids


def segment_leaves_watershed(segmentation_mask, image, output_dir, stem):

    # create a binary mask where the background is removed
    binary_mask = (segmentation_mask > 0).astype(np.uint8)

    # get the distance map
    distance = ndimage.distance_transform_edt(binary_mask)

    # get the local maxi seeds
    local_maxi = peak_local_max(
        distance,
        indices=False,
        labels=segmentation_mask,
        footprint=np.ones((50, 50))
    )

    # display the markers
    output_image = image.copy()

    y_coords, x_coords = np.nonzero(local_maxi)

    for (x, y) in zip(x_coords, y_coords):
        # draw a circle on the output image at each marker location
        cv2.circle(output_image, (x, y), 5, (255, 0, 0), thickness=-1)
        
    outpath = os.path.join(output_dir, f"{stem}_seed_points.png")
    cv2.imwrite(outpath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    markers = ndimage.label(local_maxi)[0]
    watershed_mask = watershed(-distance, markers, mask=binary_mask)

    # do contour detect on the labels:
    bboxes, centroids = get_leaf_contours_from_segmentation_mask(watershed_mask)

    # visualise the contours and bounding boxes
    output_image = image.copy()

    for centroid in centroids:
        cv2.circle(output_image, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)

    for bbox in bboxes:
        cv2.rectangle(output_image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
    
    outpath = os.path.join(output_dir, f"{stem}_contours_from_mask.png")
    cv2.imwrite(outpath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

    bboxes = np.array(bboxes)

    return watershed_mask, bboxes

# def segment_leaves_watershed(segmentation_mask, image, output_dir, stem):
#     output_image = image.copy()
#     centroids = []
#     bboxs = []
#
#     # Preprocess mask: smooth and clean
#     # segmentation_mask = cv2.GaussianBlur(segmentation_mask, (3, 3), 0)
#     # kernel = np.ones((3, 3), np.uint8)
#     # segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, kernel)
#
#     for plant_id in np.unique(segmentation_mask):
#         if plant_id <= 1:
#             continue
#
#         # Isolate plant area
#         plant_mask = (segmentation_mask == plant_id).astype(np.uint8)
#
#         plt.subplot(1, 3, 1)
#         plt.imshow(plant_mask, cmap='gray')
#
#         # masked_image = cv2.bitwise_and(image, image, mask=plant_mask)
#         masked_image = image.copy()
#         masked_image[plant_mask == 0] = 0
#
#         # Convert to grayscale for watershed
#         gray = cv2.cvtColor(masked_image, cv2.COLOR_RGB2GRAY)
#
#         # Compute distance map
#         distance = ndimage.distance_transform_edt(plant_mask)
#
#         plt.subplot(1, 3, 2)
#         plt.imshow(distance, cmap='gray')
#
#         # Detect local maxima as seeds
#         # local_maxi = peak_local_max(
#         #     distance,
#         #     indices=True,
#         #     labels=plant_mask,
#         #     footprint=np.ones((20, 20))  # tweak as needed
#         # )
#
#         # print(local_maxi)
#         # labels = label(local_maxi)[0]
#         # print(labels)
#
#
#         # Detect local maxima as seeds
#         local_maxi = peak_local_max(
#             distance,
#             indices=False,
#             labels=plant_mask,
#             # footprint=np.ones((20, 20))  # tweak as needed
#             min_distance=20
#         )
#
#         # so then local_maxi are the points of the leaf detection
#
#         markers = ndimage.label(local_maxi)[0]
#         #
#         # print(markers.shape)
#         #
#         # # Make a copy of the original image
#         # overlay = image.copy()
#         #
#         # # Create random colors for each marker
#         # num_markers = markers.max()
#         # print(num_markers)
#         # colors = np.random.randint(0, 255, (num_markers + 1, 3), dtype=np.uint8)
#         #
#         # # Create a color overlay for markers
#         # marker_overlay = np.zeros_like(image, dtype=np.uint8)
#         #
#         # for label in range(1, num_markers + 1):
#         #     marker_mask = (markers == label)
#         #     marker_overlay[marker_mask] = colors[label]
#         #
#         # # Blend overlay with the original image
#         # alpha = 0.5  # transparency factor
#         # cv2.addWeighted(marker_overlay, alpha, overlay, 1 - alpha, 0, overlay)
#         #
#         # # Display the result
#         # # cv2.imshow("Markers Overlay", overlay)
#         # # cv2.waitKey(0)
#         # # cv2.destroyAllWindows()
#         #
#         # plt.subplot(1, 3, 3)
#         # plt.imshow(overlay)
#         # plt.show()
#
#         gradient = cv2.Laplacian(gray, cv2.CV_64F)
#         gradient = np.uint8(np.absolute(gradient))
#
#         # Apply watershed using negative distance (basins grow from seeds)
#         # leaf_labels = watershed(-distance, markers, mask=plant_mask)
#         leaf_labels = watershed(gradient, markers, mask=plant_mask)
#
#         # Find contours from leaf segmentation
#         bbox, cnt = get_leaf_contours_from_segmentation_mask(leaf_labels)
#         centroids.append(cnt)
#         bboxs.append(bbox)
#
#         # Draw centroids
#         for centroid in cnt:
#             cv2.circle(output_image, (centroid[0], centroid[1]), 4, (255, 0, 0), -1)
#
#         # Draw bounding boxes
#         for bbox_ind in bbox:
#             cv2.rectangle(output_image, (bbox_ind[0], bbox_ind[1]), (bbox_ind[2], bbox_ind[3]), (0, 255, 0), 2)
#
#     # Save result
#     outpath = os.path.join(output_dir, f"{stem}_contours_from_mask.png")
#     cv2.imwrite(outpath, cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

def watershed_segment_leaves(plant_mask, image):
    masked_image = image.copy()

    # apply the mask to the image
    binary_mask = (plant_mask == 5).astype(np.uint8)

    # convert to 3 channels
    binary_mask_3 = cv2.merge([binary_mask] * 3)

    masked_image = masked_image * binary_mask_3

    # watershed the image
    processed = pcv.watershed_segmentation(rgb_img=masked_image, mask=binary_mask, distance=5, label='default')

    print(np.unique(processed))

    plt.subplot(1, 4, 1)
    plt.imshow(image)

    plt.subplot(1, 4, 2)
    plt.imshow(masked_image)

    plt.subplot(1, 4, 3)
    plt.imshow(processed, cmap='nipy_spectral')

    plt.subplot(1, 4, 4)
    plt.imshow(binary_mask, cmap='gray')


    plt.tight_layout()

    plt.show()


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

def crop_images(image:np.ndarray, output_dir, stem, plant_segmentation:np.ndarray):

    cropped_plants = []
    plants = np.unique(plant_segmentation)

    os.makedirs(os.path.join(output_dir, "Crops", stem), exist_ok=True)

    for plant in plants:

        if plant == 0:
            continue

        # mask the image to only include that plant
        masked_plant = image.copy()
        binary_mask = (plant_segmentation == plant).astype(np.uint8)
        binary_mask_3 = cv2.merge([binary_mask] * 3)
        masked_plant = masked_plant * binary_mask_3

        # Find bounding box
        coords = cv2.findNonZero(binary_mask)
        if coords is None:
            continue  # skip if no pixels found

        x, y, w, h = cv2.boundingRect(coords)

        # Crop the original image using bounding box
        crop = masked_plant[y:y+h, x:x+w]
        cropped_plants.append(crop)

        # save the image
        cv2.imwrite(os.path.join(output_dir, "Crops", stem, f"plant_{plant}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))

    # images = []
    #
    # # round the bbox data
    # bboxes = np.round(bboxes).astype(int)
    #
    # # make the directory for the crops
    # os.makedirs(os.path.join(output_dir, "Crops", stem), exist_ok=True)
    #
    # for i, (x_min, y_min, x_max, y_max) in enumerate(bboxes):
    #
    #     print(i)
    #     print(x_min, y_min, x_max, y_max)
    #
    #     crop = image[y_min:y_max, x_min:x_max]
    #     images.append(crop)
    #
    #     # save the image
    #     cv2.imwrite(os.path.join(output_dir, "Crops", stem, f"plant_{i}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))


def get_leaf_points(image, plant_segmentation_mask, bboxes):

    # array containing each plants leaf detection prediction points
    leaf_points = []

    # go through each plant
    for plant_id in np.unique(plant_segmentation_mask):
        # 0 is background (not plant)
        if plant_id == 0:
            continue

        # Isolate the current plant
        plant_mask = (plant_segmentation_mask == plant_id).astype(np.uint8)
        
        # mask the image such that everything is 0 except the plant
        masked_image = image.copy()
        masked_image[plant_mask == 0] = 0

        # get the distance map
        distance_map = ndimage.distance_transform_edt(plant_mask)

        # find the local maxi (leaves detection prediction)
        local_maxi = peak_local_max(
            distance_map,
            indices=True,
            labels=plant_mask,
            min_distance=20
        )

        local_maxi = local_maxi[:, [1, 0]] # flip from y, x to x, y (what SAM wants)
        local_maxi_pruned = []

        # maybe remove any points that are too close to the middle? if there are more than 2 points???
        if len(local_maxi) >= 3:
            # get the center of the bbox of this plant:
            center_x = (bboxes[plant_id - 1][2] - bboxes[plant_id - 1][0]) / 2 + bboxes[plant_id - 1][0]
            center_y = (bboxes[plant_id - 1][3] - bboxes[plant_id - 1][1]) / 2 + bboxes[plant_id - 1][1]

            # then remove the points that are too close to this center point
            center_distance_thr = 30
            for maxi in local_maxi:
                distance_from_center = math.sqrt((center_x - maxi[0])**2 + (center_y - maxi[1])**2)
                if distance_from_center > center_distance_thr:
                    local_maxi_pruned.append(maxi)

            local_maxi_pruned = np.array(local_maxi_pruned)

            leaf_points.append(local_maxi_pruned)
        else:
            leaf_points.append(local_maxi)
        # leaf_points.append(local_maxi)

        # markers = label(local_maxi)[0]
        # labels = watershed(-distance_map, markers, mask=plant_mask)
        #
        # # then we have the labels from the watershed
        # label_rgb = color.label2rgb(labels, image=image, bg_label=0, alpha=0.5)
        # plt.imshow(label_rgb)
        # plt.axis("off")
        # plt.tight_layout()
        # plt.show()
        #
        # # cluster the points
        # eps = 30 # max distance between points to be considered in same cluster
        # min_samples = 1
        #
        # clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(local_maxi)
        # clustered_points = []
        # for cluster_id in np.unique(clustering.labels_):
        #     cluster_points = local_maxi[clustering.labels_ == cluster_id]
        #     # take the mean point of the cluster
        #     centroid = cluster_points.mean(axis=0).astype(int)
        #     clustered_points.append(centroid)
        #
        # clustered_points = np.array(clustered_points)
        # clustered_points = clustered_points[:, [1, 0]] # swap from y, x to x, y format
        # leaf_points.append(clustered_points)

    # now make the points for the test

    return np.array(leaf_points)

def segment_plants(image:np.ndarray, bboxes:np.ndarray, predictor:SamPredictor) -> np.ndarray:

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

    # === combine the masks ===
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for i, mask in enumerate(masks, 1):
        combined_mask[mask] = i

    # === Remove the predictor/embeddings we are done with them ===
    predictor.reset_image()
    torch.cuda.empty_cache()

    return combined_mask

# segment all the plants in a given image return mask of segments
def segment_plants_leaves(image:np.ndarray, bboxes:np.ndarray, predictor:SamPredictor) -> Tuple[np.ndarray, np.ndarray]:

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


    # === combine the masks ===
    combined_mask = np.zeros_like(masks[0], dtype=np.uint8)
    for i, mask in enumerate(masks, 1):
        combined_mask[mask] = i

    # create a new image now with everything except the plants masked out
    binary_mask = (combined_mask > 0).astype(np.uint8)
    binary_mask_3 = np.stack([binary_mask]*3, axis=-1)
    masked_image = image * binary_mask_3

    # set this as the new image in the predictor
    predictor.set_image(masked_image)

    # === segment the leaves ===
    leaf_points = get_leaf_points(image, combined_mask, bboxes)

    # TEMP

    # display the points on the image
    plt.imshow(masked_image)
    plt.axis("off")
    plt.tight_layout(pad=0)
    for plant in leaf_points:
        plt.scatter(plant[:,0], plant[:,1], c='red', s=5)
    plt.savefig(f'./output/semantics/points/points.png')

    # TEMPEND

    # leaf_masks = []
    leaf_masks = np.zeros_like(combined_mask)
    leaf_areas = {} # dictionary with label to area
    leaf_label = 1
    # feed these leaf points into SAM
    for plant_id, plant in enumerate(leaf_points):
        # if plant_id != 1:
        #     continue
        num_leaves = len(plant)

        for i in range(num_leaves):
            labels = np.zeros(num_leaves)
            labels[i] = 1

            leaf_mask, scores, logits = predictor.predict(
                point_coords=plant,
                point_labels=labels,
                # multimask_output=False,
                multimask_output=True
            )

            # select the segmentation with the lowest area
            areas = [np.sum(m) for m in leaf_mask]
            min_area_idx = np.argmin(areas)
            high_score_mask = leaf_mask[min_area_idx]

            current_area = areas[min_area_idx]
            # find the pixels where this leaf applies
            y_list, x_list = np.where(high_score_mask)

            # only replace label in mask if it is 0 (background) or a smaller area than the current one
            for y, x in zip(y_list, x_list):
                existing_label = leaf_masks[y, x]
                if existing_label == 0:
                    leaf_masks[y, x] = leaf_label

                elif leaf_areas[existing_label] > current_area:
                    leaf_masks[y, x] = leaf_label
                    leaf_areas[existing_label] -= 1
                else:
                    # the pixel could not be assigned
                    current_area -= 1

            # leaf_masks.append(high_score_mask)
            # combine the current leaf into the mask
            # leaf_masks[high_score_mask] = leaf_label

            leaf_areas[leaf_label] = current_area
            leaf_label += 1

            # height, width = image.shape[:2]
            #
            # temp_leaf_mask = np.ma.masked_where(high_score_mask == 0, high_score_mask) # mask out the zero values
            #
            # # === Visualise with matplot === This is quite slow (takes majority of the runtime), could definitely be done faster using cv2 (or equivalent) directly
            # fig = plt.figure(figsize=(width / DPI, height / DPI), dpi=DPI)
            # plt.imshow(image)
            # plt.imshow(temp_leaf_mask, alpha=0.5, cmap='tab10')
            # plt.axis("off")
            # plt.tight_layout(pad=0)
            # plt.savefig(f"./output/semantics/leaves/plant_{plant_id},leaf_{i}.png", bbox_inches='tight', pad_inches=0)
            # plt.close()

    # === combine the leaf masks ===
    # combined_leaf_mask = np.zeros_like(leaf_masks[0], dtype=np.uint8)
    # for i, mask in enumerate(leaf_masks, 1):
    #     combined_leaf_mask[mask] = i

    # === Remove the predictor/embeddings we are done with them ===
    predictor.reset_image()
    torch.cuda.empty_cache()

    return combined_mask, leaf_masks # combined mask of all individual plant segmentations

# segment images given bounding boxes
def segment_images(data_path:str, bbox_path:str, output_dir:str, sam_checkpoint:str, visualise:bool=True) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    # === Load Sam Model ===
    model_type = "vit_h"
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

        print(stem)

        # === Load the image ===
        image = load_image(img_path)

        # === Load the bboxes ===
        bboxes = get_sam_bounding_boxes(bbox_dir, image.shape[0])

        # === segment the image ===
        # plant_segmentations, leaf_segmentations = segment_plants(image, bbox_dir, predictor)

        segmentation_mask = segment_plants(image, bboxes, predictor)
        watershed_mask, bboxes = segment_leaves_watershed(segmentation_mask, image, output_dir, stem)
        #
        # visualise_segmentations(watershed_mask, image, output_dir, stem)
        visualise_segmentation_random(watershed_mask, image, output_dir, stem)
        #
        # # can then segment using the bboxes from watershed using SAM again
        # leaf_segmentation = segment_plants(image, bboxes, predictor)
        #
        # stem = stem + "_watershed_sam"
        # visualise_segmentation_random(watershed_mask, image, output_dir, stem)

        # plant_segmentations, leaf_segmentations = segment_plants_leaves(image, bboxes, predictor)
        # visualise_segmentation_random(leaf_segmentations, image, output_dir, stem)


        # crop_images(image, output_dir, stem, segmentation_mask)


        # TEMP
        # print("before:", np.unique(segmentation_mask))
        # # segmentation_mask = segmentation_mask == 3
        # segmentation_mask = np.where(segmentation_mask == 3, 3, 0)
        # print("after:", np.unique(segmentation_mask))

        # visualise_segmentations(segmentation_mask, image, output_dir, stem)

        # watershed_segment_leaves(segmentation_mask, image)
        # get_leaf_contours_from_segmentation_mask(segmentation_mask, output_dir, stem, image)

        # if visualise:
        #     # === visualise the segmentations and save them as a png ===
        #     visualise_segmentations(segmentation_mask, image, output_dir, stem)



        # leaf_seg = segment_leaves_watershed(segmentation_mask)
        # visualise_segmentations(leaf_seg, image, output_dir, stem)

        # bboxes, centroids = get_leaf_contours_from_segmentation_mask(segmentation_mask, image, output_dir, stem)
        # segment_leaves_watershed(segmentation_mask, image, output_dir, stem)



    # === Clean up ===
    del predictor
    del sam
    torch.cuda.empty_cache()
    gc.collect()



