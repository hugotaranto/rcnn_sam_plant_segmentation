import os
from rcnn.test import rcnn_plant_detect
from semantic_sam import segment_images

rcnn_config_file = './rcnn/configs/config.yaml' # configuration for rcnn plant detection
rcnn_checkpoint = './rcnn/last.pt'              # rcnn checkpoint .pt file
# datapath = './images'                           # directory containing .png images
datapath = '../phenobench/image_test/resized_images'
# datapath = './single'
output_dir = './output'                         # directory to output results to
sam_checkpoint = './sam_base_checkpoint.pth'    # SAM checkpoint .pth file (using base model)

os.makedirs(output_dir, exist_ok=True)
# bbox_output_dir = os.path.join(output_dir, 'plant_bboxes')
semantic_segment_output_dir = os.path.join(output_dir, 'semantics')

# print("RCNN finding Bounding Boxes")
# rcnn_plant_detect(rcnn_config_file, rcnn_checkpoint, bbox_output_dir, datapath)
# print("Bounding Boxes Saved to {}".format(bbox_output_dir))

temp_bbox_dir = "../phenobench/phenobench-baselines/leaf_detection/yolov7/src/runs/detect/exp9/labels"

print("Segmenting with Sam")
# segment_images(datapath, bbox_output_dir, semantic_segment_output_dir, sam_checkpoint)
segment_images(datapath, temp_bbox_dir, semantic_segment_output_dir, sam_checkpoint, visualise=True)
print("Visuals saved to {}".format(semantic_segment_output_dir))

