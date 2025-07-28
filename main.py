import os
from rcnn.test import rcnn_plant_detect
# from semantic_sam import segment_plants
from semantic_sam import segment_images

rcnn_config_file = './rcnn/configs/config.yaml'
rcnn_checkpoint = './rcnn/last.pt'
datapath = './images'
output_dir = './output'
os.makedirs(output_dir, exist_ok=True)

bbox_output_dir = os.path.join(output_dir, 'plant_bboxes')

sam_checkpoint = './sam_base_checkpoint.pth'
semantic_segment_output_dir = os.path.join(output_dir, 'semantics')

print("RCNN finding Bounding Boxes")
rcnn_plant_detect(rcnn_config_file, rcnn_checkpoint, bbox_output_dir, datapath)
print("Bounding Boxes Saved to {}".format(bbox_output_dir))

print("Segmenting with Sam")
segment_images(datapath, bbox_output_dir, semantic_segment_output_dir, sam_checkpoint)
print("Visuals saved to {}".format(semantic_segment_output_dir))

