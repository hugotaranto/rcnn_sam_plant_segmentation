import numpy as np
import click
import os
import torch
from os.path import join, dirname, abspath
from torch.utils.data import DataLoader
from .dataloaders.datasets import Plants, collate_pdc, collate_pdc_inference, PlantsImages
from . import models
import yaml


def save_model(model, epoch, optim, name):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        }, name)

def rcnn_plant_detect_dir(config, ckpt_file, out, datapath):
    cfg = yaml.safe_load(open(config))

    val_dataset = Plants(datapath=datapath, overfit=cfg['train']['overfit'], is_inference=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc_inference, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    weights = torch.load(ckpt_file)["model_state_dict"]
    model.load_state_dict(weights)
    
    # os.makedirs(out, exist_ok=True)
    # os.makedirs(os.path.join(out,'plant_bboxes/'), exist_ok=True)
    os.makedirs(out, exist_ok=True)

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()

        for idx, item in enumerate(iter(val_loader)):
            with torch.no_grad():

                size = item['image'][0].shape[1]
                predictions = model.inference_test_step(item)

                res_names = item['name']
                for i in range(len(res_names)):
                    fname_box = os.path.join(out, res_names[i].replace('png','txt'))

                    size = item['image'][i].shape[1] # scales by the height!!!
                    scores = predictions[i]['scores'].cpu().numpy()
                    labels = predictions[i]['labels'].cpu().numpy()
                    # converting boxes to center, width, height format
                    boxes_ = predictions[i]['boxes'].cpu().numpy()
                    num_pred = len(boxes_)
                    cx = (boxes_[:,2] + boxes_[:,0])/2
                    cy = (boxes_[:,3] + boxes_[:,1])/2
                    bw = boxes_[:,2] - boxes_[:,0]
                    bh = boxes_[:,3] - boxes_[:,1]

                    # ready to be saved
                    pred_cls_box_score = np.hstack((
                                         labels.reshape(num_pred,1), 
                                         cx.reshape(num_pred,1)/size,
                                         cy.reshape(num_pred,1)/size,
                                         bw.reshape(num_pred,1)/size,
                                         bh.reshape(num_pred,1)/size,
                                         scores.reshape(num_pred,1)
                                        ))
                    np.savetxt(fname_box, pred_cls_box_score, fmt='%f')


def rcnn_plant_detect(config, ckpt_file, images):

    cfg = yaml.safe_load(open(config))

    val_dataset = PlantsImages(images=images, overfit=cfg['train']['overfit'])
    val_loader = DataLoader(val_dataset, batch_size=cfg['train']['batch_size'], collate_fn=collate_pdc_inference, shuffle=False, drop_last=False, num_workers=cfg['train']['workers'])

    model = models.get_model(cfg)
    weights = torch.load(ckpt_file)["model_state_dict"]
    model.load_state_dict(weights)
    
    bboxes = []

    with torch.autograd.set_detect_anomaly(True):
        model.network.eval()

        for idx, item in enumerate(iter(val_loader)):
            with torch.no_grad():

                predictions = model.inference_test_step(item)

                res_names = item['name']
                for i in range(len(res_names)):
                    boxes_ = predictions[i]['boxes'].cpu().numpy()
                    bboxes.append(boxes_)

    return np.array(bboxes)
