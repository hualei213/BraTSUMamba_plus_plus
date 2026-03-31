import warnings

import monai.metrics
import numpy
import numpy as np
import torch
from medpy import metric

from medpy.metric import binary
from monai.metrics import HausdorffDistanceMetric


def dice(pred, label):
    if (label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
def hd(pred, gt):
    if pred.sum() > 0 and gt.sum() > 0:
        hd95 = binary.hd95(pred, gt)
        return hd95
    else:
        return 0


def get_percentile_distance_monai(mask_pred, mask_gt, percentile=95):
    mask_pred = np.expand_dims(mask_pred,axis=[0,1])
    mask_gt = np.expand_dims(mask_gt,axis=[0,1])
    print(mask_pred.shape)
    hd = monai.metrics.compute_hausdorff_distance(y_pred=mask_pred,y=mask_gt,percentile=percentile,directed=False)

    # mask_pred = np.expand_dims(mask_pred,axis=[0])
    # mask_gt = np.expand_dims(mask_gt,axis=[0])
    # mask_pred = torch.from_numpy(mask_pred)
    # mask_gt = torch.from_numpy(mask_gt)
    # hausdorff_metric = HausdorffDistanceMetric(include_background=False, reduction='mean', percentile=percentile)
    # hd = hausdorff_metric(y_pred=mask_pred,y=mask_gt)
    return hd.item()


def get_percentile_distance(mask_pred, mask_gt, percentile=95):
    if np.any(mask_gt) and np.any(mask_pred):
        pred_gt_distance = metric.binary.__surface_distances(mask_pred, mask_gt)
        gt_pred_distance = metric.binary.__surface_distances(mask_gt, mask_pred)
        result = np.concatenate([pred_gt_distance, gt_pred_distance])
        return np.percentile(result, percentile)
    else:
        return 0
