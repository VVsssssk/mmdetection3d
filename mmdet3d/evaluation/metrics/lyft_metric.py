# Copyright (c) OpenMMLab. All rights reserved.
import logging
import os
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmcv
import numpy as np
import pandas as pd
from lyft_dataset_sdk.lyftdataset import LyftDataset as Lyft
from lyft_dataset_sdk.utils.data_classes import Box as LyftBox
from mmengine import load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from pyquaternion import Quaternion

from mmdet3d.evaluation import lyft_eval
from mmdet3d.registry import METRICS


@METRICS.register_module()
class LyftMetric(BaseMetric):
    """Lyft evaluation metric.

    Args:
        data_root (str): Path of dataset root.
        ann_file (str): Path of annotation file.
        metric (str | list[str]): Metrics to be evaluated.
            Default to 'bbox'.
        modality (dict): Modality to specify the sensor data used
            as input. Defaults to dict(use_camera=False, use_lidar=True).
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
        jsonfile_prefix (str, optional): The prefix of json files including
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Default: None.
        csv_savepath (str, optional): The path for saving csv files.
            It includes the file path and the csv filename,
            e.g., "a/b/filename.csv". If not specified,
            the result will not be converted to csv file.
        collect_device (str): Device name used for collecting results
            from different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
    """

    def __init__(
        self,
        data_root: str,
        ann_file: str,
        metric: Union[str, List[str]] = 'bbox',
        modality=dict(
            use_camera=False,
            use_lidar=True,
        ),
        prefix: Optional[str] = None,
        jsonfile_prefix: str = None,
        csv_savepath: str = None,
        collect_device: str = 'cpu',
        file_client_args: dict = dict(backend='disk')
    ) -> None:
        self.default_prefix = 'Lyft metric'
        super(LyftMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.jsonfile_prefix = jsonfile_prefix
        self.file_client_args = file_client_args
        self.csv_savepath = csv_savepath
        self.metrics = metric if isinstance(metric, list) else [metric]

    def process(self, data_batch: Sequence[dict],
                predictions: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``,
        which will be used to compute the metrics when all batches
        have been processed.

        Args:
            data_batch (Sequence[dict]): A batch of data
                from the dataloader.
            predictions (Sequence[dict]): A batch of outputs from
                the model.
        """
        assert len(data_batch) == len(predictions)
        for data, pred in zip(data_batch, predictions):
            result = dict()
            for pred_result in pred:
                if pred[pred_result] is not None:
                    for attr_name in pred[pred_result]:
                        pred[pred_result][attr_name] = pred[pred_result][
                            attr_name].to(self.collect_device)
                    result[pred_result] = pred[pred_result]
                sample_idx = data['data_sample']['sample_idx']
                result['sample_idx'] = sample_idx
        self.results.append(result)

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['CLASSES']
        self.version = self.dataset_meta['version']
        # load annotations

        self.data_infos = load(
            self.ann_file, file_client_args=self.file_client_args)['data_list']
        result_dict, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix)

        metric_dict = {}
        for metric in self.metrics:
            ap_dict = self.lyft_evaluate(
                result_dict, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def format_results(self,
                       results: List[dict],
                       classes: List[str] = None,
                       jsonfile_prefix: str = None,
                       csv_savepath: str = None) -> Tuple:
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[dict]): Testing results of the dataset.
            classes (list[String], optional): A list of class name. Defaults
                to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Default: None.
            csv_savepath (str, optional): The path for saving csv files.
                It includes the file path and the csv filename,
                e.g., "a/b/filename.csv". If not specified,
                the result will not be converted to csv file.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where `result_dict` is a
                dict containing the json filepaths, `tmp_dir` is the temporal
                directory created for saving json files when
                `jsonfile_prefix` is not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_dict = dict()
        sample_id_list = [result['sample_idx'] for result in results]

        for name in results[0]:
            if 'pred' in name and '3d' in name and name[0] != '_':
                print(f'\nFormating bboxes of {name}')
                # format result of model output in Det3dDataSample,
                # include 'pred_instances_3d','pts_pred_instances_3d',
                # 'img_pred_instances_3d'
                results_ = [out[name] for out in results]
                tmp_file_ = osp.join(jsonfile_prefix, name)
                result_dict[name] = self._format_bbox(results_, sample_id_list,
                                                      classes, tmp_file_)
        if csv_savepath is not None:
            if 'pred_instances_3d' in result_dict:
                self.json2csv(result_dict['pred_instances_3d'], csv_savepath)
            elif 'pts_pred_instances_3d' in result_dict:
                self.json2csv(result_dict['pts_pred_instances_3d'],
                              csv_savepath)
        return result_dict, tmp_dir

    def json2csv(self, json_path: str, csv_savepath: str) -> None:
        """Convert the json file to csv format for submission.

        Args:
            json_path (str): Path of the result json file.
            csv_savepath (str): Path to save the csv file.
        """
        results = mmcv.load(json_path)['results']
        sample_list_path = osp.join(self.data_root, 'sample_submission.csv')
        data = pd.read_csv(sample_list_path)
        Id_list = list(data['Id'])
        pred_list = list(data['PredictionString'])
        cnt = 0
        print('Converting the json to csv...')
        for token in results.keys():
            cnt += 1
            predictions = results[token]
            prediction_str = ''
            for i in range(len(predictions)):
                prediction_str += \
                    str(predictions[i]['score']) + ' ' + \
                    str(predictions[i]['translation'][0]) + ' ' + \
                    str(predictions[i]['translation'][1]) + ' ' + \
                    str(predictions[i]['translation'][2]) + ' ' + \
                    str(predictions[i]['size'][0]) + ' ' + \
                    str(predictions[i]['size'][1]) + ' ' + \
                    str(predictions[i]['size'][2]) + ' ' + \
                    str(Quaternion(list(predictions[i]['rotation']))
                        .yaw_pitch_roll[0]) + ' ' + \
                    predictions[i]['name'] + ' '
            prediction_str = prediction_str[:-1]
            idx = Id_list.index(token)
            pred_list[idx] = prediction_str
        df = pd.DataFrame({'Id': Id_list, 'PredictionString': pred_list})
        mmcv.mkdir_or_exist(os.path.dirname(csv_savepath))
        df.to_csv(csv_savepath, index=False)

    def _format_bbox(self,
                     results: List[dict],
                     sample_id_list: List[int],
                     classes: List[str] = None,
                     jsonfile_prefix: str = None) -> str:
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            sample_id_list (list[int]): List of result sample id.
            classes (list[String], optional): A list of class name. Defaults
                to None.
            jsonfile_prefix (str, optional): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        lyft_annos = {}

        print('Start to convert detection format...')
        for i, det in enumerate(mmcv.track_iter_progress(results)):
            annos = []
            boxes = output_to_lyft_box(det)
            sample_id = sample_id_list[i]
            sample_token = self.data_infos[sample_id]['token']
            boxes = lidar_lyft_box_to_global(self.data_infos[sample_id], boxes)
            for i, box in enumerate(boxes):
                name = classes[box.label]
                lyft_anno = dict(
                    sample_token=sample_token,
                    translation=box.center.tolist(),
                    size=box.wlh.tolist(),
                    rotation=box.orientation.elements.tolist(),
                    name=name,
                    score=box.score)
                annos.append(lyft_anno)
            lyft_annos[sample_token] = annos
        lyft_submissions = {
            'meta': self.modality,
            'results': lyft_annos,
        }

        mmcv.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_lyft.json')
        print('Results writes to', res_path)
        mmcv.dump(lyft_submissions, res_path)
        return res_path

    def lyft_evaluate(self,
                      result_dict: dict,
                      metric: str = 'bbox',
                      logger: logging.Logger = None) -> dict:
        """Evaluation in Lyft protocol.

        Args:
            result_dict (dict): Formatted results of the dataset.
            metric (str): Metrics to be evaluated.
                Default: 'bbox'.
            classes (list[String], optional): A list of class name. Defaults
                to None.
            logger (MMLogger, optional): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Evaluation results.
        """
        metric_dict = dict()
        for name in result_dict:
            print('Evaluating bboxes of {}'.format(name))
            ret_dict = self._evaluate_single(
                result_dict[name], logger=logger, result_name=name)
        metric_dict.update(ret_dict)
        return metric_dict

    def _evaluate_single(self,
                         result_path: str,
                         logger: MMLogger = None,
                         result_name: str = 'pts_bbox') -> dict:
        """Evaluation for a single model in Lyft protocol.

        Args:
            result_path (str): Path of the result file.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Default: None.
            metric (str): Metric name used for evaluation.
                Default: 'bbox'.
            result_name (str): Result name in the metric prefix.
                Default: 'pts_bbox'.

        Returns:
            dict: Dictionary of evaluation details.
        """
        output_dir = osp.join(*osp.split(result_path)[:-1])
        lyft = Lyft(
            data_path=osp.join(self.data_root, self.version),
            json_path=osp.join(self.data_root, self.version, self.version),
            verbose=True)
        eval_set_map = {
            'v1.01-train': 'val',
        }
        metrics = lyft_eval(lyft, self.data_root, result_path,
                            eval_set_map[self.version], output_dir, logger)

        # record metrics
        detail = dict()
        metric_prefix = f'{result_name}_Lyft'

        for i, name in enumerate(metrics['class_names']):
            AP = float(metrics['mAPs_cate'][i])
            detail[f'{metric_prefix}/{name}_AP'] = AP

        detail[f'{metric_prefix}/mAP'] = metrics['Final mAP']
        return detail


def output_to_lyft_box(detection: dict) -> List[LyftBox]:
    """Convert the output to the box class in the Lyft.

    Args:
        detection (dict): Detection results.

    Returns:
        list[:obj:`LyftBox`]: List of standard LyftBoxes.
    """
    bbox3d = detection['bbox_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    # our LiDAR coordinate system -> Lyft box coordinate system
    lyft_box_dims = box_dims[:, [1, 0, 2]]

    box_list = []
    for i in range(len(bbox3d)):
        quat = Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
        box = LyftBox(
            box_gravity_center[i],
            lyft_box_dims[i],
            quat,
            label=labels[i],
            score=scores[i])
        box_list.append(box)
    return box_list


def lidar_lyft_box_to_global(info: dict,
                             boxes: List[LyftBox]) -> List[LyftBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the
            calibration information.
        boxes (list[:obj:`LyftBox`]): List of predicted LyftBoxes.

    Returns:
        list: List of standard LyftBoxes in the global
            coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        box.rotate(Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list
