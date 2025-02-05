import copy
import h5py
from pathlib import Path
import cv2
import os
from ouster.sdk import client
import pickle
import random
import numpy as np
from ..dataset import DatasetTemplate
from . import m3ed_utils
from ...utils import box_utils
from m3ed_pcdet.config import cfg

class M3ED_SEQ:
    def __init__(self, sequence_path, dataset_cfg, class_names):
        self.dataset_cfg = dataset_cfg
        self.sequence_path = sequence_path
        self.class_names = class_names
        self.ori_class_names = ['Vehicle', 'Pedestrian', 'Pedestrian']
        files = os.listdir(self.sequence_path)
        h5_files = [file for file in files if file.endswith('.h5')]
        for file in h5_files:

            if 'data' in file:
                self.sequence_data_path = self.sequence_path /file
            if 'pose' in file:
                self.sequence_pose_path = self.sequence_path /file
        self.anno_label_path = self.sequence_path / 'ps_labels' / 'final_ps_dict.pkl'
        self.load_seq()

    def load_seq(self):
        f = h5py.File(self.sequence_data_path)
        self.sequence_data = f['/ouster/data']
        self.lidar_calib = f['/ouster/calib']
        self.lidar_pose = h5py.File(self.sequence_pose_path)['/Ln_T_L0'][...]
        self.sweeps_num = min([self.sequence_data.shape[0], self.lidar_pose.shape[0]])
        ouster_metadata_str = f['/ouster/metadata'][...].tolist()
        self.metadata = client.SensorInfo(ouster_metadata_str)
        self.xyzlut = client.XYZLut(self.metadata)
        rgb_camera_calib = f['/ovc/rgb/calib']
        Event_T_Lidar = self.lidar_calib['T_to_prophesee_left'][:]
        Event_T_RGB = rgb_camera_calib['T_to_prophesee_left'][:]

        Liar_T_RGB = m3ed_utils.transform_inv(Event_T_RGB) @ (Event_T_Lidar)
        self.extristric = Liar_T_RGB
        K = np.eye(3)
        K[0, 0] = rgb_camera_calib['intrinsics'][0]
        K[1, 1] = rgb_camera_calib['intrinsics'][1]
        K[0, 2] = rgb_camera_calib['intrinsics'][2]
        K[1, 2] = rgb_camera_calib['intrinsics'][3]
        width, height = rgb_camera_calib['resolution']
        self.K = K
        self.D = rgb_camera_calib['distortion_coeffs'][:]
        self.image_shape = np.array([width, height])

        with open(self.anno_label_path, 'rb') as f:
            anno_dict = pickle.load(f)
            self.sweeps_num = min([self.sweeps_num, len(anno_dict)])
        self.anno_dict = anno_dict

    def get_lidar(self, idx):
        ouster_sweep = self.sequence_data[idx][...]
        packets = client.Packets([client.LidarPacket(opacket, self.metadata) for opacket in ouster_sweep], self.metadata)
        scans = iter(client.Scans(packets))
        scan = next(scans)
        xyz = self.xyzlut(scan.field(client.ChanField.RANGE))
        signal = scan.field(client.ChanField.REFLECTIVITY)
        signal = client.destagger(self.metadata, signal)
        signal = np.divide(signal, np.amax(signal), dtype=np.float32)
        points = np.concatenate([xyz, signal[:,:,None]], axis=-1)
        points = points.reshape(-1,4)
        filtered_points = points[~np.all(points[:,:3] == [0, 0, 0], axis=1)]
        return filtered_points

    def remove_ego_points(self, points, center_radius=1.0):
        mask = ~((np.abs(points[:, 0]) < center_radius) & (np.abs(points[:, 1]) < center_radius))
        return points[mask]  

    def get_sequence_data(self, points, sample_idx, max_sweeps):
        """
        Transform historical frames to current frame with odometry and concatenate them
        """    
        points = self.remove_ego_points(points, center_radius=1.5)
        points = np.hstack([points, np.zeros((points.shape[0], 1)).astype(points.dtype)])        
        
        pose_cur = self.lidar_pose[sample_idx]
        pose_all = [pose_cur]
        used_num_sweeps = random.randint(1, max_sweeps+1)
        sample_idx_pre_list = np.clip(sample_idx + np.arange(-int(used_num_sweeps-1), 0), 0, 0x7FFFFFFF)
        sample_idx_pre_list = sample_idx_pre_list[::-1]

        points_pre_all = []
        for sample_idx_pre in sample_idx_pre_list:
            points_pre = self.get_lidar(sample_idx_pre)
            pose_pre = self.lidar_pose[sample_idx_pre].reshape((4, 4))
            expand_points_pre = np.concatenate([points_pre[:, :3], np.ones((points_pre.shape[0], 1))], axis=-1)
            points_pre_global = np.dot(expand_points_pre, m3ed_utils.transform_inv(pose_pre).T)[:, :3]
            expand_points_pre_global = np.concatenate([points_pre_global, np.ones((points_pre_global.shape[0], 1))], axis=-1)
            points_pre2cur = np.dot(expand_points_pre_global, pose_cur.T)[:, :3]
            points_pre = np.concatenate([points_pre2cur, points_pre[:, 3:]], axis=-1)
            
            # Append relative timestamp (each frame is 0.1s apart for 10Hz lidar)
            points_pre = np.hstack([points_pre, 0.1 * (sample_idx - sample_idx_pre) * np.ones((points_pre.shape[0], 1)).astype(points_pre.dtype)])  # one frame 0.1s
            
            points_pre = self.remove_ego_points(points_pre, 1)
            points_pre_all.append(points_pre)
            pose_all.append(pose_pre)
            
        points = np.concatenate([points] + points_pre_all, axis=0).astype(np.float32)
        poses = np.concatenate(pose_all, axis=0).astype(np.float32)

        return points, poses

    def get_fov_flag(self, points, mask=True, return_depth=False, downsample=1):
        extend_points = m3ed_utils.cart_to_hom(points[:,:3])
        points_cam = extend_points @ self.extristric.T
        rvecs = np.zeros((3,1))
        tvecs = np.zeros((3,1))

        pts_img, _ = cv2.projectPoints(points_cam[:,:3].astype(np.float32), rvecs, tvecs,
                self.K, self.D)
        imgpts = pts_img[:,0,:]
        depth = points_cam[:,2]

        if mask:
            imgpts = np.round(imgpts)
            kept1 = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < self.image_shape[1]) & \
                        (imgpts[:, 0] >= 0) & (imgpts[:, 0] < self.image_shape[0]) & \
                        (depth > 0.0)
            if not return_depth:
                return kept1
            
            else:
                width, height= self.image_shape / downsample
                depth_map = np.zeros((height, width))
                depth_map_mask = np.zeros((height, width)).astype(np.bool_)

                kept1 = (imgpts[:, 1] >= 0) & (imgpts[:, 1] < height) & \
                            (imgpts[:, 0] >= 0) & (imgpts[:, 0] < width) & \
                            (depth > 0.0)

                imgpts, depth = imgpts[kept1], depth[kept1]
                ranks = imgpts[:, 0] + imgpts[:, 1] * width
                sort = np.argsort((ranks + 1 - depth / 100.))
                imgpts, depth, ranks = imgpts[sort], depth[sort], ranks[sort]

                kept2 = np.ones(imgpts.shape[0]).astype(np.bool_)
                kept2[1:] = (ranks[1:] != ranks[:-1])
                imgpts, depth = imgpts[kept2], depth[kept2]
                imgpts = imgpts.astype(np.int64)
                depth_map[imgpts[:, 1], imgpts[:, 0]] = depth
                depth_map_mask[imgpts[:, 1], imgpts[:, 0]] = True

                return kept1, depth_map, depth_map_mask
        else:
            imgpts[:, 1] = np.clip(imgpts[:, 1], 0, self.image_shape[1] - 1)
            imgpts[:, 0] = np.clip(imgpts[:, 0], 0, self.image_shape[0] - 1)
            return imgpts, depth

    def __getitem__(self, index):
        input_dict = {}
        input_dict['frame_id'] = str(index).zfill(5)

        frame_index = index
        points = self.get_lidar(frame_index)
        points, _ = self.get_sequence_data(points, frame_index, max_sweeps=self.dataset_cfg.MAX_SWEEPS)
        if self.dataset_cfg.FOV_POINTS_ONLY:
            fov_flag = self.get_fov_flag(points)
            points = points[fov_flag]

        if self.dataset_cfg.get('SHIFT_COOR', None):
            points[:, 0:3] += np.array(self.dataset_cfg.SHIFT_COOR, dtype=np.float32)
        input_dict['points'] = points
        frame_id = str(index).zfill(5)
        gt_boxes_info = self.anno_dict[frame_id]['gt_boxes']
        if gt_boxes_info.shape[0] > 0:
            gt_names = np.array(self.ori_class_names)[np.abs(gt_boxes_info[:,-2].astype(np.int32)) - 1]
            gt_boxes_lidar = gt_boxes_info[:,:7]
            if self.dataset_cfg.get('SHIFT_COOR', None):
                gt_boxes_lidar[:, 0:3] += self.dataset_cfg.SHIFT_COOR
            input_dict.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_lidar
            })
        else:
            input_dict.update({
                'gt_names': np.empty([0]).astype(str),
                'gt_boxes': np.zeros([0, 7])
            })

        if cfg.get('SELF_TRAIN', None) and cfg.SELF_TRAIN.get('REPORT_PS_LABEL_QUALITY', None) and cfg.SELF_TRAIN.REPORT_PS_LABEL_QUALITY:
            gt_names = np.array(['Vehicle'if item == 'Car' else item for item in gt_names])

            input_dict.update({
                'gt_names': gt_names,
            })

        return input_dict

class OFFM3EDDatasetSeqs(DatasetTemplate):

    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        sequences = dataset_cfg.INFO_SEQUENCES[self.mode]
        sequence_paths = [Path(self.root_path) / seq_path for seq_path in \
                              sequences]
        self.ori_class_names = ['Vehicle', 'Pedestrian', 'Pedestrian']
        self.include_m3ed_sequence_data(sequence_paths)

    def include_m3ed_sequence_data(self, sequence_paths):
        if self.logger is not None:
            self.logger.info('Loading M3ED dataset')

        self.seq_num = len(sequence_paths)
        for i in range(self.seq_num):
            setattr(self, f'seq_{i}', M3ED_SEQ(sequence_paths[i], 
                                               self.dataset_cfg,
                                               self.class_names))

        seq_frame_num = [getattr(getattr(self, f'seq_{i}'), 'sweeps_num') \
                         for i in range(self.seq_num)]
        frame_info = []
        for i, num in enumerate(seq_frame_num):
            frame_info.extend(
                [f'{i}_{frame_id}' for frame_id in range(num)]
            )
        self.frame_info = frame_info
        if self.logger is not None:
            self.logger.info('Total samples for M3ED dataset: %d' % (len(frame_info)))

    def generate_prediction_dicts(self, batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:
        Returns:
        """
        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7]), 'pred_labels': np.zeros(num_samples)
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            if self.dataset_cfg.get('SHIFT_COOR', None):
                pred_boxes[:, 0:3] -= self.dataset_cfg.SHIFT_COOR

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes
            pred_dict['pred_labels'] = pred_labels

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            annos.append(single_pred_dict)

        return annos


    def kitti_eval(self, eval_det_annos, eval_gt_annos, class_names):
        from ..kitti.kitti_object_eval_python import eval as kitti_eval

        map_name_to_kitti = {
            'car': 'Car',
            'Vehicle': 'Car',
            'Pedestrian': 'Pedestrian'
        }

        def transform_to_kitti_format(annos, info_with_fakelidar=False, is_gt=False):
            for anno in annos:
                if 'name' not in anno:
                    anno['name'] = anno['gt_names']
                    anno.pop('gt_names')

                for k in range(anno['name'].shape[0]):
                    if anno['name'][k] in map_name_to_kitti:
                        anno['name'][k] = map_name_to_kitti[anno['name'][k]]
                    else:
                        anno['name'][k] = 'Person_sitting'

                if 'boxes_lidar' in anno:
                    gt_boxes_lidar = anno['boxes_lidar'].copy()
                else:
                    gt_boxes_lidar = anno['gt_boxes'].copy()

                # filter by range
                if self.dataset_cfg.get('GT_FILTER', None) and \
                        self.dataset_cfg.GT_FILTER.RANGE_FILTER:
                    if self.dataset_cfg.GT_FILTER.get('RANGE', None):
                        point_cloud_range = self.dataset_cfg.GT_FILTER.RANGE
                    else:
                        point_cloud_range = self.point_cloud_range
                        point_cloud_range[2] = -10
                        point_cloud_range[5] = 10

                    mask = box_utils.mask_boxes_outside_range_numpy(gt_boxes_lidar,
                                                                    point_cloud_range,
                                                                    min_num_corners=1)
                    gt_boxes_lidar = gt_boxes_lidar[mask]
                    anno['name'] = anno['name'][mask]
                    if not is_gt:
                        anno['score'] = anno['score'][mask]
                        anno['pred_labels'] = anno['pred_labels'][mask]

                # filter by fov
                if is_gt and self.dataset_cfg.get('GT_FILTER', None):
                    if self.dataset_cfg.GT_FILTER.get('FOV_FILTER', None):
                        fov_gt_flag = self.extract_fov_gt(
                            gt_boxes_lidar, self.dataset_cfg['FOV_DEGREE'], self.dataset_cfg['FOV_ANGLE']
                        )
                        gt_boxes_lidar = gt_boxes_lidar[fov_gt_flag]
                        anno['name'] = anno['name'][fov_gt_flag]

                anno['bbox'] = np.zeros((len(anno['name']), 4))
                anno['bbox'][:, 2:4] = 50  # [0, 0, 50, 50]
                anno['truncated'] = np.zeros(len(anno['name']))
                anno['occluded'] = np.zeros(len(anno['name']))

                if len(gt_boxes_lidar) > 0:
                    if info_with_fakelidar:
                        gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(gt_boxes_lidar)

                    gt_boxes_lidar[:, 2] -= gt_boxes_lidar[:, 5] / 2
                    anno['location'] = np.zeros((gt_boxes_lidar.shape[0], 3))
                    anno['location'][:, 0] = -gt_boxes_lidar[:, 1]  # x = -y_lidar
                    anno['location'][:, 1] = -gt_boxes_lidar[:, 2]  # y = -z_lidar
                    anno['location'][:, 2] = gt_boxes_lidar[:, 0]  # z = x_lidar
                    dxdydz = gt_boxes_lidar[:, 3:6]
                    anno['dimensions'] = dxdydz[:, [0, 2, 1]]  # lwh ==> lhw
                    anno['rotation_y'] = -gt_boxes_lidar[:, 6] - np.pi / 2.0
                    anno['alpha'] = -np.arctan2(-gt_boxes_lidar[:, 1], gt_boxes_lidar[:, 0]) + anno['rotation_y']
                else:
                    anno['location'] = anno['dimensions'] = np.zeros((0, 3))
                    anno['rotation_y'] = anno['alpha'] = np.zeros(0)

        # self.filter_det_results(eval_det_annos, self.oracle_infos)
        transform_to_kitti_format(eval_det_annos)
        transform_to_kitti_format(eval_gt_annos, info_with_fakelidar=False, is_gt=True)

        kitti_class_names = []
        for x in class_names:
            if x in map_name_to_kitti:
                kitti_class_names.append(map_name_to_kitti[x])
            else:
                kitti_class_names.append('Person_sitting')
        ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
        )
        return ap_result_str, ap_dict

    def anno_dict2_list(self):
        anno_dict = copy.deepcopy(getattr(getattr(self, f'seq_{0}'), 'anno_dict'))
        anno_list = []
        for frame_id, box_info in anno_dict.items():
            single_anno_info = {}
            gt_boxes_3d = box_info['gt_boxes']
            gt_names = np.array(self.ori_class_names)[np.abs(gt_boxes_3d[:,-2].astype(np.int32)) - 1]
            single_anno_info.update({
                'gt_names': gt_names,
                'gt_boxes': gt_boxes_3d[:,:7]
            })
            anno_list.append(single_anno_info)
        return anno_list

    def evaluation(self, det_annos, class_names, **kwargs):
        if kwargs['eval_metric'] == 'kitti':
            eval_det_annos = copy.deepcopy(det_annos)
            eval_gt_annos = copy.deepcopy(self.anno_dict2_list())
            return self.kitti_eval(eval_det_annos, eval_gt_annos, class_names)
        else:
            raise NotImplementedError

    def __len__(self):

        if self._merge_all_iters_to_one_epoch:
            return len(self.frame_info) * self.total_epochs

        return len(self.frame_info)


    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.frame_info)

        single_frame_info = self.frame_info[index]
        seq_idx, frame_idx = [eval(fac) for fac in single_frame_info.split('_')]
        input_dict = getattr(self, f'seq_{seq_idx}').__getitem__(frame_idx)


        if self.dataset_cfg.get('REMOVE_ORIGIN_GTS', None) and self.training:
            input_dict['points'] = box_utils.remove_points_in_boxes3d(input_dict['points'], input_dict['gt_boxes'])
            mask = np.zeros(input_dict['gt_boxes'][:,:7].shape[0], dtype=np.bool_)
            input_dict['gt_boxes'] = input_dict['gt_boxes'][mask]
            input_dict['gt_names'] = input_dict['gt_names'][mask]

        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            input_dict['gt_boxes'] = None

        # load saved pseudo label for unlabel data
        if self.dataset_cfg.get('USE_PSEUDO_LABEL', None) and self.training:
            self.fill_pseudo_labels(input_dict)

        # load free space
        if self.training:
            aug_names = [aug.NAME for aug in self.dataset_cfg.DATA_AUGMENTOR.AUG_CONFIG_LIST]
            if 'occ_gt_sampling' in aug_names:
                free_logits_file_path = self.root_path / 'free_space_logits' / 'training' / ('%s.npy' % input_dict['frame_id'])
                input_dict['free_space_logits'] = np.load(free_logits_file_path)

        data_dict = self.prepare_data(data_dict=input_dict)

        return data_dict
    
