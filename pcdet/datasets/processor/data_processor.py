from functools import partial
import math,torch
torch.cuda.current_device()
import numpy as np
from skimage import transform

from ...ops.pointnet2.pointnet2_stack import pointnet2_utils as pointnet2_stack_utils
from ...utils import box_utils, common_utils

tv = None
try:
    import cumm.tensorview as tv
except:
    pass


class VoxelGeneratorWrapper():
    def __init__(self, vsize_xyz, coors_range_xyz, num_point_features, max_num_points_per_voxel, max_num_voxels):
        try:
            from spconv.utils import VoxelGeneratorV2 as VoxelGenerator
            self.spconv_ver = 1
        except:
            try:
                from spconv.utils import VoxelGenerator
                self.spconv_ver = 1
            except:
                from spconv.utils import Point2VoxelCPU3d as VoxelGenerator
                self.spconv_ver = 2

        if self.spconv_ver == 1:
            self._voxel_generator = VoxelGenerator(
                voxel_size=vsize_xyz,
                point_cloud_range=coors_range_xyz,
                max_num_points=max_num_points_per_voxel,
                max_voxels=max_num_voxels
            )
        else:
            self._voxel_generator = VoxelGenerator(
                vsize_xyz=vsize_xyz,
                coors_range_xyz=coors_range_xyz,
                num_point_features=num_point_features,
                max_num_points_per_voxel=max_num_points_per_voxel,
                max_num_voxels=max_num_voxels
            )

    def generate(self, points):
        if self.spconv_ver == 1:
            voxel_output = self._voxel_generator.generate(points)
            if isinstance(voxel_output, dict):
                voxels, coordinates, num_points = \
                    voxel_output['voxels'], voxel_output['coordinates'], voxel_output['num_points_per_voxel']
            else:
                voxels, coordinates, num_points = voxel_output
        else:
            assert tv is not None, f"Unexpected error, library: 'cumm' wasn't imported properly."
            voxel_output = self._voxel_generator.point_to_voxel(tv.from_numpy(points))
            tv_voxels, tv_coordinates, tv_num_points = voxel_output
            # make copy with numpy(), since numpy_view() will disappear as soon as the generator is deleted
            voxels = tv_voxels.numpy()
            coordinates = tv_coordinates.numpy()
            num_points = tv_num_points.numpy()
        return voxels, coordinates, num_points


class DataProcessor(object):
    def __init__(self, processor_configs, point_cloud_range, training, num_point_features):
        self.point_cloud_range = point_cloud_range
        self.training = training
        self.num_point_features = num_point_features
        self.mode = 'train' if training else 'test'
        self.grid_size = self.voxel_size = None
        self.data_processor_queue = []

        self.voxel_generator = None

        for cur_cfg in processor_configs:
            cur_processor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_processor_queue.append(cur_processor)

    def mask_points_and_boxes_outside_range(self, data_dict=None, config=None):
        # 如果data_dict为空，则返回一个函数，该函数将在稍后被调用
        if data_dict is None:
            return partial(self.mask_points_and_boxes_outside_range, config=config)

        # 如果data_dict中存在points，则将其范围限制在point_cloud_range内
        if data_dict.get('points', None) is not None:
            mask = common_utils.mask_points_by_range(data_dict['points'], self.point_cloud_range)
            data_dict['points'] = data_dict['points'][mask]

        # 如果data_dict中存在gt_boxes，则将其范围限制在point_cloud_range内
        if data_dict.get('gt_boxes', None) is not None and config.REMOVE_OUTSIDE_BOXES and self.training:
            mask = box_utils.mask_boxes_outside_range_numpy(
                data_dict['gt_boxes'], self.point_cloud_range, min_num_corners=config.get('min_num_corners', 1), 
                use_center_to_filter=config.get('USE_CENTER_TO_FILTER', True)
            )
            data_dict['gt_boxes'] = data_dict['gt_boxes'][mask]
        return data_dict

    # 乱序
    def shuffle_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.shuffle_points, config=config)

        if config.SHUFFLE_ENABLED[self.mode]:
            points = data_dict['points']
            shuffle_idx = np.random.permutation(points.shape[0])
            points = points[shuffle_idx]
            data_dict['points'] = points

        return data_dict

    def transform_points_to_voxels_placeholder(self, data_dict=None, config=None):
        # just calculate grid size
        if data_dict is None:
            # 网格大小
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.transform_points_to_voxels_placeholder, config=config)
        
        return data_dict
        
    def transform_points_to_voxels(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            # just bind the config, we will create the VoxelGeneratorWrapper later,
            # to avoid pickling issues in multiprocess spawn
            return partial(self.transform_points_to_voxels, config=config)

        if self.voxel_generator is None:
            self.voxel_generator = VoxelGeneratorWrapper(
                vsize_xyz=config.VOXEL_SIZE,
                coors_range_xyz=self.point_cloud_range,
                num_point_features=self.num_point_features,
                max_num_points_per_voxel=config.MAX_POINTS_PER_VOXEL,
                max_num_voxels=config.MAX_NUMBER_OF_VOXELS[self.mode],
            )

        points = data_dict['points']
        voxel_output = self.voxel_generator.generate(points)
        voxels, coordinates, num_points = voxel_output

        if not data_dict['use_lead_xyz']:
            voxels = voxels[..., 3:]  # remove xyz in voxels(N, 3)

        data_dict['voxels'] = voxels
        data_dict['voxel_coords'] = coordinates
        data_dict['voxel_num_points'] = num_points
        return data_dict

    """
    点云随机采样，其实就是随机的，从所有的点云采样到16384个。
    并不是最远点采样
    选择一些点，凑满num_points，pointrcnn里是16384
    选点的过程如下:
    """
    def sample_points(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.sample_points, config=config)

        num_points = config.NUM_POINTS[self.mode]
        if num_points == -1:  # 意思是不采样吧
            return data_dict

        points = data_dict['points']

        # 1、想采样的点云数 < 输入的点云数：划分近点和远点（距原点40m）
        if num_points < len(points):
            pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
            pts_near_flag = pts_depth < 40.0
            far_idxs_choice = np.where(pts_near_flag == 0)[0]
            near_idxs = np.where(pts_near_flag == 1)[0]
            choice = []
            # （1)若想要的点云数 > 远点数量 ，则从近处随机选点来补充
            if num_points > len(far_idxs_choice):
                near_idxs_choice = np.random.choice(near_idxs, num_points - len(far_idxs_choice), replace=False)
                choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0) \
                    if len(far_idxs_choice) > 0 else near_idxs_choice
            # （2）否则，从远处近处随机选
            else: 
                choice = np.arange(0, len(points), dtype=np.int32)
                choice = np.random.choice(choice, num_points, replace=False)
            np.random.shuffle(choice)
        # 2、想采样的点云数 > 输入数： 多出来的，从输入重复随机选取
        else:
            choice = np.arange(0, len(points), dtype=np.int32)
            if num_points > len(points):
                extra_choice = np.random.choice(choice, num_points - len(points), replace=False)
                choice = np.concatenate((choice, extra_choice), axis=0)
            np.random.shuffle(choice)
        data_dict['points'] = points[choice]
        return data_dict

    # 基于距离的FPS采样
    def bin_based_fps(self, data_dict=None, config=None, num_sampled_points=[], distance_list=[]):
        """
        Args:
            points: (N, 3)
            num_sampled_points_list: list[int]
            distance_list: list[int]

        Returns:
            sampled_points: (N_out, 3)
        """
        if data_dict is None:
            return partial(self.bin_based_fps, config=config)

        # 在多卡时用cuda，单卡时用cpu
        device = 'cuda'
        points = data_dict['points'] # (N, 4)
        sampled_ratios = config.sampled_ratios # [0.6,  0.7,  0.8, 0.9, 1]
        distance_list  = config.distance_list # [0,    15,   30,  45,  60, 1000]
        points_distances = np.linalg.norm(points[:, 0:3], axis=1)
        points = torch.from_numpy(points).float().to(device)
        # 初始化一些变量
        xyzr_points_list = []
        xyzr_batch_cnt = []
        num_sampled_points_list = []
        cur_num_points_list = []
        for k in range(len(distance_list)-1):
            # 计算属于该区间的点的掩码      
            mask = (distance_list[k] <= points_distances) & (points_distances < distance_list[k+1])
            # 计算该距离区间的点的数量
            cur_num_points = mask.sum().item() # 转标量
            # 若当前区间存在点
            if cur_num_points > 0:
                # 采样前的点，采用前点的数量，采样后的点的数量
                xyzr_points_list.append(points[mask])
                xyzr_batch_cnt.append(cur_num_points)
                # 计算将采样点的数目
                cur_num_points_list.append(cur_num_points)
                num_sampled_points_list.append(
                    math.ceil(sampled_ratios[k] * cur_num_points)
                )
        
        # print(f'xyz_batch_cnt={xyzr_batch_cnt}')
        # print(f'num_sampled_points_list={num_sampled_points_list},sum={sum(num_sampled_points_list)}')
        # print(f'cur_num_points_list={cur_num_points_list},sum={sum(cur_num_points_list)}')
        # 如果所有距离区间都没有点，几乎不会发生
        if len(xyzr_batch_cnt) == 0:
            # 将所有点、点的数量、采样点的数量加入列表
            xyzr_points_list.append(points)
            xyzr_batch_cnt.append(len(points))
            num_sampled_points_list.append(num_sampled_points)
            print(f'Warning: empty points detected in distance_FPS: points.shape={points.shape}')
        # 将所有点拼接在一起，注意，这里points里点的顺序变了
        points = torch.cat(xyzr_points_list, dim=0)
        # 将每个区间的点的数量转换为张量，.int()把所有元素转换为整数型
        xyzr_batch_cnt = torch.tensor(xyzr_batch_cnt, device=points.device).int()
        # 将每个扇形采样点的数量转换为张量
        sampled_points_batch_cnt = torch.tensor(num_sampled_points_list, device=points.device).int()
        # 对所有点进行FPS采样
        sampled_pt_idxs = pointnet2_stack_utils.stack_farthest_point_sample(
            points[:,0:3].contiguous(), xyzr_batch_cnt, sampled_points_batch_cnt
        ).long()

        # 为了凑整，额外需要的点
        num_extra_points = config.NUM_POINTS[self.mode] - sampled_pt_idxs.shape[0]
        # 额外采样
        if num_extra_points >= 0:
            extra_index = torch.from_numpy(np.random.choice(range(points.shape[0]), num_extra_points)).cuda()
            sampled_pt_idxs = torch.cat([sampled_pt_idxs, extra_index], dim=0)
            # print(f'sampled_points.shape={sampled_pt_idxs.shape}')
        # 采样点过多，随机删除 TODO:这里会卡死 
        elif num_extra_points < 0:
            sampled_pt_idxs = sampled_pt_idxs[torch.randperm(sampled_pt_idxs.shape[0])][:config.NUM_POINTS[self.mode]]
            print(f'sampled_points.shape={sampled_pt_idxs.shape}')
        # 根据采样点的索引得到采样点
        sampled_points = points[sampled_pt_idxs]
        # print(f'sampled_points.shape={sampled_points.shape}')
        if device == 'cuda':
            sampled_points = sampled_points.cpu()
        data_dict['points'] = sampled_points.numpy() # sampled_points.cpu().numpy()
        return data_dict

    def calculate_grid_size(self, data_dict=None, config=None):
        if data_dict is None:
            grid_size = (self.point_cloud_range[3:6] - self.point_cloud_range[0:3]) / np.array(config.VOXEL_SIZE)
            self.grid_size = np.round(grid_size).astype(np.int64)
            self.voxel_size = config.VOXEL_SIZE
            return partial(self.calculate_grid_size, config=config)
        return data_dict

    def downsample_depth_map(self, data_dict=None, config=None):
        if data_dict is None:
            self.depth_downsample_factor = config.DOWNSAMPLE_FACTOR
            return partial(self.downsample_depth_map, config=config)

        data_dict['depth_maps'] = transform.downscale_local_mean(
            image=data_dict['depth_maps'],
            factors=(self.depth_downsample_factor, self.depth_downsample_factor)
        )
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
                gt_names: optional, (N), string
                ...

        Returns:
        """

        for cur_processor in self.data_processor_queue:
            data_dict = cur_processor(data_dict=data_dict)

        return data_dict
