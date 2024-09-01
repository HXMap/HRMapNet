import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
from mmcv.runner import get_dist_info
import torch.nn.functional as F

def gen_matrix(ego2global_rotation, ego2global_translation):
    rotation_xyz = np.roll(ego2global_rotation, shift=-1)
    trans = np.eye(4)
    trans[:3, 3] = ego2global_translation
    trans[:3, :3] = R.from_quat(rotation_xyz).as_matrix()
    return trans

def get_bev_coords(bev_bound_w, bev_bound_h, bev_w, bev_h):
    """
    Args:
        bev_bound_w (tuple:2):
        bev_bound_h (tuple:2):
        bev_w (int):
        bev_h (int):
    Returns: (bev_h, bev_w, 4)

    """
    sample_coords = torch.stack(torch.meshgrid(
        torch.linspace(bev_bound_w[0], bev_bound_w[1], int(bev_w), dtype=torch.float32),
        torch.linspace(bev_bound_h[0], bev_bound_h[1], int(bev_h), dtype=torch.float32)
    ), axis=2).transpose(1, 0)
    assert sample_coords.shape[0] == bev_h, sample_coords.shape[1] == bev_w
    zeros = torch.zeros((bev_h, bev_w, 1), dtype=sample_coords.dtype)
    ones = torch.ones((bev_h, bev_w, 1), dtype=sample_coords.dtype)
    sample_coords = torch.cat([sample_coords, zeros, ones], dim=-1)
    return sample_coords


class GlobalMap:
    def __init__(self, map_cfg):
        self.map_type = torch.uint8
        self.fuse_method_val = map_cfg['fuse_method']
        self.fuse_method = map_cfg['fuse_method']
        self.bev_h = map_cfg['bev_h']
        self.bev_w = map_cfg['bev_w']
        self.pc_range = map_cfg['pc_range']
        self.load_map_path = map_cfg['load_map_path']
        self.save_map_path = map_cfg['save_map_path']
        self.bev_patch_h = self.pc_range[4] - self.pc_range[1]
        self.bev_patch_w = self.pc_range[3] - self.pc_range[0]
        bev_radius = np.sqrt(self.bev_patch_h ** 2 + self.bev_patch_w ** 2) / 2
        dataset = map_cfg['dataset']
        if dataset == 'av2':
            self.city_list = ['WDC', 'MIA', 'PAO', 'PIT', 'ATX', 'DTW']
            bev_radius = bev_radius * 5
            self.train_min_lidar_loc = {'WDC': np.array([2327.78751629, 25.76974403]) - bev_radius,
                                        'MIA': np.array([-1086.92985063, -464.15366362]) - bev_radius,
                                        'PAO': np.array([-2225.36229607, -309.44287914]) - bev_radius,
                                        'PIT': np.array([695.66044791, -443.89844576]) - bev_radius,
                                        'ATX': np.array([589.98724063, -2444.36667873]) - bev_radius,
                                        'DTW': np.array([-6111.0784155, 628.12019426]) - bev_radius}
            self.train_max_lidar_loc = {'WDC': np.array([6951.32050819, 4510.96637507]) + bev_radius,
                                        'MIA': np.array([6817.02338386, 4301.35442342]) + bev_radius,
                                        'PAO': np.array([1646.099298, 2371.23617712]) + bev_radius,
                                        'PIT': np.array([7371.45409948, 3314.83461676]) + bev_radius,
                                        'ATX': np.array([3923.01840213, -1161.67712224]) + bev_radius,
                                        'DTW': np.array([11126.80825267, 6045.01530619]) + bev_radius}
            self.val_min_lidar_loc = {'WDC': np.array([1664.20793519, 344.29333819]) - bev_radius,
                                      'MIA': np.array([-885.96340492, 257.79835061]) - bev_radius,
                                      'PAO': np.array([-3050.01628955, -18.25448306]) - bev_radius,
                                      'PIT': np.array([715.98981458, -136.13570664]) - bev_radius,
                                      'ATX': np.array([840.66655697, -2581.61138577]) - bev_radius,
                                      'DTW': np.array([36.60503836, 2432.04117045]) - bev_radius}
            self.val_max_lidar_loc = {'WDC': np.array([6383.48765357, 4320.74293797]) + bev_radius,
                                      'MIA': np.array([6708.79270643, 4295.23306249]) + bev_radius,
                                      'PAO': np.array([654.02351246, 2988.66862304]) + bev_radius,
                                      'PIT': np.array([7445.46486881, 3160.2406237]) + bev_radius,
                                      'ATX': np.array([3726.62166299, -1296.12914951]) + bev_radius,
                                      'DTW': np.array([10896.30840694, 6215.31771939]) + bev_radius}
        else:
            self.city_list = ['singapore-onenorth', 'boston-seaport',
                              'singapore-queenstown', 'singapore-hollandvillage']
            self.train_min_lidar_loc = {'singapore-onenorth': np.array([118., 419.]) - bev_radius,
                                      'boston-seaport': np.array([298., 328.]) - bev_radius,
                                      'singapore-queenstown': np.array([347., 862.]) - bev_radius,
                                      'singapore-hollandvillage': np.array([442., 902.]) - bev_radius}
            self.train_max_lidar_loc = {'singapore-onenorth': np.array([1232., 1777.]) + bev_radius,
                                      'boston-seaport': np.array([2527., 1896.]) + bev_radius,
                                      'singapore-queenstown': np.array([2685., 3298.]) + bev_radius,
                                      'singapore-hollandvillage': np.array([2490., 2839.]) + bev_radius}
            self.val_min_lidar_loc = {'singapore-onenorth': np.array([118., 409.]) - bev_radius,
                                    'boston-seaport': np.array([411., 554.]) - bev_radius,
                                    'singapore-queenstown': np.array([524., 870.]) - bev_radius,
                                    'singapore-hollandvillage': np.array([608., 2006.]) - bev_radius}
            self.val_max_lidar_loc = {'singapore-onenorth': np.array([1232., 1732.]) + bev_radius,
                                    'boston-seaport': np.array([2368., 1720.]) + bev_radius,
                                    'singapore-queenstown': np.array([2043., 3334.]) + bev_radius,
                                    'singapore-hollandvillage': np.array([2460., 2836.]) + bev_radius}
            self.mix_min_lidar_loc = {'singapore-onenorth': np.array([118., 409.]) - bev_radius,
                                      'boston-seaport': np.array([298., 328.]) - bev_radius,
                                      'singapore-queenstown': np.array([347., 862.]) - bev_radius,
                                      'singapore-hollandvillage': np.array([442., 902.]) - bev_radius}
            self.mix_max_lidar_loc = {'singapore-onenorth': np.array([1232., 1777.]) + bev_radius,
                                      'boston-seaport': np.array([2527., 1896.]) + bev_radius,
                                      'singapore-queenstown': np.array([2685., 3334.]) + bev_radius,
                                      'singapore-hollandvillage': np.array([2490., 2839.]) + bev_radius}

        bev_bound_h, bev_bound_w = \
            [(-row[0] / 2 + row[0] / row[1] / 2, row[0] / 2 - row[0] / row[1] / 2)
             for row in ((self.bev_patch_h, self.bev_h), (self.bev_patch_w, self.bev_w))]
        self.bev_grid_len_h = self.bev_patch_h / self.bev_h
        self.bev_grid_len_w = self.bev_patch_w / self.bev_w
        self.bev_coords = get_bev_coords(bev_bound_w, bev_bound_h, self.bev_w, self.bev_h)
        self.bev_coords = self.bev_coords.reshape(-1, 4).permute(1, 0)

        self.global_map_raster_size = map_cfg['raster_size']
        self.global_map_dict = {}

        self.map_status = None
        self.epoch_point = -2
        self.update_value = 30
        self.use_mix = self.load_map_path is not None or self.save_map_path is not None

    def load_map(self, device):
        self.global_map_dict = torch.load(self.load_map_path, map_location=device)

    def check_map(self, device, epoch, status):
        if status == 'train':
            self.fuse_method = 'all'   # To keep consistent with our initial setting
        else:
            self.fuse_method = self.fuse_method_val
        if self.map_status is None:
            self.epoch_point = epoch
            self.map_status = status
            if self.load_map_path is not None:
                self.load_map(device)
            else:
                self.create_map(device, status)
        elif status != self.map_status:
            self.epoch_point = epoch
            self.map_status = status
            self.create_map(device, status)
        elif epoch != self.epoch_point:
            self.epoch_point = epoch
            self.map_status = status
            self.reset_map()

    def reset_map(self):
        for city_name in self.city_list:
            self.global_map_dict[city_name].zero_()
            print("reset map", city_name, "for epoch", self.epoch_point, "status", self.map_status)
            if self.fuse_method == 'prob':
                self.global_map_dict[city_name] = self.global_map_dict[city_name] + 100

    def get_city_bound(self, city_name, status):
        if self.use_mix:
            return self.mix_min_lidar_loc[city_name], self.mix_max_lidar_loc[city_name]

        if status == 'train':
            return self.train_min_lidar_loc[city_name], self.train_max_lidar_loc[city_name]
        elif status == 'val':
            return self.val_min_lidar_loc[city_name], self.val_max_lidar_loc[city_name]
        elif status == 'mix':
            return self.mix_min_lidar_loc[city_name], self.mix_max_lidar_loc[city_name]

    def create_map(self, device, status):
        for city_name in self.city_list:
            city_min_bound, city_max_bound = self.get_city_bound(city_name, status)
            city_grid_size = (city_max_bound - city_min_bound) / np.array(self.global_map_raster_size, np.float32)
            map_height = city_grid_size[0]
            map_width = city_grid_size[1]
            map_height_ceil = int(np.ceil(map_height))
            map_width_ceil = int(np.ceil(map_width))
            city_map = torch.zeros((map_height_ceil, map_width_ceil, 3), dtype=self.map_type, device=device)
            if self.fuse_method == 'prob':
                city_map = city_map + 100
            print("create map", city_name, status, "on", device, "for epoch", self.epoch_point, "map: ", map_height_ceil, "*", map_width_ceil)
            self.global_map_dict[city_name] = city_map
        self.map_status = status

    def update_map(self, city_name, trans, raster, status):
        trans = self.bev_coords.new_tensor(trans)
        trans_bev_coords = trans @ self.bev_coords
        bev_coord_w = trans_bev_coords[0, :]
        bev_coord_h = trans_bev_coords[1, :]
        city_min_bound, city_max_bound = self.get_city_bound(city_name, status)

        bev_index_w = torch.floor((bev_coord_w - city_min_bound[0]) / self.bev_grid_len_w).to(torch.int64)
        bev_index_h = torch.floor((bev_coord_h - city_min_bound[1]) / self.bev_grid_len_w).to(torch.int64)
        bev_index_w = bev_index_w.reshape(self.bev_h, self.bev_w)
        bev_index_h = bev_index_h.reshape(self.bev_h, self.bev_w)
        bev_coord_mask = \
            (city_min_bound[0] <= bev_coord_w) & (bev_coord_w < city_max_bound[0]) & \
            (city_min_bound[1] <= bev_coord_h) & (bev_coord_h < city_max_bound[1])
        bev_coord_mask = bev_coord_mask.reshape(self.bev_h, self.bev_w)
        index_h, index_w = torch.where(bev_coord_mask)
        new_map = raster[index_h, index_w, :]
        old_map = self.global_map_dict[city_name][bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :]
        update_map = self.fuse_map(new_map, old_map)
        self.global_map_dict[city_name][bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :] = update_map

    def get_map(self, city_name, trans, status):
        trans = self.bev_coords.new_tensor(trans)
        trans_bev_coords = trans @ self.bev_coords
        bev_coord_w = trans_bev_coords[0, :]
        bev_coord_h = trans_bev_coords[1, :]
        city_min_bound, city_max_bound = self.get_city_bound(city_name, status)

        bev_index_w = torch.floor((bev_coord_w - city_min_bound[0]) / self.bev_grid_len_w).to(torch.int64)
        bev_index_h = torch.floor((bev_coord_h - city_min_bound[1]) / self.bev_grid_len_h).to(torch.int64)
        bev_index_w = bev_index_w.reshape(self.bev_h, self.bev_w)
        bev_index_h = bev_index_h.reshape(self.bev_h, self.bev_w)
        bev_coord_mask = \
            (city_min_bound[0] <= bev_coord_w) & (bev_coord_w < city_max_bound[0]) & \
            (city_min_bound[1] <= bev_coord_h) & (bev_coord_h < city_max_bound[1])
        bev_coord_mask = bev_coord_mask.reshape(self.bev_h, self.bev_w)
        index_h, index_w = torch.where(bev_coord_mask)
        local_map = self.global_map_dict[city_name][bev_index_w[index_h, index_w], bev_index_h[index_h, index_w], :]
        if self.map_type == torch.uint8:
            local_map_float = local_map.float()
            if self.fuse_method == 'prob':
                local_map_float = (local_map_float - 100.0) / (self.update_value)
                local_map_float = torch.clamp(local_map_float, 0.0, 1.0)
            else:
                local_map_float = local_map_float / 255.0
            return local_map_float
        else:
            return local_map

    def fuse_map(self, new_map, old_map):
        if self.fuse_method == 'all':
            if self.map_type == torch.uint8:
                new_map = (new_map * 255).to(torch.uint8)
            return torch.max(new_map, old_map)
        elif self.fuse_method == 'prob':
            new_map = new_map * self.update_value + 100
            update_map = torch.max(new_map, old_map)
            new_map[new_map > 100] = 0
            new_map[new_map > 1] = -1
            update_map = update_map + new_map
            update_map = torch.clamp(update_map, 2, 240)
            update_map = update_map.to(self.map_type)
            return update_map

    def get_global_map(self):
        return self.global_map_dict

    def save_global_map(self):
        if self.save_map_path is not None:
            torch.save(self.global_map_dict, self.save_map_path)
            print("Save constructed map at", self.save_map_path, "!!!!!")