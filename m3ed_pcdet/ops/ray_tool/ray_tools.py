import torch
from torch.utils.cpp_extension import _import_module_from_library
from pathlib import Path

voxelizer = _import_module_from_library(
    "voxelizer",
    path=Path(__file__).parent,
    is_python_module=True
)

raymaxer = _import_module_from_library(
    "raymaxer",
    path=Path(__file__).parent,
    is_python_module=True
)

class Ray_Tool:
    def __init__(self, pc_range, voxel_size):
        super(Ray_Tool, self).__init__()
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.n_height = int((self.pc_range[5] - self.pc_range[2]) / self.voxel_size[2])
        self.n_length = int((self.pc_range[4] - self.pc_range[1]) / self.voxel_size[1])
        self.n_width = int((self.pc_range[3] - self.pc_range[0]) / self.voxel_size[0])
        self.input_grid = [1, self.n_height, self.n_length, self.n_width]
        self.output_grid = [1, self.n_length, self.n_width]

        self.offset = torch.nn.parameter.Parameter(
            torch.Tensor(self.pc_range[:3])[None, None, :], requires_grad=False
        ).cuda()
        self.scaler = torch.nn.parameter.Parameter(
            torch.Tensor(self.voxel_size)[None, None, :], requires_grad=False
        ).cuda()

    def data2tensor(self, data):
        if not isinstance(data, torch.Tensor):
            data = torch.from_numpy(data)
        return data.to(torch.device('cuda'))

    def voxelizer(self, input_points):
        self._normalize(input_points)
        input_tensor = voxelizer.voxelize(input_points, self.input_grid)
        input_tensor = input_tensor.reshape(
            (-1, self.n_height, self.n_length, self.n_width)
        )
        return input_tensor
    
    def _normalize(self, points):
        points[:, :, :3] = (points[:, :, :3] - self.offset) / self.scaler

    def raymaxer(self, points):
        points = points.reshape(1,-1,4)
        output_points = self.data2tensor(points)
        voxel = self.voxelizer(output_points)
        sigma = voxel[0,2,...] + voxel[0,3,...]
        output_origins = output_points.new_zeros([1,1,3])
        self._normalize(output_origins)
        # self._normalize(output_points)
        sigma = sigma[None,None,:,:]

        argmax_yy, argmax_xx = raymaxer.argmax(sigma, output_origins, output_points)
        argmax_yy = argmax_yy.long()
        argmax_xx = argmax_xx.long()

        ii = torch.arange(len(output_origins))
        tt = torch.arange(1)

        nvf_logits = sigma[
            ii[:, None, None, None], tt[None, :, None, None], argmax_yy, argmax_xx
        ]
        nvf_logits = torch.where(nvf_logits>0, 1, 0)
        nvf_logits = nvf_logits.squeeze().permute(1,0).detach().cpu().numpy()

        return nvf_logits