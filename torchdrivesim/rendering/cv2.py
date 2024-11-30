from dataclasses import dataclass

import cv2
import numpy as np
import torch

from torchdrivesim.mesh import BirdviewMesh, RGBMesh, tensor_color
from torchdrivesim.rendering import BirdviewRenderer, RendererConfig
from torchdrivesim.rendering.base import Cameras
from torchdrivesim.utils import Resolution


@dataclass
class CV2RendererConfig(RendererConfig):
    backend: str = 'cv2'
    trim_mesh_before_rendering: bool = True


class CV2Renderer(BirdviewRenderer):
    """
    Renderer based on OpenCV. Slow, but easy to install. Renders on CPU.
    """
    def __init__(self, cfg: CV2RendererConfig, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.cfg: CV2RendererConfig = cfg

    def render_rgb_mesh(self, mesh: RGBMesh, res: Resolution, cameras: Cameras) -> torch.Tensor:
        if self.cfg.shift_mesh_by_camera_before_rendering:
            mesh = mesh.translate(-cameras.xy)
            cameras = Cameras(xy=torch.zeros_like(cameras.xy), sc=cameras.sc, scale=cameras.scale)
        if self.cfg.trim_mesh_before_rendering:
            # For efficiency, remove faces that are not visible anyway
            viewing_polygon = cameras.reverse_transform_points_screen(
                torch.tensor([
                    [0, 0], [0, res.height], [res.width, res.height], [res.width, 0]
                ], device=mesh.device)[None].repeat_interleave(cameras.xy.shape[0], dim=0), res=res
            )
            viewing_polygon_center = viewing_polygon.mean(dim=1, keepdim=True)
            viewing_polygon = viewing_polygon_center + (viewing_polygon - viewing_polygon_center) * 1.05  # safety margin
            mesh = mesh.trim(viewing_polygon)

        #  We assume all vertices of the same face are on the same plane
        rendering_order = mesh.verts[:, None, :, 2].expand(-1, mesh.faces.shape[1], -1).flatten(0,1)
        rendering_order = torch.gather(rendering_order, dim=1,
            index=mesh.faces.flatten(0,1))[..., 0].reshape(*mesh.faces.shape[:2])
        rendering_order = rendering_order.argsort(dim=1, descending=True).unsqueeze(-1).cpu()
        pixel_verts = cameras.transform_points_screen(mesh.verts[..., :2], res=res).cpu().to(torch.int32)
        mesh_faces = mesh.faces.cpu().gather(dim=1, index=rendering_order.expand_as(mesh.faces))
        color_attrs = (mesh.attrs * (1.0 - 1e-3) * 256).floor().to(torch.uint8).cpu()

        image_batch = []
        for batch_idx in range(mesh.batch_size):
            image = np.zeros((res.height, res.width, 3), dtype=np.float32)
            faces = mesh_faces[batch_idx]
            for face_idx in range(faces.shape[0]):
                polygon = pixel_verts[batch_idx, faces[face_idx]].numpy()
                color = color_attrs[batch_idx, face_idx].numpy().tolist()
                image = cv2.fillConvexPoly(img=image, points=polygon, color=color, shift=0, lineType=cv2.LINE_AA)
            image = torch.from_numpy(image)
            image = image.transpose(-2, -3)  # point x upwards, flip to right-handed coordinate frame

            if self.cfg.left_handed_coordinates:
                image = image.flip(dims=(-2,))  # flip horizontally

            image_batch.append(image)
        images = torch.stack(image_batch, dim=0).to(mesh.device)
        return images
