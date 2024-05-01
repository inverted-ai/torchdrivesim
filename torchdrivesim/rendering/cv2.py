from dataclasses import dataclass

import cv2
import numpy as np
import torch

from torchdrivesim.mesh import BirdviewMesh, tensor_color
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

    def render_mesh(self, mesh: BirdviewMesh, res: Resolution, cameras: Cameras) -> torch.Tensor:

        if self.cfg.trim_mesh_before_rendering:
            # For efficiency, remove faces that are not visible anyway
            viewing_polygon = cameras.reverse_transform_points_screen(
                torch.tensor([
                    [0, 0], [0, res.height], [res.width, res.height], [res.width, 0]
                ], device=mesh.device), res=res
            )
            viewing_polygon_center = viewing_polygon.mean(dim=-2)
            viewing_polygon = viewing_polygon_center + (viewing_polygon - viewing_polygon_center) * 1.05  # safety margin
            mesh = mesh.trim(viewing_polygon)

        image_batch = []
        pixel_verts = cameras.transform_points_screen(mesh.verts, res=res).cpu().to(torch.int32)

        for k in mesh.categories:
            if k not in mesh.colors:
                mesh.colors[k] = tensor_color(self.color_map[k])
            if k not in mesh.zs:
                mesh.zs[k] = self.rendering_levels[k]
        rendering_order = sorted(range(len(mesh.categories)), key=lambda i: mesh.zs[mesh.categories[i]], reverse=True)
        category_colors = (
                torch.stack([mesh.colors[cat] for cat in mesh.categories], dim=0) * (1.0 - 1e-3) * 256
        ).floor().to(torch.uint8).cpu()

        for batch_idx in range(mesh.batch_size):
            image = np.zeros((res.height, res.width, 3), dtype=np.float32)
            for cat_idx in rendering_order:
                color = category_colors[cat_idx].numpy().tolist()
                face_category = mesh.vert_category[batch_idx, mesh.faces[batch_idx, :, 0]]
                faces = mesh.faces[batch_idx][face_category == cat_idx].cpu()
                for face in faces:
                    polygon = pixel_verts[batch_idx, face].numpy()
                    image = cv2.fillConvexPoly(img=image, points=polygon, color=color, shift=0, lineType=cv2.LINE_AA)
            image = torch.from_numpy(image)
            image = image.transpose(-2, -3)  # point x upwards, flip to right-handed coordinate frame

            if self.cfg.left_handed_coordinates:
                image = image.flip(dims=(-2,))  # flip horizontally

            image_batch.append(image)
        images = torch.stack(image_batch, dim=0).to(mesh.device)
        return images
