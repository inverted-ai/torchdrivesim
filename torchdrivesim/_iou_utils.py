"""
torch implementation of 2d oriented box intersection

author: Lanxiao Li
copied from https://github.com/lilanxiao/Rotated_IoU
2020.8

MIT License

Copyright (c) 2020 Lanxiao Li

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
from typing import Tuple

import numpy as np
import torch

EPSILON = 1e-8


def precision_rounding(x, n_digits=6):
    return (x * 10**n_digits).round() / (10**n_digits)


def box_intersection_th(corners1: torch.Tensor, corners2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """find intersection points of rectangles.
    Convention: if two edges are collinear, there is no intersection point.

    Args:
        corners1: B, N, 4, 2
        corners2: B, N, 4, 2

    Returns:
        A tuple (intersectons, mask) where:
            intersections: B, N, 4, 4, 2
            mask: B, N, 4, 4; bool
    """
    # build edges from corners
    line1 = torch.cat([corners1, corners1[:, :, [1, 2, 3, 0], :]], dim=3) # B, N, 4, 4: Batch, Box, edge, point
    line2 = torch.cat([corners2, corners2[:, :, [1, 2, 3, 0], :]], dim=3)
    # duplicate data to pair each edges from the boxes
    # (B, N, 4, 4) -> (B, N, 4, 4, 4) : Batch, Box, edge1, edge2, point
    line1_ext = line1.unsqueeze(3).repeat([1,1,1,4,1])
    line2_ext = line2.unsqueeze(2).repeat([1,1,4,1,1])
    x1 = line1_ext[..., 0]
    y1 = line1_ext[..., 1]
    x2 = line1_ext[..., 2]
    y2 = line1_ext[..., 3]
    x3 = line2_ext[..., 0]
    y3 = line2_ext[..., 1]
    x4 = line2_ext[..., 2]
    y4 = line2_ext[..., 3]
    # math: https://en.wikipedia.org/wiki/Line%E2%80%93line_intersection
    num = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    den_t = (x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)
    t = den_t / num
    t[torch.abs(num) < 1e-4] = -1.
    mask_t = (t > 0) * (t < 1)                # intersection on line segment 1
    den_u = (x1-x2)*(y1-y3) - (y1-y2)*(x1-x3)
    u = -den_u / num
    u[torch.abs(num) < 1e-4] = -1.
    mask_u = (u > 0) * (u < 1)                # intersection on line segment 2
    mask = mask_t * mask_u
    t = den_t / (num + EPSILON)                 # overwrite with EPSILON. otherwise numerically unstable
    intersections = torch.stack([x1 + t*(x2-x1), y1 + t*(y2-y1)], dim=-1)
    intersections = intersections * mask.float().unsqueeze(-1)
    return intersections, mask


def box1_in_box2(corners1: torch.Tensor, corners2: torch.Tensor) -> torch.Tensor:
    """check if corners of box1 lie in box2
    Convention: if a corner is exactly on the edge of the other box, it's also a valid point

    Args:
        corners1: (B, N, 4, 2)
        corners2: (B, N, 4, 2)

    Returns:
        c1_in_2: (B, N, 4) Bool
    """
    a = corners2[:, :, 0:1, :]  # (B, N, 1, 2)
    b = corners2[:, :, 1:2, :]  # (B, N, 1, 2)
    d = corners2[:, :, 3:4, :]  # (B, N, 1, 2)
    ab = b - a                  # (B, N, 1, 2)
    am = corners1 - a           # (B, N, 4, 2)
    ad = d - a                  # (B, N, 1, 2)
    p_ab = torch.sum(ab * am, dim=-1)       # (B, N, 4)
    norm_ab = torch.sum(ab * ab, dim=-1)    # (B, N, 1)
    p_ad = torch.sum(ad * am, dim=-1)       # (B, N, 4)
    norm_ad = torch.sum(ad * ad, dim=-1)    # (B, N, 1)
    # NOTE: the expression looks ugly but is stable if the two boxes are exactly the same
    # also stable with different scale of bboxes
    cond1 = precision_rounding(p_ab / norm_ab)
    cond1 = (cond1 > - 1e-6) * (cond1 < 1 + 1e-6)   # (B, N, 4)
    cond2 = precision_rounding(p_ad / norm_ad)
    cond2 = (cond2 > - 1e-6) * (cond2 < 1 + 1e-6)   # (B, N, 4)
    return cond1*cond2


def box_in_box_th(corners1:torch.Tensor, corners2:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """check if corners of two boxes lie in each other

    Args:
        corners1: (B, N, 4, 2)
        corners2: (B, N, 4, 2)

    Returns:
        A tuple (c1_in_2, c2_in_1) where:
            c1_in_2: (B, N, 4) Bool. i-th corner of box1 in box2
            c2_in_1: (B, N, 4) Bool. i-th corner of box2 in box1
    """
    c1_in_2 = box1_in_box2(corners1, corners2)
    c2_in_1 = box1_in_box2(corners2, corners1)
    return c1_in_2, c2_in_1


def build_vertices(corners1: torch.Tensor, corners2: torch.Tensor, c1_in_2: torch.Tensor, c2_in_1: torch.Tensor,
                   inters: torch.Tensor, mask_inter: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """find vertices of intersection area

    Args:
        corners1: (B, N, 4, 2)
        corners2: (B, N, 4, 2)
        c1_in_2: Bool, (B, N, 4)
        c2_in_1: Bool, (B, N, 4)
        inters: (B, N, 4, 4, 2)
        mask_inter: (B, N, 4, 4)

    Returns:
        A tuple (vertices, mask) where:
            vertices: (B, N, 24, 2) vertices of intersection area. only some elements are valid
            mask: (B, N, 24) indicates valid elements in vertices
    """
    # NOTE: inter has elements equals zero and has zeros gradient (masked by multiplying with 0).
    # can be used as trick
    B = corners1.size()[0]
    N = corners1.size()[1]
    vertices = torch.cat([corners1, corners2, inters.view([B, N, -1, 2])], dim=2) # (B, N, 4+4+16, 2)
    mask = torch.cat([c1_in_2, c2_in_1, mask_inter.view([B, N, -1])], dim=2) # Bool (B, N, 4+4+16)
    return vertices, mask


def sort_indices(vertices: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """

    Args:
        vertices: float (B, N, 24, 2)
        mask: bool (B, N, 24)

    Returns:
        bool (B, N, 9)

    Note:
        why 9? the polygon has maximal 8 vertices. +1 to duplicate the first element.
        the index should have following structure:
            (A, B, C, ... , A, X, X, X)
        and X indicates the index of arbitary elements in the last 16 (intersections not corners) with
        value 0 and mask False. (cause they have zero value and zero gradient)
    """
    with torch.no_grad():
        B, N = vertices.shape[:2]
        num_valid = torch.sum(mask.int(), dim=2).int()      # (B, N)
        center = torch.sum(vertices * mask.float().unsqueeze(-1), dim=2, keepdim=True) / num_valid.unsqueeze(-1).unsqueeze(-1)
        r = torch.sqrt((vertices - center).pow(2).sum(dim=-1))
        angles = torch.where((vertices[...,1] - center[...,1])>0,
                         torch.arccos((vertices[...,0] - center[...,0])/r),
                         2*np.pi-torch.arccos((vertices[...,0] - center[...,0])/r))
        inds_sorted = torch.argsort(angles, descending=False)
        inds_sorted = torch.masked_select(inds_sorted, torch.gather(mask, 2, inds_sorted))
        num_valid = num_valid.view(-1)
        mask = mask.view(-1, 24)
        index = torch.ones(B*N, 9, dtype=torch.long, device=inds_sorted.device)*8
        curr_ind = 0
        pad_values = torch.argmin(mask[:, 8:].float(), dim=-1) + 8
        for b in range(B*N):
            b_num_valid = num_valid[b].item()
            if b_num_valid < 3:
                index[b] = pad_values[b].item()
            elif b_num_valid > 0:
                index[b, :b_num_valid] = inds_sorted[curr_ind:curr_ind+b_num_valid]
                index[b, b_num_valid] = inds_sorted[curr_ind]
                index[b, b_num_valid+1:] = pad_values[b].item()
            curr_ind += b_num_valid
        index = index.view(B,N,9)
    return index


def calculate_area(idx_sorted: torch.Tensor, vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """calculate area of intersection

    Args:
        idx_sorted (torch.Tensor): (B, N, 9)
        vertices (torch.Tensor): (B, N, 24, 2)

    Returns:
        Tuple of (area, selected), where
            area: (B, N), area of intersection
            selected: (B, N, 9, 2), vertices of polygon with zero padding
    """
    idx_ext = idx_sorted.unsqueeze(-1).repeat([1,1,1,2])
    selected = torch.gather(vertices, 2, idx_ext)
    total = selected[:, :, 0:-1, 0]*selected[:, :, 1:, 1] - selected[:, :, 0:-1, 1]*selected[:, :, 1:, 0]
    total = torch.sum(total, dim=2)
    area = torch.abs(total) / 2
    return area, selected


def oriented_box_intersection_2d(corners1: torch.Tensor, corners2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Calculates the intersection area of 2D rectangles.

    Args:
        corners1: (B, N, 4, 2)
        corners2: (B, N, 4, 2)

    Returns:
        Tuple of (area, selected):
            area: (B, N), area of intersection
            selected: (B, N, 9, 2), vertices of polygon with zero padding
    """

    inters, mask_inter = box_intersection_th(corners1, corners2)
    c12, c21 = box_in_box_th(corners1, corners2)
    vertices, mask = build_vertices(corners1, corners2, c12, c21, inters, mask_inter)
    sorted_indices = sort_indices(vertices, mask)
    return calculate_area(sorted_indices, vertices)


def box2corners_th(box: torch.Tensor) -> torch.Tensor:
    """convert box coordinate to corners

    Args:
        box: (B, N, 5) with x, y, w, h, alpha

    Returns:
        (B, N, 4, 2) corners
    """
    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w     # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    rotated[..., 0] += x
    rotated[..., 1] += y
    return rotated


def box2corners_with_rear_factor(box: torch.Tensor, rear_factor: float = 1) -> torch.Tensor:
    """Convert bounding box coordinates to corners. The returned corners are starting from the rear of the bounding box
    up to a rear factor. If the rear factor is 1, then the whole bounding box is returned.

    Args:
        box: (B, N, 5) with x, y, w, h, alpha
        rear_factor: The relative amount of the bounding box will be preserved starting from the rear corners and up to
            rear_factor * w. If the factor is 1, then the whole bounding box is preserved.

    Returns:
        (B, N, 4, 2) corners
    """

    B = box.size()[0]
    x = box[..., 0:1]
    y = box[..., 1:2]
    w = box[..., 2:3]
    h = box[..., 3:4]
    alpha = box[..., 4:5] # (B, N, 1)
    x4 = torch.FloatTensor([0.5, -0.5, -0.5, 0.5]).unsqueeze(0).unsqueeze(0).to(box.device) # (1,1,4)
    x4 = x4 * w  * rear_factor    # (B, N, 4)
    y4 = torch.FloatTensor([0.5, 0.5, -0.5, -0.5]).unsqueeze(0).unsqueeze(0).to(box.device)
    y4 = y4 * h     # (B, N, 4)
    corners = torch.stack([x4, y4], dim=-1)     # (B, N, 4, 2)
    sin = torch.sin(alpha)
    cos = torch.cos(alpha)
    row1 = torch.cat([cos, sin], dim=-1)
    row2 = torch.cat([-sin, cos], dim=-1)       # (B, N, 2)
    rot_T = torch.stack([row1, row2], dim=-2)   # (B, N, 2, 2)
    rotated = torch.bmm(corners.view([-1,4,2]), rot_T.view([-1,2,2]))
    rotated = rotated.view([B,-1,4,2])          # (B*N, 4, 2) -> (B, N, 4, 2)
    center_correction = torch.cat([
        (w * (1 - rear_factor))/2,
        torch.zeros_like(h)
    ], dim=-1)
    center_correction = torch.bmm(center_correction.view([-1,1,2]), rot_T.view([-1,2,2]))
    center_correction = center_correction.view([B,-1,2])
    rotated[..., 0] += x - center_correction[..., 0:1]
    rotated[..., 1] += y - center_correction[..., 1:2]
    return rotated


def iou_differentiable_fast(box1:torch.Tensor, box2:torch.Tensor) -> torch.Tensor:
    """Calculates the differentiable (approx.) IOU between two sets of bounding boxes.
    B is the batch size and N is the number of bounding boxes. A bounding box is
    described by 5 values in this order: (x, y, w, h, alpha). This method uses the
    shoelace formula (https://en.wikipedia.org/wiki/Shoelace_formula) to calculate
    the area of a convex polygon (the overlapping area of two rectangles).
    It offers fast computation at the expense of some IoU accuracy.

    Args:
        box1: (B, N, 5)
        box2: (B, N, 5)

    Returns:
        iou: (B, N)
    """

    corners1 = box2corners_th(box1)
    corners2 = box2corners_th(box2)
    inter_area, _ = oriented_box_intersection_2d(corners1, corners2)        #(B, N)
    area1 = box1[:, :, 2] * box1[:, :, 3]
    area2 = box2[:, :, 2] * box2[:, :, 3]
    u = area1 + area2 - inter_area
    iou = inter_area / u
    return iou


def iou_non_differentiable(boxes:torch.Tensor) -> torch.Tensor:
    """
    Args:
        boxes: (N, 5)

    Returns:
        iou: (N, N)
    """

    from pytorch3d.ops import box3d_overlap
    corners = box2corners_th(boxes.unsqueeze(0)).squeeze(0)

    corners_3d = torch.nn.functional.pad(corners, (0, 1))
    corners_3d = torch.cat([corners_3d, corners_3d], dim=1)

    corners_3d[:, 4:, 2] = 1.0

    _, iou = box3d_overlap(corners_3d, corners_3d)
    return iou
