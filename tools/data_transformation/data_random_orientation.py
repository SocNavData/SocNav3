import math
import copy
import random
import torch
from data_conversions import clone_sequence

def tensor_transform_with_random_orientation(tDict_sequence):
    if tDict_sequence['goal']['th_a'][0] < math.pi:
        return tDict_sequence

    a = random.uniform(-math.pi, math.pi)
    angle = torch.full(tDict_sequence['goal']['a'].shape, a, dtype=torch.float64)

    transformed_sequence = clone_sequence(tDict_sequence)


    sin_theta = torch.sin(angle)
    cos_theta = torch.cos(angle)

    rx = transformed_sequence['robot']['x']
    ry = transformed_sequence['robot']['y']
    ra = transformed_sequence['robot']['a']
    vx = transformed_sequence['robot']['vx']
    vy = transformed_sequence['robot']['vy']
    va = transformed_sequence['robot']['va']
    new_rx = rx * cos_theta - ry * sin_theta
    new_ry = rx * sin_theta + ry * cos_theta
    new_ra = torch.arctan2(torch.sin(ra + angle), torch.cos(ra + angle))
    new_vx = vx * cos_theta - vy * sin_theta
    new_vy = vx * sin_theta + vy * cos_theta

    transformed_sequence['robot']['x'] = new_rx
    transformed_sequence['robot']['y'] = new_ry
    transformed_sequence['robot']['a'] = new_ra
    transformed_sequence['robot']['vx'] = new_vx
    transformed_sequence['robot']['vy'] = new_vy

    if torch.numel(transformed_sequence['people']['exists'])>0:
        px = transformed_sequence['people']['x']
        py = transformed_sequence['people']['y']
        pa = transformed_sequence['people']['a']
        angle2d = torch.repeat_interleave(torch.unsqueeze(angle, 1), px.shape[1], dim=1)
        sin_theta2d = torch.repeat_interleave(torch.unsqueeze(sin_theta, 1), px.shape[1], dim=1)
        cos_theta2d = torch.repeat_interleave(torch.unsqueeze(cos_theta, 1), px.shape[1], dim=1)
        new_px = px * cos_theta2d - py * sin_theta2d
        new_py = px * sin_theta2d + py * cos_theta2d
        new_pa = torch.arctan2(torch.sin(pa + angle2d), torch.cos(pa + angle2d))

        transformed_sequence['people']['x'] = new_px
        transformed_sequence['people']['y'] = new_py
        transformed_sequence['people']['a'] = new_pa

    if torch.numel(transformed_sequence['objects']['exists'])>0:
        ox = transformed_sequence['objects']['x']
        oy = transformed_sequence['objects']['y']
        oa = transformed_sequence['objects']['a']
        angle2d = torch.repeat_interleave(torch.unsqueeze(angle, 1), ox.shape[1], dim=1)
        sin_theta2d = torch.repeat_interleave(torch.unsqueeze(sin_theta, 1), ox.shape[1], dim=1)
        cos_theta2d = torch.repeat_interleave(torch.unsqueeze(cos_theta, 1), ox.shape[1], dim=1)
        new_ox = ox * cos_theta2d - oy * sin_theta2d
        new_oy = ox * sin_theta2d + oy * cos_theta2d
        new_oa = torch.arctan2(torch.sin(oa - angle2d), torch.cos(oa - angle2d))

        transformed_sequence['objects']['x'] = new_ox
        transformed_sequence['objects']['y'] = new_oy
        transformed_sequence['objects']['a'] = new_oa

    if torch.numel(transformed_sequence['walls']['x'])>0:
        wx = transformed_sequence['walls']['x']
        wy = transformed_sequence['walls']['y']
        sin_theta2d = torch.repeat_interleave(torch.unsqueeze(sin_theta, 1), wx.shape[0], dim=1)
        cos_theta2d = torch.repeat_interleave(torch.unsqueeze(cos_theta, 1), wx.shape[0], dim=1)
        new_wx = wx * cos_theta2d[-1] - wy * sin_theta2d[-1]
        new_wy = wx * sin_theta2d[-1] + wy * cos_theta2d[-1]

        transformed_sequence['walls']['x'] = new_wx
        transformed_sequence['walls']['y'] = new_wy


    return transformed_sequence

