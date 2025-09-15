import math
import random
import torch
from data_conversions import clone_sequence


def tensor_transform_with_random_noise(tDict_sequence):
    transformed_sequence = clone_sequence(tDict_sequence)
    transformed_sequence['robot']['x'] += torch.rand(transformed_sequence['robot']['x'].shape, dtype=torch.float64)*0.1
    transformed_sequence['robot']['y'] += torch.rand(transformed_sequence['robot']['y'].shape, dtype=torch.float64)*0.1
    transformed_sequence['robot']['a'] += torch.rand(transformed_sequence['robot']['a'].shape, dtype=torch.float64)*math.pi/90.
    transformed_sequence['people']['x'] += torch.rand(transformed_sequence['people']['x'].shape, dtype=torch.float64)*0.1
    transformed_sequence['people']['y'] += torch.rand(transformed_sequence['people']['y'].shape, dtype=torch.float64)*0.1
    transformed_sequence['people']['a'] += torch.rand(transformed_sequence['people']['a'].shape, dtype=torch.float64)*math.pi/90.
    transformed_sequence['objects']['x'] += torch.rand(transformed_sequence['objects']['x'].shape, dtype=torch.float64)*0.1
    transformed_sequence['objects']['y'] += torch.rand(transformed_sequence['objects']['y'].shape, dtype=torch.float64)*0.1
    transformed_sequence['objects']['a'] += torch.rand(transformed_sequence['objects']['a'].shape, dtype=torch.float64)*math.pi/90.
    return transformed_sequence

