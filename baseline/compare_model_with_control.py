import sys
import os
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from rnn import RNNModel
from dataset_rnn import TrajectoryDataset, collate_fn
import json
import matplotlib.pyplot as plt

checkpoint_path = sys.argv[1]
print(checkpoint_path)
model_name = checkpoint_path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the checkpoint
checkpoint = torch.load(checkpoint_path, map_location=device)

input_size = checkpoint['input_size']
hidden_size = checkpoint['hidden_size']
num_layers = checkpoint['num_layers']

if 'use_new_context' in checkpoint.keys():
    USE_NEW_CONTEXT = checkpoint['use_new_context']
else:
    USE_NEW_CONTEXT = False
if 'rnn_type' in checkpoint.keys():
    RNN_TYPE = checkpoint['rnn_type']
else:
    RNN_TYPE = 'GRU'
if 'frame_threshold' in checkpoint.keys():
    FRAME_THRESHOLD = checkpoint['frame_threshold']
else:
    FRAME_THRESHOLD = 0.1

if 'linear_layers' in checkpoint.keys():
    LINEAR_LAYERS = checkpoint['linear_layers']
else:
    LINEAR_LAYERS = []

if 'activation' in checkpoint.keys():
    ACTIVATION = checkpoint['activation']
else:
    ACTIVATION = 'linear'

if 'context_features' in checkpoint.keys():
    CONTEXT_FEATURES = checkpoint['context_features']
else:
    CONTEXT_FEATURES = 0


model = RNNModel(input_size, hidden_size, num_layers, rnn_type = RNN_TYPE, linear_layers = LINEAR_LAYERS, activation=ACTIVATION, context_vars=CONTEXT_FEATURES).to(device)
    
model = model.to(device)


model.load_state_dict(checkpoint['model_state_dict'], strict=True)

controldir = sys.argv[2]
contextQ_file = sys.argv[3]

control_indices = [3007,2007,1007,7,302,1302,2302,3102,2002,1002,2,3094,2894,1879,834]
control_indices.sort()

control_scenarios = dict()
evaluation_trajectories = list()
file_list = os.listdir(controldir)
file_list.sort()
for filename in file_list:
    scenario = int(filename.split('_')[0])
    filepath = os.path.join(controldir, filename)
    if scenario not in control_scenarios.keys():
        control_scenarios[scenario] = []
        evaluation_trajectories.append(filepath)
    with open(filepath, 'r') as traj_file:
        trajectory_data = json.load(traj_file)
    control_scenarios[scenario].append(trajectory_data['label'])


mean_control = list()
median_control = list()
std_control = list()
for s in control_scenarios:
    mean = np.mean(control_scenarios[s])
    median = np.median(control_scenarios[s])
    std = np.std(control_scenarios[s])
    mean_control.append(mean)
    median_control.append(median)
    std_control.append(std)

dataset = TrajectoryDataset(evaluation_trajectories, contextQ_file, path = '.', limit=-1, frame_threshold=FRAME_THRESHOLD, data_augmentation=False, reload = False)
loader = DataLoader(dataset,batch_size=32, shuffle=False, collate_fn=collate_fn)

model.eval()
loss_mse = 0
loss_mae = 0
with torch.no_grad():
    for val_data in loader:
        traj_batch, label_batch, seq_length = val_data
        traj_batch = traj_batch.to(device)
        seq_length = seq_length.to(device)

        # Get predictions for the whole batch
        predictions = model(traj_batch, seq_length).squeeze()
        # print(predictions, label_batch)
        print(predictions)
predictions = predictions.tolist()

print('mean', mean_control)
print('median', median_control)  
print('std', std_control)  

MSE_mean = np.square(np.subtract(predictions,mean_control)).mean()
MAE_mean = np.absolute(np.subtract(predictions,mean_control)).mean()
MSE_median = np.square(np.subtract(predictions,median_control)).mean()

print('mse mean', MSE_mean)
print('mae mean', MAE_mean)
print('mse median', MSE_median)


indices = np.arange(0, len(control_indices))

width = 0.2

pos1 = indices - width/2
pos2 = indices + width/2


plt.bar(pos1, predictions, width=width, label='prediction')
plt.bar(pos2, mean_control, width=width, label='mean score')


plt.xlabel('Control trajectory')
plt.ylabel('Score')
plt.title('Comparison with the control ratings')
plt.xticks(indices)  
plt.legend()

plt.savefig('comparison_with_control.pdf', format='pdf', pad_inches=0)
plt.show()