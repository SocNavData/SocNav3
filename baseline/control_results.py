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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple
import argparse

parser = argparse.ArgumentParser(prog='control results',
                    description='Evaluate the specified model againts the control questions.')
parser.add_argument('--model', type=str, nargs="?", required = True, help="Model used for rating prediction.")
parser.add_argument('--context', type=str, nargs="?", required = True,  help="Context file for the generation of the context variables.")
parser.add_argument('--control_path', type=str, nargs="?", default='.', help="Path to the labeled control trajectories.")

args = parser.parse_args()

checkpoint_path = args.model
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

controldir = args.control_path
contextQ_file = args.context

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

        predictions = model(traj_batch, seq_length).squeeze()
        # print('predictions', predictions.tolist())
predictions = predictions.tolist()

x = [x for x in range(len(mean_control))]

means, predictions = zip(*sorted(zip(mean_control, predictions)))
means, x_sort,     = zip(*sorted(zip(mean_control, x)))
means, std_control     = zip(*sorted(zip(mean_control, std_control)))

errors = [p-m for m,p in zip(means,predictions)]
errorsP = [ [0,          max(0, e)] for e in errors ]
errorsN = [ [-min(0, e),         0] for e in errors ]

x, x_sort, means, predictions = np.array(x), np.array(x_sort), np.array(means), np.array(predictions)
errors, errorsP, errorsN = np.array(errors), np.array(errorsP), np.array(errorsN)

from matplotlib.legend_handler import HandlerBase
from matplotlib.lines import Line2D
import matplotlib.patches as mpatches

# Custom handler that draws crossing lines
class HandlerCross(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        cx = xdescent + width / 2
        cy = ydescent + height / 2
        # size = min(width, height) / 2.5
        DX=width/9
        DY=height/3

        style = orig_handle.get('style', '+')

        if style == "x":
            line1 = Line2D([cx-DX, cx+DX],     [cy, cy], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line2 = Line2D([cx+DX, cx+DX*1.8], [cy, cy+DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line3 = Line2D([cx+DX, cx+DX*1.8], [cy, cy-DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line4 = Line2D([cx-DX, cx-DX*1.8], [cy, cy+DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)
            line5 = Line2D([cx-DX, cx-DX*1.8], [cy, cy-DY], color=orig_handle.get('color', 'darkorange'), linewidth=orig_handle.get('linewidth', 2), transform=trans)

            return [line1, line2, line3, line4, line5]
        else:
            ddx = DX*0.1
            linewidth = orig_handle.get('linewidth', 2)
            line1 = Line2D([cx-ddx, cx+ddx], [cy, cy], color=orig_handle.get('color', 'darkorange'), linewidth=linewidth, transform=trans)
            return [line1]


mse = np.mean((means - predictions) ** 2)
mae = np.mean(np.abs(means - predictions))
print(f"MSE={mse}  MAE={mae}")



fig, ax = plt.subplots(figsize=(5, 3.5))
plt.ylim(0, 1.1)
plt.xlabel("control trajectories (sorted)")
plt.ylabel("score")


plt.axhline(y=1, color='gray', linestyle='-', linewidth=1, zorder=1)
plt.errorbar(x, means, yerr=std_control,   fmt='s', color='black', ecolor='black', elinewidth=1, capsize=3, ms=4.5,  zorder=2)
# plt.scatter(x, predictions, facecolors=None,  zorder=3)

LW=1.1
DX=0.1
DY=0.01

for xi, yi in zip(x, predictions):
    plt.plot([xi-DX, xi+DX], [yi, yi], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi+DX, xi+DX*1.8], [yi, yi+DY], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi+DX, xi+DX*1.8], [yi, yi-DY], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi-DX, xi-DX*1.8], [yi, yi+DY], color='darkorange', linewidth=LW, zorder=3)
    plt.plot([xi-DX, xi-DX*1.8], [yi, yi-DY], color='darkorange', linewidth=LW, zorder=3)


cross_handle = {'color': 'darkorange', 'linewidth': LW, "style":"x"}
box_handle = {'color': 'black', 'linewidth': LW*6, "style":"b"}

plt.legend(handles=[cross_handle, box_handle], labels=['estimations', 'mean scores'],
           handler_map={dict: HandlerCross()},
           loc='lower right')


ax.set_xticks(x)
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_xticklabels(x_sort)
plt.tight_layout()

plt.savefig("control_results.pdf", bbox_inches='tight', pad_inches=0)
plt.show()