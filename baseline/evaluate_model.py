import sys
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
import numpy as np
from rnn import RNNModel
from dataset_rnn import TrajectoryDataset, collate_fn


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

datafile = sys.argv[2]
contextQ_file = sys.argv[3]

dataset = TrajectoryDataset(datafile, contextQ_file, path = '../dataset', limit=-1, frame_threshold=FRAME_THRESHOLD, data_augmentation=False, reload = False)
loader = DataLoader(dataset,batch_size=32, shuffle=False, collate_fn=collate_fn)

mse_function = torch.nn.MSELoss().to(device)
mae_function = torch.nn.L1Loss().to(device)

model.eval()
loss_mse = 0
loss_mae = 0
with torch.no_grad():
    for val_data in loader:
        traj_batch, label_batch, seq_length = val_data
        label_batch = label_batch.to(device)
        traj_batch = traj_batch.to(device)
        seq_length = seq_length.to(device)

        # Get predictions for the whole batch
        predictions = model(traj_batch, seq_length).squeeze()
        # print(predictions, label_batch)
        loss_mse += mse_function(predictions, label_batch).item() * len(label_batch)
        loss_mae += mae_function(predictions, label_batch).item() * len(label_batch)
    
loss_mse /= len(dataset)
loss_mae /= len(dataset)

print("MSE", loss_mse)
print("MAE", loss_mae)
