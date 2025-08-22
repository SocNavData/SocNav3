import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import os, io

import metrics
import yaml
import argparse
# import warnings
# warnings.simplefilter("error")
import wandb

from dataset_rnn import TrajectoryDataset, collate_fn
from rnn import RNNModel

from utils import plot_predictions_vs_expected, plot_qualitative_multiple

DEFAULT_SEED = 42
DATA_AUGMENTATION = True

help = "You can specify a training task file other than `train.yaml` and/or overwrite the config-based seed."
parser = argparse.ArgumentParser(description=help)
parser.add_argument('--task', type=str, default="Simple MLP baseline/RNN_Model.yaml", help="yaml file path")
args = parser.parse_args()


config_data = yaml.load(open(args.task, "r"), Loader=yaml.Loader)

LOSS = config_data["LOSS"]
TRAIN_FILE = config_data["TRAIN_FILE"]
DEV_FILE = config_data["DEV_FILE"]
TEST_FILE = config_data["TEST_FILE"]
BATCH_SIZE = config_data["BATCH_SIZE"]
MAX_EPOCHS = config_data["MAX_EPOCHS"]
MAX_PATIENCE = config_data["MAX_PATIENCE"]
# Q_TEST_DIR = config_data["Q_TEST_DIR"]
SAVE_PLOTS_LOCALLY = config_data["SAVE_PLOTS_LOCALLY"]
HIDDEN_SIZE = config_data["HIDDEN_SIZE"]
ACTIVATION = config_data["ACTIVATION"]
ADD_CONTEXT_TO_OUTPUT = config_data['ADD_CONTEXT_TO_OUTPUT']
SEED = config_data["SEED"]
DATA_LIMIT = config_data["DATAPOINT_LIMIT"]
LR = config_data["LR"]
RNN_TYPE = config_data["RNN_TYPE"]
NUM_LAYERS = config_data["NUM_LAYERS"]
LINEAR_LAYERS = config_data["LINEAR_LAYERS"]
DROPOUT = config_data["DROPOUT"]
UPLOAD_TO_WANDB = config_data["UPLOAD_TO_WANDB"]
PROJECT_NAME = config_data["PROJECT NAME"]
FRAME_THRESHOLD = config_data["FRAME THRESHOLD"]
DATASET_PATH = config_data["DATASET_PATH"]
CONTEXTQ_FILE = config_data["CONTEXTQ_FILE"]
#SOCIAL_SPACE_THRESHOLD = 0.5
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

Q_TEST_PATH = config_data["Q_TEST_PATH"]
config_qtest = yaml.load(open(config_data['Q_TEST_FILE'], "r"), Loader=yaml.Loader)
Q_FILES = config_qtest["FILES"]
Q_NAMES = config_qtest["NAMES"]
COLORS = config_qtest["COLORS"]
CONTEXTS = config_qtest["CONTEXTS"]


SAVE_NAME = '_'.join(['ALL_rm_rf_try', RNN_TYPE, LOSS, str(NUM_LAYERS), str(LR), str(HIDDEN_SIZE)])

torch.manual_seed(SEED)
np.random.seed(SEED)
if UPLOAD_TO_WANDB is True:
    import wandb
    wandb.login()
    wandb.init(
        # Set the project where this run will be logged
        project=PROJECT_NAME,
        # Track hyperparameters and run metadata
        config={
        "learning_rate": LR,
        "name": SAVE_NAME,
        "epochs": MAX_EPOCHS,
        "seed": SEED,
        })
    wandb.save(args.task, policy="now")

SAVE_NAME = SAVE_NAME if UPLOAD_TO_WANDB is False else wandb.run.name
Patience = 50


def train_model(hidden_size=256, num_layers=10, 
                batch_size=32, num_epochs=1000, patience =50, learning_rate=0.00005, 
                checkpoint_dir='checkpoints'):
    # Create checkpoint directory if it doesn't exist
    model_name = SAVE_NAME+'.pytorch'
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, model_name)

    
    train_dataset = TrajectoryDataset(TRAIN_FILE, CONTEXTQ_FILE, path = DATASET_PATH, limit=DATA_LIMIT, frame_threshold=FRAME_THRESHOLD, data_augmentation = DATA_AUGMENTATION, reload = True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)

    all_features = train_dataset.get_all_features()
    print("Num Features:", len(all_features))
    input_size = len(all_features)

    if ADD_CONTEXT_TO_OUTPUT:
        context_features = len(train_dataset.get_context_features())
    else:
        context_features = 0

    val_dataset = TrajectoryDataset(DEV_FILE, CONTEXTQ_FILE, path = DATASET_PATH, limit=DATA_LIMIT, frame_threshold=FRAME_THRESHOLD, data_augmentation = DATA_AUGMENTATION, reload = True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    model = RNNModel(input_size, hidden_size, num_layers, RNN_TYPE, linear_layers = LINEAR_LAYERS, 
                     activation = ACTIVATION, context_vars = context_features, dropout = DROPOUT)
    model = model.to(device)
    if LOSS == "mse":
        criterion = nn.MSELoss().to(device)
    elif LOSS == "bce":
        criterion = nn.BCELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    min_val_loss = float('inf')
    best_epoch = 1
    patience_counter = patience  # Using your global Patience variable
    train_losses, val_losses = [], []
    
    print(f"Starting training for {num_epochs} epochs (or until early stopping)")
    print(f"Model will be saved to {checkpoint_path} when validation loss improves")
    
    for epoch in range(num_epochs):
        # Training Loop
        model.train()
        epoch_loss = 0
        for trajectories, labels, slengths in train_dataloader:
            trajectories = trajectories.to(device)
            labels = labels.to(device)
            slengths = slengths.to(device)
            outputs = model(trajectories, slengths)
            # print(f"outputs shape: {outputs.shape}, labels shape: {labels.shape}")
            loss = criterion(outputs.squeeze(), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()*len(labels)
        
        train_loss = epoch_loss / len(train_dataset)
        train_losses.append(train_loss)
        
        # Validation Loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for trajectories, labels, slengths in val_dataloader:
                trajectories = trajectories.to(device)
                labels = labels.to(device)
                slengths = slengths.to(device)
                outputs = model(trajectories, slengths)
                loss = criterion(outputs.squeeze(), labels)
                val_loss += loss.item()*len(labels)
        full_preds = []
        #Get Qualitative predictions and the plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 6*2))

        axes = axes.reshape(1, -1)

        # Flatten the axes array for easy iteration
        axes_flat = axes.flatten()

        # Hide any unused subplots
        for i in range(4, len(axes_flat)):
            axes_flat[i].set_visible(False)
        for i_d, speed in enumerate(list(qual_loaders.keys())):
            q_preds = {}
            q_list = qual_loaders[speed]
            # for file in q_list:
            # print(f"File : {q_list}")
            for i, q_tuple in enumerate(q_list):
                full_preds = []
                # print(f"Q_tuple {i}:{len(q_tuple)}")
                context, files, qual_loader = q_tuple
                # print(f"Context :{context}, files: {files}, qual_loader: {qual_loader}")
                with torch.no_grad():
                    # print(f"======================\n{filename=}")
                    for trajectories, _, slengths in qual_loader:
                        # Pass the whole batched graph sequence to the model at once
                        trajectories = trajectories.to(device)
                        slengths = slengths.to(device)
                        preds = model(trajectories, slengths)
                        full_preds += preds.tolist()
                # print(f"Predictions for Context {context}: \n {all_preds}")
                q_preds[context] = full_preds
            for idx, context in enumerate(q_preds.keys()):
                # axes_flat[idx] = plot_qualitative_ratings(q_preds[context], speed, context, COLORS[i_d], ax=axes_flat[idx])
                axes_flat[idx] = plot_qualitative_multiple(q_preds[context], speed, context, COLORS[i_d], ax=axes_flat[idx])


        # model_name = f"{save_dir[:10]}_ep_{str(epoch)}"
        plt.tight_layout(pad=1.08, h_pad=None, w_pad=None, rect=None)
        save_dir = os.path.join('plots', SAVE_NAME+'_q_test')
        save_dir = save_dir if UPLOAD_TO_WANDB is False else os.path.join('plots',wandb.run.name)
        os.makedirs(save_dir, exist_ok=True)
        save_loc = os.path.join(save_dir, 'e_'+str(epoch+1))
        if SAVE_PLOTS_LOCALLY:
            plt.savefig(save_loc)
        if UPLOAD_TO_WANDB:
            q_plt_buf = io.BytesIO()
            plt.savefig(q_plt_buf, format='png')
            q_plt_buf.seek(0)
            q_plt_img = wandb.Image(Image.open(q_plt_buf))
        plt.close()
        # Get Test Plots
        preds = []
        labels = []
        # print("TESTLOADER LENGTH:", len(test_dataloader))
        with torch.no_grad():
            for trajectories, label, slengths in test_dataloader:
                # print("TESTING")
                trajectories = trajectories.to(device)
                slengths = slengths.to(device)
                outputs = model(trajectories, slengths)
                preds.extend(outputs.squeeze().tolist())
                labels.extend(label.squeeze().tolist())

        # Second plot: Predictions vs Expected
        plt.figure(figsize=(12, 6))  # Create a new figure with proper size
        pve_plt = plot_predictions_vs_expected(preds, labels)
        # Save a copy of the current figure for local storage
        if SAVE_PLOTS_LOCALLY:
            pve_plt_path = os.path.join(save_dir, f"pve_{epoch+1}.png")
            pve_plt.savefig(pve_plt_path)

        # Create a temporary buffer for wandb
        if UPLOAD_TO_WANDB:
            pve_plt_buf = io.BytesIO()
            pve_plt.savefig(pve_plt_buf, format='png')
            pve_plt_buf.seek(0)
            pve_plt_img = wandb.Image(Image.open(pve_plt_buf))
        plt.close()  # Close the figure
        
        val_loss /= len(val_dataset)
        val_losses.append(val_loss)

        # Checkpointing - save when validation loss improves
        if val_loss < min_val_loss:
            improvement = min_val_loss - val_loss
            min_val_loss = val_loss
            best_epoch = epoch + 1
            patience_counter = patience  # Reset patience counter
            
            # Save the checkpoint with detailed information
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
                'train_loss': train_loss,
                'rnn_type': RNN_TYPE,
                'input_size': input_size,
                'hidden_size': hidden_size,
                'num_layers': num_layers,
                'linear_layers': LINEAR_LAYERS,
                'frame_threshold': FRAME_THRESHOLD,
                'activation': ACTIVATION,
                'context_features': context_features
            }
            
            torch.save(checkpoint, checkpoint_path)
            
            print(f"Checkpoint saved at epoch {epoch+1}")
            print(f"Val Loss: {val_loss:.6f} (improved by {improvement:.6f})")
            
            # Optionally save a timestamped version too if you want a history
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # history_path = os.path.join(checkpoint_dir, f'model_e{epoch+1}_{timestamp}.pth')
            # torch.save(checkpoint, history_path)
        else:
            patience_counter -= 1
            print(f"****Epoch {epoch+1}: Val Loss: {val_loss:.6f} (best: {min_val_loss:.6f}, patience: {patience_counter}/{patience})")
        
        if patience_counter == 0:
            print(f"Early stopping at epoch {epoch+1}. Best Val Loss: {min_val_loss:.6f}")
            break

        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
        # Log training loss, validation loss, and gradient statistics to WandB
        if UPLOAD_TO_WANDB is True:
            wandb.log({
                "training_loss": train_loss,
                "validation_loss": val_loss,
                "lr": LR,
                "Qualitative_test": q_plt_img,
                "min_val_loss": min_val_loss,
                "best_epoch": best_epoch,
                "Predicted_vs_Expected_ratings": pve_plt_img
            })
        


    # Plot Training Progression
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.grid(True, alpha=0.3)
    fig_path = os.path.join('plots', SAVE_NAME+'_progression'+'.png')
    plt.savefig(fig_path)
    # plt.show()
    print(f"\nTraining completed. Best model saved at: {checkpoint_path}")
    return model, checkpoint_path




test_dataloader = []
qual_loaders = []
test_dataset = TrajectoryDataset(TEST_FILE, CONTEXTQ_FILE, path = DATASET_PATH, frame_threshold=FRAME_THRESHOLD)
test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)


def get_qual_loader(file_path):
    # print(f"Processing file: {file_path}")
    q_loaders = []
    with open(file_path) as set_file:
        files = set_file.read().splitlines()
    # context = {"urgency":98., "importance":98., "risk":95 , "distance_from_human":95., "minimum_speed":20, "average_speed":15., "maximum_speed":15.}
    for context in CONTEXTS:
        print(f"Creating q_test for: {context}")
        qual_set = TrajectoryDataset(file_path, CONTEXTQ_FILE, path = Q_TEST_PATH, frame_threshold=FRAME_THRESHOLD, label_exists=False, overwrite_context=context)
        qual_loader = DataLoader(qual_set,  batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
        q_loaders.append((context, files, qual_loader))
    return q_loaders
# Load and process test datasets
qual_loaders = {}
for id, path in enumerate(Q_FILES):
    speed = Q_NAMES[id]
    print(f"Processing files with {speed}")
    qual_loaders[speed] = get_qual_loader(os.path.join(Q_TEST_PATH,path))

model, checkpoint = train_model(hidden_size=HIDDEN_SIZE, num_epochs=MAX_EPOCHS, num_layers=NUM_LAYERS,learning_rate=LR, patience= MAX_PATIENCE, batch_size= BATCH_SIZE)
if UPLOAD_TO_WANDB is True:
    wandb.finish()

