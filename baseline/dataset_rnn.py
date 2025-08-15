import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from shapely.geometry import Point
import math
import json
import numpy as np
import metrics
import os
import pandas as pd
import sys
import random

sys.path.append(os.path.join(os.path.dirname(__file__),'../tools/data_transformation'))
from data_mirroring import mirror_sequence
from data_normalization import transform_to_goal_fr


SOCIAL_SPACE_THRESHOLD = 0.4


# Custom Dataset for loading JSON files

class TrajectoryDataset(Dataset):
    def __init__(self, data_file, contextQ_file, path = '../dataset', limit= -1,  frame_threshold = 0.1, label_exists = True, 
                 overwrite_context = False, data_augmentation = False, reload = False):
        self.orig_data = []
        self.data = []
        self.mirrored_data = []
        self.labels = []
        self.traj_metrics = []
        self.mirrored_traj_metrics = []
        self.mirrored_orig_data = []
        self.sequence_indices = []
        self.path = path
        self.data_file = data_file

        if type(data_file) is str:
            DA = '_DA_' if data_augmentation else ''
            self.reload_fname = '.'.join(data_file.split('.')[:-1]) + '_' + contextQ_file.split('/')[-1].split('.csv')[0] + DA + '.pytorch'
        else:
            self.reload_fname = ''
            reload = False
        self.label_exists = label_exists
        self.limit = limit
        self.overwrite_contexts = overwrite_context
        self.frame_threshold = frame_threshold
        self.data_augmentation = data_augmentation
        self.context_df = pd.read_csv(contextQ_file, index_col='context') 
        
        self.robot_features = ['robot_x', 'robot_y', 'robot_a', 'speed_x', 'speed_y', 'speed_a', 'acceleration_x', 'acceleration_y']

        self.metrics_features = ['success', 'hum_exists', 'wall_exists', 'dist_nearest_hum', 'dist_nearest_obj', 'dist_wall', 'dist_goal',
                            'hum_collision_flag', 'object_collision_flag', 'wall_collision_flag', 'social_space_intrusionA',
                            'social_space_intrusionB', 'social_space_intrusionC', 'num_near_humansA', 'num_near_humansB', 'num_near_humansC',
                           'num_near_humansA2', 'num_near_humansB2', 'num_near_humansC2',
                             'min_time_to_collision', 'min_time_to_collision2', 'max_fear', 'max_panic',
                             'global_dist_nearest_hum', 'path_efficiency_ratio', 'step_ratio', 'episode_end']
        self.context_features = ['urgency', 'importance', 'risk', 'distance_from_human', 'distance_from_object', 'speed', 'comfort', 
                                 'bumping_human', 'bumping_object', 'predictability']
        self.goal_features = ['goal_pos_threshold', 'goal_angle_threshold']
        
        self.all_features = self.robot_features + self.metrics_features + self.goal_features + self.context_features

        self.MAX_TTC = 10
        self.MAX_LSPEED = 2

        self.max_metric_values = {'robot_x': 10, 'robot_y': 10, 'robot_a': np.pi, 'speed_x': self.MAX_LSPEED, 
                            'speed_y': self.MAX_LSPEED, 'speed_a': np.pi, 'acceleration_x': 3, 
                            'acceleration_y': 3, 'success': 1, 'hum_exists': 1, 'wall_exists': 1,
                            'dist_nearest_hum': 10, 'dist_nearest_obj': 10, 'dist_wall': 10, 'dist_goal': 10,
                            'hum_collision_flag': 1, 'object_collision_flag': 1, 'wall_collision_flag': 1, 
                            'social_space_intrusionA': 1, 'social_space_intrusionB': 1, 'social_space_intrusionC': 1,
                            'num_near_humansA': 10, 'num_near_humansB': 10, 'num_near_humansC': 10, 
                            'min_time_to_collision': self.MAX_TTC, 'max_fear': 10, 'max_panic': 10,
                            'num_near_humansA2': 100, 'num_near_humansB2': 100, 'num_near_humansC2': 100,
                            'min_time_to_collision2': self.MAX_TTC**2,
                            'global_dist_nearest_hum': 10, 'path_efficiency_ratio': 1, 'step_ratio': 1, 'episode_end': 1,
                            'goal_pos_threshold': 10, 'goal_angle_threshold': np.pi}


        if reload is True:
            if os.path.exists(self.reload_fname):
                loaded = torch.load(self.reload_fname)
                self.data = loaded['data']
                self.mirrored_data = loaded['mirrored_data']
                self.labels = loaded['labels']
                print("number of trajectories for ", self.data_file, len(self.data))
                return


        if type(self.data_file) is str and self.data_file.endswith('.txt'):
            print(self.data_file)
            with open(self.data_file) as set_file:
                ds_files = set_file.read().splitlines()

            print("number of files for ", self.data_file, len(ds_files))
        elif type(self.data_file) is str and self.data_file.endswith('.json'):
            ds_files = [self.data_file]
        elif type(self.data_file) is list:
            ds_files = self.data_file

        for i, filename in enumerate(ds_files):
            if filename.endswith('.json'):
                file_path = os.path.join(self.path, filename)
                with open(file_path, 'r', encoding="utf-8") as f:
                    try:
                        t_data = json.load(f)
                    except:
                        print("FileName :", file_path)
                    # Adjust keys based on your file structure
                    
                    t_data_normalized = transform_to_goal_fr(t_data)
                    self.orig_data.append(t_data_normalized)
                    trajectory, feats_dict, seq_indices = self.gather_data(t_data_normalized)
                    if self.label_exists and 'label' in t_data: 
                        rating = t_data['label']
                    else:
                        rating = 0.0
                    self.data.append(trajectory)
                    self.labels.append(rating)
                    self.traj_metrics.append(feats_dict)
                    self.sequence_indices.append(seq_indices)


                    if self.data_augmentation:

                        t_data_mirrored = mirror_sequence(t_data_normalized)

                        self.orig_data.append(t_data_mirrored)
                        trajectory, feats_dict, seq_indices = self.gather_data(t_data_mirrored)
                        self.data.append(trajectory)
                        self.labels.append(rating)
        
                        self.traj_metrics.append(feats_dict)
                        self.sequence_indices.append(seq_indices)

            if i%1000 == 0:
                print(i)
            if i + 1 >= self.limit and self.limit > 0:
                print('Stop including more samples to speed up dataset loading')
                break
        if reload:
            torch.save({
                'data': self.data,
                'mirrored_data': self.mirrored_data,
                'labels': self.labels
                }, self.reload_fname)


    def get_metrics(self, frame, walls, prev_frame, cur_step, last_step):
        cur_metrics = {}
        cur_timestamp = frame['timestamp']
        # print("Previous Frame",prev_frame)
        prev_timestamp = prev_frame['timestamp']
        window = cur_timestamp - prev_timestamp
        for feature in self.metrics_features:
            cur_metrics[feature] = 0
        #Get robot features for later
        r_x = frame['robot']['x']
        r_y = frame['robot']['y']
        r_a = frame['robot']['angle']
        r_vx = frame['robot']['speed_x']
        r_vy = frame['robot']['speed_y']
        r_va = frame['robot']['speed_a']
        x_moved = abs(r_x - prev_frame['robot']['x'])
        y_moved = abs(r_y - prev_frame['robot']['y'])
        dist_moved = math.sqrt((x_moved)**2 + (y_moved)**2)
        self.distance_travelled += dist_moved


        cur_metrics['robot_x'] = r_x
        cur_metrics['robot_y'] = r_y
        cur_metrics['robot_a'] = r_a

        cur_metrics['speed_x'] = r_vx
        cur_metrics['speed_y'] = r_vy
        cur_metrics['speed_a'] = r_va

        if window != 0:
            p_vx, p_vy = prev_frame['robot']['speed_x'], prev_frame['robot']['speed_y']
            acc_x = (cur_metrics['speed_x'] - p_vx) / window
            acc_y = (cur_metrics['speed_y'] - p_vy) / window
        else:
            acc_x = 0
            acc_y = 0


        cur_metrics['acceleration_x'] = acc_x
        cur_metrics['acceleration_y'] = acc_y

        #Goal Features
        g_x = frame['goal']['x']
        g_y = frame['goal']['y']
        g_a = frame['goal']['angle']


        r_radius = frame['robot']['shape']['length']/2.   #since length and width are the same

        g_dist =  max(0, math.sqrt((g_x - r_x)**2 + (g_y - r_y)**2))
        if self.initial_distance_to_goal < 0:
            self.initial_distance_to_goal = g_dist
        cur_metrics['dist_goal'] = g_dist
        g_thr = frame['goal']['pos_threshold'] + 0.1
        a_thr = frame['goal']['angle_threshold']
        cur_metrics['goal_pos_threshold'] = g_thr
        cur_metrics['goal_angle_threshold'] = a_thr
        cur_metrics['path_efficiency_ratio'] = min(1, self.initial_distance_to_goal/(self.distance_travelled+1e-6))
        # print(cur_timestamp, self.distance_travelled)
        #Check for goal success
        if g_dist <= g_thr and abs(np.arctan2(np.sin(g_a - r_a), np.cos(g_a - r_a))) <= a_thr:
            cur_metrics['success'] = 1
        #Calculate human metrics
        min_hdist = self.max_metric_values['dist_nearest_hum']    #Initialising distance with a large value
        h_radius = 0.3
        min_ttc = float('inf')     #Time to collision
        max_fear = -1
        max_panic = -1
        for human in frame['people']:
            cur_metrics['hum_exists'] = 1
            h_x = human['x']
            h_y = human['y']


            dist_to_robot = max(0, math.sqrt((h_x - r_x)**2 + (h_y - r_y)**2) - (r_radius + h_radius))
            min_hdist = min(min_hdist, dist_to_robot)
            if min_hdist == 0:
                cur_metrics['hum_collision_flag'] = 1
            #Calculate num of humans in vicinity
            if dist_to_robot < SOCIAL_SPACE_THRESHOLD:                
                cur_metrics['num_near_humansA'] += 1
                cur_metrics['social_space_intrusionA'] = 1
            if dist_to_robot < SOCIAL_SPACE_THRESHOLD*1.5:
                cur_metrics['num_near_humansB'] += 1
                cur_metrics['social_space_intrusionB'] = 1
            if dist_to_robot < SOCIAL_SPACE_THRESHOLD*2.0:
                cur_metrics['num_near_humansC'] += 1
                cur_metrics['social_space_intrusionC'] = 1
        cur_ttc = metrics.get_ttc(frame, prev_frame)
        valid_ttc_exists = False    
        # Process each dictionary in the list
        for item in cur_ttc:
            # Handle ttc - exclude -1 values
            if item['ttc'] != -1:
                valid_ttc_exists = True
                min_ttc = min(min_ttc, item['ttc'])
                
            # Handle fear - find maximum value
            if item['fear'] > max_fear:
                max_fear = item['fear']
                
            # Handle panic - find maximum value
            if item['panic'] > max_panic:
                max_panic = item['panic']
        
        # If no valid ttc values were found, set min_ttc to -1
        if not valid_ttc_exists:
            min_ttc = self.MAX_TTC #-1
        if max_panic<0:
            max_panic = 0
        if max_fear<0:
            max_fear = 0
        cur_metrics['min_time_to_collision'] = min_ttc
        cur_metrics['max_fear'] = max_fear
        cur_metrics['max_panic'] = max_panic

        cur_metrics['dist_nearest_hum'] = min_hdist
        self.global_dist_nearest_hum = min(min_hdist, self.global_dist_nearest_hum)
        cur_metrics['global_dist_nearest_hum'] = self.global_dist_nearest_hum

        min_odist = self.max_metric_values['dist_nearest_obj']
        robot = Point(r_x, r_y).buffer(r_radius)
        for object in frame['objects']:
            o_x = object['x']
            o_y = object['y']
            o_angle = object['angle']
            dist_to_robot = metrics.get_dist_from_obj(object, o_x, o_y, o_angle, robot)
            min_odist = min(min_odist, dist_to_robot)
            if min_odist == 0:
                cur_metrics['object_collision_flag'] = 1
        cur_metrics['dist_nearest_obj'] = min_odist

        #Calculate distance to wall
        min_wdist = self.max_metric_values['dist_wall']
        if len(walls)>0:
            cur_metrics['wall_exists'] = 1
            for wall in walls:
                w_x1, w_y1 = wall[0], wall[1]
                w_x2, w_y2 = wall[2], wall[3]
                w_dist = metrics.get_wall_distance(r_x, r_y, r_radius, w_x1, w_y1, w_x2, w_y2)
                min_wdist = min(w_dist, min_wdist)
                if w_dist == 0:
                    cur_metrics['wall_collision_flag'] = 1
        cur_metrics['dist_wall'] = min_wdist

        # Squared metrics
        cur_metrics['num_near_humansA2'] = cur_metrics['num_near_humansA']**2
        cur_metrics['num_near_humansB2'] = cur_metrics['num_near_humansB']**2
        cur_metrics['num_near_humansC2'] = cur_metrics['num_near_humansC']**2
        cur_metrics['min_time_to_collision2'] = cur_metrics['min_time_to_collision']**2

        cur_metrics['step_ratio'] = cur_step / last_step
        cur_metrics['episode_end'] = 1 if cur_step == last_step else 0

        for m in cur_metrics:
            max_val = self.max_metric_values[m]
            cur_metrics[m] = max(-max_val, min(cur_metrics[m], max_val))/max_val

        return cur_metrics

    def gather_data(self, data):
        sequence = data['sequence']
        trajectory_data = []
        detailed_feats = []
        walls = data['walls']
        last_timestamp = -float('inf') #np.sequence[0]['timestamp']
        seq_indices = []
        self.initial_distance_to_goal = -1
        self.distance_travelled = 0
        self.global_dist_nearest_hum = self.max_metric_values['global_dist_nearest_hum']
        prev_index = 0

        last_i = len(sequence)-1
        for i, frame in enumerate(sequence):
            current_timestamp = frame['timestamp']
            if current_timestamp-last_timestamp >= self.frame_threshold or i==last_i:
                frame_features = []
                seq_indices.append(i)
                if i == 0:
                    prev_frame = frame
                else:
                    prev_frame = sequence[prev_index]
                # print("Prev Frame here:", prev_frame)
                cur_metrics = self.get_metrics(frame, walls, prev_frame, i, last_i)
                # json_metrics.append(cur_metrics)
                for feature in self.robot_features:
                    frame_features.append(cur_metrics.get(feature, 0.0))
                for feature in self.metrics_features:
                    frame_features.append(cur_metrics.get(feature, 0.0))
                for feature in self.goal_features:
                    frame_features.append(cur_metrics.get(feature, 0.0))
                if 'context_description' in data.keys(): #not self.overwrite_contexts:
                    context_desc = data['context_description']
                else:
                    context_desc = self.overwrite_contexts
                context = self.context_df.loc[context_desc.rstrip()].to_dict()
                for feature in self.context_features:
                    # print(f"Contexts :{context.get(feature)}")
                    frame_features.append(context.get(feature, 0.0)/100)
                #add goal 
                # goal = frame['goal']
                # for feature in goal_features:
                #     frame_features.append(goal.get(feature, 0.0))
                last_timestamp = current_timestamp
                # print(frame_features, len(frame_features))
                prev_index = i
                trajectory_data.append(frame_features)
                detailed_feats.append(dict(map(lambda i,j : (i,j) , self.all_features, frame_features)))
                # print(f"Frame Features :{detailed_feats}")
        if trajectory_data:  # Only add non-empty sequences
            # steps -= 1
            # step_feat = self.all_features.index('step_ratio')
            # for t in range(len(trajectory_data)):
            #     trajectory_data[t][step_feat] /= steps
            #     detailed_feats[t]['step_ratio'] = trajectory_data[t][step_feat]

            # with open(metrics_file, "w") as f:
            #     json.dump(json_metrics, f, indent=4)

            return trajectory_data, detailed_feats, seq_indices
        else:
            print("sequence too short!! Returning empty sequence")
            return [[0.0] * len(self.all_features)], detailed_feats, seq_indices  # Return a dummy sequence if empty
        
    def get_all_features(self):
        return self.all_features

    def get_context_features(self):
        return self.context_features

    def __len__(self):
        return len(self.data)+len(self.mirrored_data)
    
    def __getitem__(self, idx):
        # data = self.orig_data[idx]
        # if self.data_augmentation:
        #     if random.randint(0,1)==1:
        #         data = mirror_sequence(data)

        # trajectory, feats_dict, seq_indices = self.gather_data(data)

        # if self.data_augmentation and random.randint(0,1)==1:
        #     data = self.mirrored_data[idx]
        # else:
        #     data = self.data[idx]

        # if self.data_augmentation:
        #     if idx%2==0:
        #       data = self.data[idx//2]
        #     else:
        #       data = self.mirrored_data[idx//2]
        #     label = self.labels[idx//2]
        # else:
        #     data = self.data[idx]
        #     label = self.labels[idx]

        data = self.data[idx]
        label = self.labels[idx]


        traj = torch.tensor(data, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        return traj, label

def collate_fn(batch):
    sequences, labels = zip(*batch)  # Separate sequences and labels
    sequence_lengths = [len(s)-1 for s in sequences]
    sequences = pad_sequence(sequences, batch_first=True, padding_value=0)  # Pad sequences
    labels = torch.stack(labels)  # Convert labels to tensor
    return sequences, labels, torch.tensor(sequence_lengths, dtype=torch.long)

