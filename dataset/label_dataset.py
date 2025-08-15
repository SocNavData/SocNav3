import os
import json
import sys
import argparse
import uuid
import shutil

# sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../tools/data_analysis'))
# print(os.path.join(os.path.dirname(os.path.realpath(__file__)),'../tools/data_analysis'))
# import check_quality


import multiprocessing


NUM_PROCESSES = multiprocessing.cpu_count()-4  # -4 not to be too greedy :-)

def add_label(label, context, trajectories_folder, dataset_organization, index, labeled_output_dir, save_subdir = True):
    traj = "0"*(9-len(str(index)))+str(index)+".json"
    for s in dataset_organization:
        if traj in dataset_organization[s]:
            subdir = s
            break

    trajectory_path = os.path.join(os.path.join(trajectories_folder, subdir), traj)

    with open(trajectory_path, 'r') as traj_file:
        trajectory_data = json.load(traj_file)

    trajectory_data['context_description'] = context

    trajectory_data['label'] = label

    output_dir = labeled_output_dir
    if save_subdir:
        output_dir = os.path.join(output_dir, subdir)
        os.makedirs(output_dir, exist_ok=True)        
    
    save_json_with_unique_name(trajectory_data, traj, output_dir)


def save_json_with_unique_name(trajectory_data, trajectory_file, output_dir):
    # Extract the original file name (without extension) and add a unique identifier
    original_name = os.path.splitext(trajectory_file)[0]
    unique_name = f"{original_name}_{uuid.uuid4().hex[:8]}.json"
    output_path = os.path.join(output_dir, unique_name)
        
    # Save the updated JSON to the output directory
    with open(output_path, 'w') as output_file:
        json.dump(trajectory_data, output_file, indent=4)
    print(f"Saved updated trajectory to {output_path}")


def label_dataset(rating_file, trajectories_folder, dataset_organization, control_indices, labeled_output_dir, control_labeled_output_dir):
    # i, response = t
    # file_path = os.path.join(survey_dir, response.id + ".json")
    # print(file_path)

    with open(rating_file, 'r') as f:
        raw_data = f.read()
        # Remove invalid characters
        sanitized_data = raw_data.replace('\n', '').replace('\t', '')
        try:
            labelled_data = json.loads(sanitized_data)
        except Exception as e:
            print(rating_file)
            print(rating_file)
            print(rating_file)
            raise e
        for id in range(len(labelled_data['answers'].keys())):
            label = labelled_data['answers'][str(id)]
            index = labelled_data['indices'][id]
            context = labelled_data['descriptions'][id]
            if index not in control_indices:
                add_label(label, context, trajectories_folder, dataset_organization, index, labeled_output_dir, save_subdir = True)
            else:
                add_label(label, context, trajectories_folder, dataset_organization, index, control_labeled_output_dir, save_subdir = False)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Trajectories dataset labeler')

    parser.add_argument('--trajectories', type=str, nargs='?', required=True, help='Directory containing the trajectories')
    parser.add_argument('--ratings', type=str, nargs='?', required=True, help='Directory containing the ratings')
    
    args = parser.parse_args()

    trajectories_folder = args.trajectories
    if trajectories_folder.endswith('/'):
        trajectories_folder = trajectories_folder[:-1]
    ratings_dir = args.ratings
    # context_file = args.context

    output_folder_path = trajectories_folder + "_labeled"
    os.makedirs(output_folder_path, exist_ok=True)
    output_control_folder_path = trajectories_folder + "_control_labeled"
    os.makedirs(output_control_folder_path, exist_ok=True)

    control_indices = [3007,2007,1007,7,302,1302,2302,3102,2002,1002,2,3094,2894,1879,834]

    dataset_organization = dict()
    subdirectories = [a for a in os.listdir(trajectories_folder) if os.path.isdir(os.path.join(trajectories_folder,a))]
    for s in subdirectories:
        files = os.listdir(os.path.join(trajectories_folder, s))
        dataset_organization[s] = files
    

    file_list = os.listdir(ratings_dir)
    file_list.sort()
    for filename in file_list:
        if filename.endswith('.json'):
            rating_file = os.path.join(ratings_dir, filename)
            label_dataset(rating_file, trajectories_folder, dataset_organization, control_indices, output_folder_path, output_control_folder_path)

    shutil.copy(os.path.join(trajectories_folder, 'trajectory_variants.json'), output_folder_path)