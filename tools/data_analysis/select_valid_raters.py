import os
import sys
import json

from collections import namedtuple
import numpy as np
import shutil

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import cohen_kappa_score


sys.stderr = sys.stdout

Response = namedtuple('Response', ['id', 'keys', 'values', 'rep_keys', 'rep_values'])

if len(sys.argv)<3:
    print("Please provide source and output directories")
JSON_DIR = sys.argv[1]
OUTPUT_DIR = sys.argv[2]

MIN_RELEVANT = 15
RELEVANT = [
    [7,  3007, "A robot is trying to locate the source of a noise in a library."],
    [11, 2007, "A robot is navigating as part of a delivery task in a museum."],
    [13, 1007, "An office assistant robot keeps track of who is in the office today."],
    [17, 7,    "A hotel robot is inspecting the floor to ensure it's safe to walk."],
    [19, 302,  "A drug delivery robot is working in a hospital."],
    [23, 1302, "A museum robot roams around looking for people interested in its services."],
    [31, 2302, "A robot is performing routine tasks in an office."],
    [53, 3102, "A museum guide robot has been asked to go to the goal shown, with no additional context."],
    [59, 2002, "A lab assistant robot is looking for potential hazards in its environment."],
    [61, 1002, "A cleaning robot working in a hospital is looking for dirty spots to clean."],
    [67, 2,    "The robot is trying to locate the glasses of a patient in a hospital."],
    [71, 3094, "A hospital assistant robot has been asked to go to the goal, with no additional context."],
    [73, 2894, "An idle robot working in a museum goes to recharge its battery. It has 13% battery left."],
    [79, 1879, "A assistant robot is performing routine tasks in a restaurant."],
    [83, 834,  "A warehouse robot is moving around while inspecting the air quality."],
    ]

MIN_REPS = 5
REPS = [
    [29, 7],
    [37, 11],
    [41, 13],
    [43, 17],
    [47, 19],
]


def is_sorted(lst):
    return lst == sorted(lst)

def firsts(t):
    return [e[0] for e in t]

def get_responses(source_dir):
    total_surveys = 0
    total_answers = 0

    responses = []

    # Iterate over files in the directory
    file_list = os.listdir(source_dir)
    file_list.sort()
    for filename in file_list:
        if filename.endswith('.json'):
            file_path = os.path.join(source_dir, filename)
            with open(file_path, 'r') as f:
                keys = []
                values = []
                rep_keys = []
                rep_values = []
                try:
                    text = ''.join(f.readlines())
                    text = text.replace("\n", "")
                    data = json.loads(text)
                    for k, v in data["answers"].items():
                        if int(k) in firsts(RELEVANT):
                            keys.append(int(k))
                            values.append(v)
                        if int(k) in firsts(REPS):
                            rep_keys.append(int(k))
                            rep_values.append(v)
                    participant_id = file_path.split("/")[-1].split(".")[0]
                    r = Response(id=participant_id, keys=keys, values=values, rep_keys=rep_keys, rep_values=rep_values)
                    responses.append(r)
                    total_surveys += 1
                    total_answers += len(data["answers"])
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in file: {filename}")

    # Now json_data contains the loaded data from all JSON files
    print(f"We have {total_surveys} surveys, containing {total_answers} answers.")
    return responses


def filter_out_invalid_responses(responses):
    good_responses = []
    keys = None
    for r in responses:

        if len(r.keys)<MIN_RELEVANT:
            print(f"Didn't answer all control ({r.id}) {len(r.keys)=}.")
            continue
        if len(r.rep_keys)<MIN_REPS:
            print(f"Didn't answer all rep control ({r.id}).")
            continue
     
        if is_sorted(r.keys) is False:
            print("KEYS ARE NOT SORTED???")
            sys.exit(-1)

        if is_sorted(r.rep_keys) is False:
            print("REP EYS ARE NOT SORTED???")

        if keys is None:
            print(f"Setting keys: {r.keys[:MIN_RELEVANT]}")
            keys = r.keys[:MIN_RELEVANT]
        if r.keys[:MIN_RELEVANT] != keys[:MIN_RELEVANT]:
            print(f"Key lists do not coincide: \n{r.keys[:MIN_RELEVANT]}\n{keys[:MIN_RELEVANT]}")
            sys.exit(-1)

        good_responses.append(r)
    return good_responses


def compute_consistency_matrix_with_intra(valid_responses):
    """
    Compute a consistency matrix with intra-rater consistency included.

    Args:
        valid_responses (list): List of valid `Response` objects.

    Returns:
        np.ndarray: Consistency matrix with intra-rater consistency on the diagonal.
    """
    # Collect responses to relevant questions
    responses_matrix = np.array([r.values[:MIN_RELEVANT] for r in valid_responses])
    rep_responses_matrix = np.array([r.rep_values for r in valid_responses])
    # Normalize responses for comparison

    responses_matrix = (responses_matrix - np.min(responses_matrix)) / (np.max(responses_matrix) - np.min(responses_matrix))
   

    num_participants = len(valid_responses)
    consistency_matrix = np.zeros((num_participants, num_participants))

    # Compute pairwise consistency (Euclidean distance)
    for i in range(num_participants):
        for j in range(num_participants):
            consistency_matrix[i, j] = np.linalg.norm(responses_matrix[i] - responses_matrix[j])

    # Add intra-rater consistency to the diagonal
    for i in range(num_participants):
        intra_rater_variance = np.var(rep_responses_matrix[i])  # Variance of repeated responses
        consistency_matrix[i, i] = intra_rater_variance

    return consistency_matrix



NUM_CLASSES = 11  # Adjust as needed
LABELS = [i for i in range(NUM_CLASSES)]
intra_consistency = []
def compute_cohens(valid_responses, only_intra = False):
    """
    Compute a consistency matrix with intra-rater consistency included using Cohen's kappa.

    Args:
        valid_responses (list): List of valid `Response` objects.

    Returns:
        np.ndarray: Consistency matrix with intra-rater consistency on the diagonal.
    """
    # Convert numerical responses to categorical classes
    responses_matrix = np.array([r.values[:MIN_RELEVANT] for r in valid_responses])
    rep_responses_matrix = np.array([r.rep_values for r in valid_responses])
    
    # Normalize responses
    # responses_matrix = (responses_matrix - np.min(responses_matrix)) / (np.max(responses_matrix) - np.min(responses_matrix))

    # Convert to categorical labels
    categorical_responses = np.round(responses_matrix * (NUM_CLASSES - 1)).astype(int)
    # print(categorical_responses)
    num_participants = len(valid_responses)
    consistency_matrix = np.zeros((num_participants, num_participants))

    # Compute pairwise consistency using Cohen's kappa
    for i in range(num_participants):
        for j in range(num_participants):
            # if i==25 or i==48 or j==48 or j==25:
            #     print(i,j)
            #     print(categorical_responses[i])
            #     print(categorical_responses[j])
            consistency_matrix[i, j] = cohen_kappa_score(categorical_responses[i], categorical_responses[j], labels=LABELS, weights="quadratic")

    # Compute intra-rater consistency using Cohen's kappa on repeated responses
    for i in range(num_participants):
        rep_categorical = np.round(rep_responses_matrix[i] * (NUM_CLASSES - 1)).astype(int)
        
        # Ensure both arrays have the same length
        min_length = min(len(rep_categorical), len(categorical_responses[i]))
        # print("Participant", valid_responses[i].id)
        # print("First Preds", rep_categorical)
        rep_categorical = rep_categorical[:min_length]
        main_categorical = categorical_responses[i][:min_length]
        # print("Repeated Preds", main_categorical)

        # Compute Cohen's kappa
        if len(rep_categorical) > 1:
            consistency_matrix[i, i] = cohen_kappa_score(rep_categorical, main_categorical, labels=LABELS, weights="quadratic")
            # print('Participant', i+1, '---------')
            # print(main_categorical)
            # print(rep_categorical)
            # print('consistency', consistency_matrix[i, i])
            intra_consistency.append(consistency_matrix[i, i])
        else:
            consistency_matrix[i, i] = 0  # Perfect agreement if only one repeated response
    if only_intra:
        return intra_consistency
    else:
        return consistency_matrix

def select_reliable_raters(responses, consistency_matrix):
    num_participants = len(responses)
    reliable_raters = []
    for i in range(num_participants):
        if consistency_matrix[i,i]>0.4:
            rater = (responses[i].id, i)
            reliable_raters.append(rater)
    print(reliable_raters)
    reliable_values = []
    for _, i in reliable_raters:
        values = responses[i].values[:MIN_RELEVANT]
        reliable_values.append(np.array(values))
    np_values = np.array(reliable_values)
    mean_values = np.mean(np_values, axis=0)
    responses_matrix = np.array([r.values[:MIN_RELEVANT] for r in responses])
    categorical_responses = np.round(responses_matrix * (NUM_CLASSES - 1)).astype(int)
    categorical_mean = np.round(mean_values * (NUM_CLASSES - 1)).astype(int)    
    for i in range(num_participants):
        if consistency_matrix[i,i]>0.1:
            consistency = cohen_kappa_score(categorical_responses[i], categorical_mean, labels=LABELS, weights="quadratic")
            if consistency>0.2:
                print('RELIABLE RATER', responses[i].id)
                src = os.path.join(JSON_DIR, responses[i].id+'.json')
                dst = os.path.join(OUTPUT_DIR, responses[i].id+'.json')
                shutil.copyfile(src, dst)




if __name__ == "__main__":
    responses = get_responses(JSON_DIR)
    valid_responses = filter_out_invalid_responses(responses)
    print(f"We have {len(valid_responses)} valid responses.")

    if os.path.exists(OUTPUT_DIR):
        if len(os.listdir(OUTPUT_DIR))!=0:
            print(f'The directory {OUTPUT_DIR} is not empty.')
            r = input('Do you want to continue? (Y/n)')
            if r!= '' and r[0].lower() != 'y':
                print('exiting')
                exit()
    else:
        os.mkdir(OUTPUT_DIR)

    # Compute and plot consistency matrix with intra-rater consistency
    consistency_matrix = compute_cohens(valid_responses, only_intra= False)
    select_reliable_raters(valid_responses, consistency_matrix)


