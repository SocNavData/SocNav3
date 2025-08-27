import os
import numpy as np
import shutil

from sklearn.metrics import cohen_kappa_score
from check_quality import get_responses, filter_out_invalid_responses, compute_cohens, \
                            MIN_RELEVANT, NUM_CLASSES, LABELS
import argparse


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
    parser = argparse.ArgumentParser(prog='select_valid_raters',
                        description='Selects a set of valid raters according to a consistency analysis')
    parser.add_argument('--ratings_dir', type=str, nargs="?", required = True, help="Directory containing the whole set of raters.")
    parser.add_argument('--output_dir', type=str, nargs="?", required = True,  help="Output directory.")
    
    args = parser.parse_args()

    JSON_DIR = args.ratings_dir
    OUTPUT_DIR = args.output_dir

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


