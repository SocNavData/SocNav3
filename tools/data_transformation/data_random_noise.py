import math
import copy
import random

def transform_with_random_noise(data):
    robot_noise_pos_x = random.gauss(0.,1.)*0.1
    robot_noise_pos_y = random.gauss(0.,1.)*0.1
    robot_noise_angle = random.gauss(0.,1.)*math.pi/90
    person_noise_pos_x = random.gauss(0.,1.)*0.1
    person_noise_pos_y = random.gauss(0.,1.)*0.1
    person_noise_angle = random.gauss(0.,1.)*math.pi/90
    object_noise_pos_x = random.gauss(0.,1.)*0.1
    object_noise_pos_y = random.gauss(0.,1.)*0.1
    object_noise_angle = random.gauss(0.,1.)*math.pi/90
    transformed_data = copy.deepcopy(data)
    for idx, frame in enumerate(transformed_data['sequence']):
        # Transform robot
        frame['robot']['x'] += robot_noise_pos_x
        frame['robot']['y'] += robot_noise_pos_y
        frame['robot']['angle'] += robot_noise_angle

        # Transform people
        for person in frame["people"]:
            person["x"] += person_noise_pos_x
            person["y"] += person_noise_pos_y
            person['angle'] += person_noise_angle

        
        # Transform objects
        for obj in frame["objects"]:
            obj["x"] += object_noise_pos_x
            obj["y"] += object_noise_pos_y
            obj['angle'] += object_noise_angle

    return transformed_data


