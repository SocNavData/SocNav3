import math
import copy
import random

def rotate_pose(x, y, a, ref_angle):
    cos_theta = math.cos(ref_angle)
    sin_theta = math.sin(ref_angle)
    
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta
    

    new_angle = math.atan2(math.sin(a + ref_angle), math.cos(a + ref_angle))
    
    return new_x, new_y, new_angle

def rotate_position(x, y, ref_angle):
    cos_theta = math.cos(ref_angle)
    sin_theta = math.sin(ref_angle)
    
    new_x = x * cos_theta - y * sin_theta
    new_y = x * sin_theta + y * cos_theta
   
    return new_x, new_y


def rotate_speed(x_vel, y_vel, ref_angle):
    """
    Transform velocities to goal-relative frame with rotation.
    """
    # Rotate velocities by negative goal angle
    cos_theta = math.cos(ref_angle)
    sin_theta = math.sin(ref_angle)
    
    new_x_vel = x_vel * cos_theta - y_vel * sin_theta
    new_y_vel = x_vel * sin_theta + y_vel * cos_theta
    
    return new_x_vel, new_y_vel


def transform_with_random_orientation(data):
    if data['sequence'][0]['goal']['angle_threshold'] < math.pi:
        return data

    angle = random.uniform(-math.pi, math.pi)
    transformed_data = copy.deepcopy(data)
    for idx, frame in enumerate(transformed_data['sequence']):
        # Transform robot
        rx = frame['robot']['x']
        ry = frame['robot']['y']
        rx_vel = frame['robot']['speed_x']
        ry_vel = frame['robot']['speed_y']
        ra = frame['robot']['angle']
        
        new_rx, new_ry, new_ra = rotate_pose(rx, ry, ra, angle)
        new_rx_vel, new_ry_vel = rotate_speed(rx_vel, ry_vel, angle)


        frame['robot']['x'] = new_rx
        frame['robot']['y'] = new_ry
        frame['robot']['angle'] = new_ra
        frame['robot']['speed_x'] = new_rx_vel
        frame['robot']['speed_y'] = new_ry_vel

        # Transform people
        for person in frame["people"]:
            px = person["x"]
            py = person["y"]
            pa = person['angle']
            
            new_px, new_py, new_pa = rotate_pose(px, py, pa, angle)

            person["x"] = new_px
            person["y"] = new_py
            person['angle'] = new_pa

        
        # Transform objects
        for obj in frame["objects"]:
            ox = obj["x"]
            oy = obj["y"]
            oa = obj['angle']
            
            new_ox, new_oy, new_oa = rotate_pose(ox, oy, oa)
            
            obj["x"] = new_ox
            obj["y"] = new_oy
            obj['angle'] = new_oa
        
    
    # Transform walls
    transformed_walls = []
    if len(data['walls']):
        for wall in data['walls']:
            x1_transformed, y1_transformed = rotate_position(
                wall[0], wall[1], angle
            )
            x2_transformed, y2_transformed = rotate_position(
                wall[2], wall[3], angle
            )
            transformed_walls.append([
                x1_transformed, y1_transformed,
                x2_transformed, y2_transformed
            ])
    transformed_data['walls'] = transformed_walls

    return transformed_data


