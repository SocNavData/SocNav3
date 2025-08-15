import math
import copy

def get_angle_transform(ga, old_pa, dx, dy):
    """
    Transform coordinates to goal-relative frame with rotation.
    
    Args:
        ga: goal angle (in radians)
        old_pa: original angle of entity
        dx: x-coordinate relative to goal position
        dy: y-coordinate relative to goal position
    
    Returns:
        tuple: (transformed_angle, transformed_x, transformed_y)
    """
    # Rotate coordinates around origin by negative goal angle
    cos_theta = math.cos(-ga)
    sin_theta = math.sin(-ga)
    
    # Apply rotation matrix
    new_x = dx * cos_theta - dy * sin_theta
    new_y = dx * sin_theta + dy * cos_theta
    
    # Transform angle relative to goal angle
    # new_angle = old_pa - ga

    new_angle = math.atan2(math.sin(old_pa - ga), math.cos(old_pa - ga))
    
    return new_angle, new_x, new_y

def transform_robot_speed(r_vx, r_vy, ra_to_goal):
    r_vlin = math.sqrt(r_vx**2 + r_vy**2)

    new_r_vx = r_vlin*math.cos(-ra_to_goal)
    new_r_vy = r_vlin*math.sin(-ra_to_goal)

    return new_r_vx, new_r_vy

def get_velocity_transform(x_vel, y_vel, ga):
    """
    Transform velocities to goal-relative frame with rotation.
    """
    # Rotate velocities by negative goal angle
    cos_theta = math.cos(-ga)
    sin_theta = math.sin(-ga)
    
    new_x_vel = x_vel * cos_theta - y_vel * sin_theta
    new_y_vel = x_vel * sin_theta + y_vel * cos_theta
    
    return new_x_vel, new_y_vel

def transform_wall_endpoints(x, y, goal_x, goal_y, goal_a):
    """
    Transform wall endpoints to goal-relative frame with rotation.
    """
    # Translate to goal-relative coordinates
    dx = x - goal_x
    dy = y - goal_y
    
    # Rotate around origin by negative goal angle
    cos_theta = math.cos(-goal_a)
    sin_theta = math.sin(-goal_a)
    
    new_x = dx * cos_theta - dy * sin_theta
    new_y = dx * sin_theta + dy * cos_theta
    
    return new_x, new_y

def transform_to_goal_fr(data):
    """
    Transform all entities' coordinates in the sequence to the goal-based frame of reference.
    """
    transformed_data = {}
    transformed_data['grid'] = copy.deepcopy(data['grid'])
    # grid_x = transformed_data['grid']['x_orig']
    # grid_y = transformed_data['grid']['y_orig']
    # grid_angle = transformed_data['grid']['angle_orig']
    if 'context_description' in data.keys():
        transformed_data['context_description'] = data['context_description']
    if 'label' in data.keys():
        transformed_data['label'] = data['label']
    transformed_sequence = []
    
    assert len(data['sequence']) > 0, "Sequence with no steps."
    for idx, frame in enumerate(data['sequence']):
        transformed_frame = {}
        transformed_frame['timestamp'] = frame['timestamp']
        
        # Get goal position and angle
        gx = frame['goal']['x']
        gy = frame['goal']['y']
        ga = frame['goal']['angle']
        
        # Transform robot
        rx = frame['robot']['x'] - gx
        ry = frame['robot']['y'] - gy
        rx_vel = frame['robot']['speed_x']
        ry_vel = frame['robot']['speed_y']
        old_ra = frame['robot']['angle']
        
        new_ra, rx, ry = get_angle_transform(ga, old_ra, rx, ry)

        # new_rx_vel, new_ry_vel = transform_robot_speed(rx_vel, ry_vel, -new_ra)

        # new_rx_vel, new_ry_vel = get_velocity_transform(rx_vel, ry_vel, -new_ra)

        if idx == 0:
            new_rx_vel = 0
            new_ry_vel = 0
            new_ra_vel = 0
        else:
            diff_time = transformed_frame['timestamp']-transformed_sequence[idx-1]['timestamp']
            new_rx_vel = (rx - transformed_sequence[idx-1]['robot']['x'])/diff_time
            new_ry_vel = (ry - transformed_sequence[idx-1]['robot']['y'])/diff_time
            diff_ra = (new_ra - transformed_sequence[idx-1]['robot']['angle'])
            new_ra_vel = math.atan2(math.sin(diff_ra), math.cos(diff_ra))/diff_time
        
        transformed_robot = {
            "x": rx,
            "y": ry,
            "angle": new_ra,
            "speed_x": new_rx_vel,
            "speed_y": new_ry_vel,
            "speed_a": new_ra_vel, 
            "shape": frame['robot']['shape']
        }
        transformed_frame['robot'] = transformed_robot
        
        # Transform people
        transformed_people = []
        for person in frame["people"]:
            dx = person["x"] - gx
            dy = person["y"] - gy
            old_pa = person['angle']
            
            pa, dx, dy = get_angle_transform(ga, old_pa, dx, dy)
            
            transformed_person = {
                "id": person['id'],
                "x": dx,
                "y": dy,
                "angle": pa
            }
            transformed_people.append(transformed_person)
        transformed_frame['people'] = transformed_people
        
        # Transform objects
        transformed_objects = []
        for obj in frame["objects"]:
            dx = obj["x"] - gx
            dy = obj["y"] - gy
            old_oa = obj['angle']
            
            oa, dx, dy = get_angle_transform(ga, old_oa, dx, dy)
            
            transformed_object = {
                "id": obj['id'],
                "x": dx,
                "y": dy,
                "angle": oa,
                "type": obj['type'],
                "shape": obj['shape']
            }
            transformed_objects.append(transformed_object)
        transformed_frame['objects'] = transformed_objects
        
        # Set goal as origin with zero angle
        transformed_goal = {
            "type": frame['goal']['type'],
            "human": frame['goal']['human'],
            "x": 0,
            "y": 0,
            "angle": 0,
            "pos_threshold": frame['goal']['pos_threshold'],
            "angle_threshold": frame['goal']['angle_threshold']
        }
        transformed_frame['goal'] = transformed_goal
        transformed_sequence.append(transformed_frame)
    
    transformed_data['sequence'] = transformed_sequence
    
    # Transform walls
    transformed_walls = []
    if len(data['walls']):
        for wall in data['walls']:
            x1_transformed, y1_transformed = transform_wall_endpoints(
                wall[0], wall[1], gx, gy, ga
            )
            # print(f"Wall1 before{wall[0], wall[1]} and wall after {x1_transformed, y1_transformed}")
            x2_transformed, y2_transformed = transform_wall_endpoints(
                wall[2], wall[3], gx, gy, ga
            )
            # print(f"Wall2 before{wall[2], wall[3]} and wall after {x2_transformed, y2_transformed}")
            transformed_walls.append([
                x1_transformed, y1_transformed,
                x2_transformed, y2_transformed
            ])
    transformed_data['walls'] = transformed_walls

    return transformed_data


