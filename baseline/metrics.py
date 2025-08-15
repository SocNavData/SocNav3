from shapely.geometry import Point, Polygon, LineString
from shapely.affinity import rotate, translate
import numpy as np

EPS = 0.01

def get_dist_from_obj(object, o_x, o_y, o_angle, robot):
    o_shape = object['shape']['type']
    if o_shape == 'circle':
        o_length = object['shape']['length']
        object_shape = Point(o_x, o_y).buffer(o_length/2)  # o_length is the radius
    elif o_shape == 'rectangle':
        o_length = object['shape']['length']
        o_width = object['shape']['width']
        half_length, half_width = o_length / 2, o_width / 2
        rect = Polygon([
            (-half_length, -half_width),
            (half_length, -half_width),
            (half_length, half_width),
            (-half_length, half_width)
        ])
        # Rotate the rectangle
        rotated_rect = rotate(rect, o_angle, origin=(0, 0), use_radians=True)
        
        # Translate (move) the rectangle to the object's actual position
        object_shape = translate(rotated_rect, xoff=o_x, yoff=o_y)
    else:
        raise ValueError("Invalid object shape. Must be 'circle' or 'rectangle'.")
    distance = robot.distance(object_shape)
    return distance


def get_wall_distance(r_x, r_y, r_radius, w_x1, w_y1, w_x2, w_y2):
    # Define robot as a circle
    robot = Point(r_x, r_y).buffer(r_radius)
    
    # Define wall as a line segment
    wall = LineString([(w_x1, w_y1), (w_x2, w_y2)])
    
    # Compute the minimum distance between the robot's boundary and the wall
    distance = robot.distance(wall)
    
    return round(distance, 2)

def transform_pose(x, y, a, FRx, FRy, FRa): #ga, old_pa, dx, dy):
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
    cos_theta = np.cos(-FRa)
    sin_theta = np.sin(-FRa)
    
    dx = x-FRx
    dy = y-FRy
    # Apply rotation matrix
    new_x = dx * cos_theta - dy * sin_theta
    new_y = dx * sin_theta + dy * cos_theta
    
    # Transform angle relative to goal angle
    new_angle = np.arctan2(np.sin(a - FRa), np.cos(a - FRa))
    
    return new_x, new_y, new_angle 


def get_ttc(cur_frame, prev_frame):
    robot_pose = np.array([cur_frame['robot']['x'], cur_frame['robot']['y']])
    time_diff = cur_frame['timestamp'] - prev_frame['timestamp']
    robot_pose_prev = np.array([prev_frame['robot']['x'], prev_frame['robot']['y']])
    if time_diff>0:
        robot_vel = (robot_pose-robot_pose_prev)/time_diff
    else:
        robot_vel = np.array([0., 0.])
    # robot_vel = np.array([cur_frame['robot']['speed_x'], cur_frame['robot']['speed_y']])
    human_radius = 0.3 ## Let's assume human radius is 0.3
    length =  cur_frame['robot']['shape']['length']
    width =  cur_frame['robot']['shape']['width']
    robot_radius = np.linalg.norm([length, width])/2
    
    radii_sum = human_radius + robot_radius ## Sum of human and robot radii
    radii_sum_sq = radii_sum * radii_sum
    
    calc_metrics = []
    for human in cur_frame['people']:
        current_metrics = {}        
        ttc = -1
        cost_panic = -1
        cost_fear = -1
        C = np.array([human['x'], human['y']]) - robot_pose # Difference between centers
        C_sq = C.dot(C)
        
        human_vel = np.array([0., 0.])
        for prev_human in prev_frame['people']:
            if prev_human['id'] == human['id']:
                pose_diff = np.array([human['x'] - prev_human['x'], human['y'] - prev_human['y']])
                if time_diff>0:
                    human_vel = pose_diff/time_diff
                break
        
        if C_sq < radii_sum_sq:
            ttc = 0                         ## Human and robot are already in Collision
        else:
            V = robot_vel - human_vel       ## Difference between human and robot velocities
            C_dot_V = C.dot(V)              ## Dot product between the vectors
            if C_dot_V > 0:
                V_sq = V.dot(V)
                f = (C_dot_V * C_dot_V) - (V_sq * (C_sq - radii_sum_sq))
                # print(f"C_dot_v :{C_dot_V}, f :{f}, V_sq :{V_sq}")
                if f > 0:
                    ttc = (C_dot_V - np.sqrt(f)) / V_sq
                else:
                    g = np.sqrt(V_sq * C_sq - C_dot_V * C_dot_V)
                    if((g - (np.sqrt(V_sq) * radii_sum)) > EPS):
                        cost_panic = np.sqrt(V_sq / C_sq) * (g / (g - (np.sqrt(V_sq) * radii_sum))) ## Panic cost
        
        if ttc > EPS:
            cost_fear = 1.0/ttc
        elif ttc>=0:
            cost_fear = 10.
                                                                            ## fear cost
            
        current_metrics['id'] = human['id']            
        current_metrics['ttc'] = ttc
        current_metrics['fear'] = cost_fear
        current_metrics['panic'] = cost_panic
        calc_metrics.append(current_metrics)       
        
    return calc_metrics
