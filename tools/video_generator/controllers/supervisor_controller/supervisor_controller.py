from controller import Supervisor
import math,json,sys, os, draw_objects
import numpy as np
import cv2


# Specify the folder containing the JSON files:
folder_path = '/home/XXX/dev/SocNav-simdata-generator/jsons'

file_names = os.listdir(folder_path)
json_files = [file for file in file_names if file.endswith('.json')]

# Avoid redrwaing the walls
wallPainted = False

# Video parameters
record_video = True
width = 1000
height = 1000
codec = 'ignored'
quality = 100
acceleration = 1
caption = False

# Create Supervisor instance
robot_supervisor = Supervisor()  
timestep = int(robot_supervisor.getBasicTimeStep())
robot_supervisor.step(timestep)
 
previous_timestamp=0

# Get the robot's node
robot_node = robot_supervisor.getFromDef('Tiago')
scale_node = robot_supervisor.getFromDef('SCALE')

# viewpoint_node = robot_supervisor.getFromDef('Viewpoint')
# New axis: [-0.45424464 -0.77276739  0.4432746 ] #New angle (radians): 4.461136822285801
# viewpoint_node.getField("orientation").setSFRotation([-0.45424464, -0.77276739, 0.4432746,4.461136822285801])


# Returns False, unless the name provided contains any of the provided patterns
def is_pattern(name, patterns):
    for pattern in patterns:
        if name.startswith(pattern) and name[len(pattern):].isdigit():
            return True
    return False

# Removes nodes with a specific pattern
def delete_nodes_by_pattern(robot, patterns):
    root_node = robot.getRoot()
    children_field = root_node.getField('children')
    num_children = children_field.getCount()

    i = 0
    while i < num_children:
        node = children_field.getMFNode(i)
        if node:
            def_name = node.getDef()
            if def_name and is_pattern(def_name, patterns):
                node.remove()
                # After removing a node, the list is not updated, so we don't increase `i`
                num_children -= 1
            else:
                i += 1
        else:
            i += 1

# HUMANS
shirtColors = [
    [0.0, 1.0, 0.0],  # green
    [1.0, 1.0, 0.0],  # yellow
    [1.0, 0.0, 0.0],  # red
    [0.0, 0.0, 1.0],  # blue
    [1.0, 0.5, 0.0],  # orange
    [0.5, 0.0, 0.5],  # purple
    [1.0, 0.0, 1.0],  # pink
    [0.0, 1.0, 1.0]   # cyan
]

# Updates pedestrians
def procesar_personas(point, robot_supervisor):
    rootNode = robot_supervisor.getRoot()  # get root of the scene tree
    rootChildrenField = rootNode.getField('children')
    
    # List JSON ids
    ids_actuales = [person["id"] for person in point['people']]

    for person in point['people']:
        person_id = person["id"]
        human_name = 'Human_' + str(person_id)
        human_node = robot_supervisor.getFromDef(human_name)
       
        if person_id not in pedestrians_id_list:
            # If the human does not exist, we need to create a new one
            if human_node is None:
                global insertado
                insertado =True
                global cont
                cont =0
                rootChildrenField.importMFNodeFromString(-1, 'DEF ' + human_name + ' Pedestrian {  }')
                # After creating the node, we fetch the corresponding object
                human_node = robot_supervisor.getFromDef(human_name)
                human_node.getField("shirtColor").setSFColor (shirtColors[person_id%8])

                
            # Add the id to the list of processed pedestrians
            pedestrians_id_list.append(person_id)

        # If it's in the list, we only need to update its pose
        else:
            x = person["x"]
            y = person["y"]
            angle = person["angle"]
            z= human_node.getField('translation').getSFVec3f()[2]
            human_node.getField("translation").setSFVec3f([x, y, z])
            human_node.getField("rotation").setSFRotation([0, 0, 1, angle])

        for id in set(pedestrians_id_list) - set(ids_actuales):
            human_name = 'Human_' + str(id)
            human_node = robot_supervisor.getFromDef(human_name)
            human_node.getField("translation").setSFVec3f([0,0,-10])
            
            
# Draws walls, only once per scenario
def draw_walls(robot_supervisor, walls, painted):    
    if painted:
        return True  # Return if they have already been drawn
    
    wall_id = 1
    for wall in walls:
        wall_name = 'Wall_' + str(wall_id)
        x1, y1, x2, y2 = wall
        distance = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)        
        half_point = ((x1 + x2) / 2, (y1 + y2) / 2)    
        angle = math.atan2(y2 - y1, x2 - x1)
        
        root_node = robot_supervisor.getRoot()
        root_children_field = root_node.getField('children')
        wall_node = robot_supervisor.getFromDef(wall_name)
        
        if wall_node is None:
            root_children_field.importMFNodeFromString(-1, 'DEF ' + wall_name + ' Wall {  }')
            wall_node = robot_supervisor.getFromDef(wall_name)
        
        wall_node.getField("translation").setSFVec3f([half_point[0], half_point[1], 0])
        wall_node.getField("rotation").setSFRotation([0, 0, 1, angle])
        
        size = wall_node.getField("size").getSFVec3f()
        size[0] = distance
        size[1] = 0.1
        size[2] = 2.4
        wall_node.getField("size").setSFVec3f(size)
        
        wall_id += 1
    
    return True

# Draws the goal for the first num_frames, frames
def draw_goal(robot_supervisor, point, i, num_frames):
    if i < num_frames:  
        return
    
    goal_x = point['goal']['x']
    goal_y = point['goal']['y']
    goal_angle = point['goal']['angle']
    target_node = robot_supervisor.getFromDef('Target')
    
    target_v3 = target_node.getField('translation').getSFVec3f()
    target_v3[0] = goal_x
    target_v3[1] = goal_y
    target_node.getField('translation').setSFVec3f(target_v3)
    target_node.getField("rotation").setSFRotation([0, 0, 1, goal_angle])

    # Set the size of the target according to the size of the robot and the
    # threshold in the position.
    target_radius = point['goal']['pos_threshold'] + point['robot']['shape']['width']/2
    target_node.getField("scale").setSFVec3f([target_radius, target_radius, 1])    

    # Set the texture of the cylinder, so that we can see the
    # angular threshold.
    texture = robot_supervisor.getFromDef("GOALTEXTURE")
    if texture is None:
        pass
    url = texture.getField("url")
    if url is None:
        pass
    v = point['goal']['angle_threshold']*180./np.pi/2
    v_int = int(v)
    image_path = f"goal_{v_int}.png"
    image = np.zeros((256,256,3), dtype=np.uint8)*255
    image[:,:,2] = 255
    cv2.ellipse(image, (128,128), (128,128), 0, -v, v, (0,255,0), thickness=-1)
    cv2.imwrite("../../worlds/"+image_path, image)
    url.setMFString(0, image_path)

# al iniciar a veces se queda morralla. Si lo haces a mano la ejecucion pasa pa paras.
#limpiar todo y listo.
patterns = ['Human_', 'Chair_', 'Table_']
delete_nodes_by_pattern(robot_supervisor, patterns)

# Read and process all JSON files
total_files = len(json_files)
file_index = 0
for file in json_files:
    file_path = os.path.join(folder_path, file)
    with open(file_path) as f:
        data = json.load(f)
    print (file_path)
    file_index += 1
    print(f"Processing file {file_index} of {total_files}: {file}")
    if record_video is True: 
        videoFile = os.path.splitext(file)[0]
        robot_supervisor.movieStartRecording(videoFile+'.mp4', width, height, codec, quality, acceleration, caption)
    else:
        print("Video recording is DEACTIVATED (record_video!=True)")
        pass

    i=0
    pedestrians_id_list = []
    previous_timestamp = int(data['sequence'][0]['timestamp']*1000)
    for point in data['sequence']:
        current_timestamp= int (point['timestamp']*1000)
        timestep=time_elapsed = current_timestamp - previous_timestamp
        previous_timestamp = current_timestamp
        #DRAW WALLS
        wallPainted=draw_walls(robot_supervisor, data['walls'], wallPainted)  
        #Draw GOAL
        num_frames=0
        draw_goal(robot_supervisor, point, i, num_frames) 
        #HUMANS
        procesar_personas(point, robot_supervisor) 
        #DRAW ROBOT       
        if point['robot']['x'] is None or point['robot']['y'] is None or point['robot']['angle'] is None:
            continue
        x = point['robot']['x']
        y = point['robot']['y']
        angle = point['robot']['angle']
        
        if point['robot']['shape']['type'] == 'circle':
            tiago_radius = 0.54/2.0
            target_radius = point['robot']['shape']['width']/2
            scale = target_radius/tiago_radius
            scale_node.getField('scale').setSFVec3f([scale, scale, scale])
        else:
            sys.exit(0)

        robot_v3=robot_node.getField('translation').getSFVec3f()
        robot_v3[0]=x
        robot_v3[1]=y
    
        
        robot_node.getField("translation").setSFVec3f(robot_v3)
        robot_node.getField("rotation").setSFRotation([0, 0, 1, angle])          
        
        excludeObjects = ['shelf', 'TV']
        draw_objects.draw_objects(robot_supervisor, point, excludeObjects)
        
        i += 1
        robot_supervisor.step(timestep)
        
        # Prevent axis from being displayed
        viewPoint_node = robot_supervisor.getFromDef('Viewpoint')
        
    delete_nodes_by_pattern(robot_supervisor, patterns) 
    
    # Stop recording before finishing if it's enabled
    if record_video:
        robot_supervisor.movieStopRecording()
        while not robot_supervisor.movieIsReady():
            robot_supervisor.step(timestep)
    
# Pause simulation
robot_supervisor.simulationSetMode(Supervisor.SIMULATION_MODE_PAUSE)

