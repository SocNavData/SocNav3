from controller import Supervisor

def get_or_create_node(robot_supervisor, name, definition):
    node = robot_supervisor.getFromDef(name)
    if node is None:
        root_node = robot_supervisor.getRoot()
        root_children_field = root_node.getField('children')
        root_children_field.importMFNodeFromString(-1, 'DEF ' + name + ' ' + definition + ' {  }')
        node = robot_supervisor.getFromDef(name)
    return node

def update_node_position(node, x, y, z, angle):
    node.getField("translation").setSFVec3f([x, y, z])
    node.getField("rotation").setSFRotation([0, 0, 1, angle])

def update_node_size(node, size_x, size_y, size_z):
    size = node.getField("size").getSFVec3f()
    size[0] = size_x
    size[1] = size_y
    size[2] = size_z
    node.getField("size").setSFVec3f(size)

def draw_objects(robot_supervisor, point, exclude=[]):
    for obj in point['objects']:
        object_type = obj['type']
        if object_type in exclude:
            continue

        object_id = obj["id"]
        x = obj["x"]
        y = obj["y"]
        angle = obj["angle"]

        if object_type == 'plant':
            name = 'Plant_' + str(object_id)
            node = get_or_create_node(robot_supervisor, name, 'MyPottedTreeV01')
            update_node_position(node, x, y, 0, angle)

        if object_type == 'chair':
            name = 'Chair_' + str(object_id)
            node = get_or_create_node(robot_supervisor, name, 'MyOfficeChair')
            update_node_position(node, x, y, 0, angle)
            size_x = obj['shape']['height']
            size_y = obj['shape']['width']
            update_node_size(node, size_x, size_y, node.getField("size").getSFVec3f()[2])

        if object_type == 'table':
            name = 'Table_' + str(object_id)
            node = get_or_create_node(robot_supervisor, name, 'Table')
            update_node_position(node, x, y, 0, angle)
            size_x = obj['shape']['height']
            size_y = obj['shape']['width']
            size_z = 0.8
            update_node_size(node, size_x, size_y,size_z)

        if object_type == 'laptop':
            name = 'Laptop_' + str(object_id)
            node = get_or_create_node(robot_supervisor, name, 'MyLaptop')
            translation = node.getField('translation').getSFVec3f()
            update_node_position(node, x, y, translation[2], angle)
            
        if object_type == 'shelf':
            name = 'Shelf_' + str(object_id)
            node = get_or_create_node(robot_supervisor, name, 'MyShelves')
            update_node_position(node, x, y, node.getField('translation').getSFVec3f()[2], 3.14 + angle)
            size_x = obj['shape']['height'] / 0.25
            size_y = obj['shape']['width'] / 0.8
            update_node_size(node, size_x, size_y, 0.75)
        
        if object_type == 'TV':
            name = 'TV_' + str(object_id)
            node = get_or_create_node(robot_supervisor, name, 'MyTelevision')
            update_node_position(node, x - 0.2, y - 0.1, 1.2, 3.14 + angle)
            node.getField("controller").setSFString('<none>')
            size_y = obj['size'][0]
            update_node_size(node, node.getField("size").getSFVec3f()[0], size_y, 1 / 0.6)
