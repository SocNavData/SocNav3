import js
import random

MAX_TASKS = 300

def get_tasks_and_probabilities():
    with open("all_contexts.txt", "r") as fd:
        lines = fd.readlines()

    tasks = []
    dict_counts = { "assign":0, "battery":0, "routine":0, "delivery":0, "collection":0, "explore":0, "clean":0, "lab":0, "fire":0 }

    for line in lines:
        if "assign" in line:
            dict_counts["assign"] += 1
        elif "battery" in line:
            dict_counts["battery"] += 1
        elif "routine tasks" in line:
            dict_counts["routine"] += 1
        elif "delivery" in line:
            dict_counts["delivery"] += 1
        elif "collection" in line:
            dict_counts["collection"] += 1
        elif "explores" in line:
            dict_counts["explore"] += 1
        elif "clean" in line:
            dict_counts["clean"] += 1
        elif "lab" in line:
            dict_counts["lab"] += 1
        elif "fire" in line:
            dict_counts["fire"] += 1
        else:
            print(line)
            break
        tasks.append(line)

    # Define the categories and their corresponding values
    categories = []
    counts = []
    for k,v in dict_counts.items():
        categories.append(k)
        counts.append(v)
        js.console.log(f"{k} --> {v}")
    

    probabilities = [ 1./v for v in counts ]
    probabilities[categories.index("lab")] /= 4   # Lower these manually
    probabilities[categories.index("fire")] /= 4  # Lower these manually

    for i in range(len(probabilities)):
        js.console.log(f"{categories[i]} --> {probabilities[i]}")
    return tasks, categories, probabilities


def should_we_accept(line, categories, probabilities):
    if "assign" in line:
        category = "assign"
    elif "battery" in line:
        category = "battery"
    elif "routine" in line:
        category = "routine"
    elif "delivery" in line:
        category = "delivery"
    elif "collection" in line:
        category = "collection"
    elif "explores" in line:
        category = "explore"
    elif "clean" in line:
        category = "clean"
    elif "lab" in line:
        category = "lab"
    elif "fire" in line:
        category = "fire"
    else:
        js.console.log(f"Unknown category for task: {line}")
        return False

    index = categories.index(category)
    probability = probabilities[index]

    sample = random.random()
 
    if probability >= sample:
        return True
    else:
        return False

def generate_descriptions():
    tasks, categories, probabilities = get_tasks_and_probabilities()

    descriptions = []
    while len(descriptions) < MAX_TASKS+5:
        accepted = False
        while accepted is False:
            task = random.randint(0, len(tasks)-1)
            accepted = should_we_accept(tasks[task], categories, probabilities)
        descriptions.append(tasks[task])

    return descriptions



def fix_fixed_tasks(structure):
    #  1 [ R E P E A T E D --  7]
    structure["indices"][7] = 3007
    structure["descriptions"][7] = "A robot is trying to locate the source of a noise in a library."
    #  2 [ R E P E A T E D --  9]
    structure["indices"][11] = 2007
    structure["descriptions"][11] = "A robot is navigating as part of a delivery task in a museum."
    #  3 [ R E P E A T E D -- 10]
    structure["indices"][13] = 1007
    structure["descriptions"][13] = "An office assistant robot keeps track of who is in the office today."
    #  4 [ R E P E A T E D -- 11]
    structure["indices"][17] = 7
    structure["descriptions"][17] = "A hotel robot is inspecting the floor to ensure it's safe to walk."
    #  5 [ R E P E A T E D -- 12]
    structure["indices"][19] = 302
    structure["descriptions"][19] = "A drug delivery robot is working in a hospital."
    #  6
    structure["indices"][23] = 1302
    structure["descriptions"][23] = "A museum robot roams around looking for people interested in its services."
    #  7
    structure["indices"][29] = structure["indices"][7]
    structure["descriptions"][29] = structure["descriptions"][7]
    #  8
    structure["indices"][31] = 2302
    structure["descriptions"][31] = "A robot is performing routine tasks in an office."
    #  9
    structure["indices"][37] = structure["indices"][11]
    structure["descriptions"][37] = structure["descriptions"][11]
    # 10
    structure["indices"][41] = structure["indices"][13]
    structure["descriptions"][41] = structure["descriptions"][13]
    # 11
    structure["indices"][43] = structure["indices"][17]
    structure["descriptions"][43] = structure["descriptions"][17]
    # 12
    structure["indices"][47] = structure["indices"][19]
    structure["descriptions"][47] = structure["descriptions"][19]
    # 13
    structure["indices"][53] = 3102
    structure["descriptions"][53] = "A museum guide robot has been asked to go to the goal shown, with no additional context."
    # 14
    structure["indices"][59] = 2002
    structure["descriptions"][59] = "A lab assistant robot is looking for potential hazards in its environment."
    # 15
    structure["indices"][61] = 1002
    structure["descriptions"][61] = "A cleaning robot working in a hospital is looking for dirty spots to clean."
    # 16
    structure["indices"][67] = 2
    structure["descriptions"][67] = "The robot is trying to locate the glasses of a patient in a hospital."
    # 17
    structure["indices"][71] = 3094
    structure["descriptions"][71] = "A hospital assistant robot has been asked to go to the goal, with no additional context."
    # 18
    structure["indices"][73] = 2894
    structure["descriptions"][73] = "An idle robot working in a museum goes to recharge its battery. It has 13% battery left."
    # 19
    structure["indices"][79] = 1879
    structure["descriptions"][79] = "A assistant robot is performing routine tasks in a restaurant."
    # 20
    structure["indices"][83] = 834
    structure["descriptions"][83] = "A warehouse robot is moving around while inspecting the air quality."

