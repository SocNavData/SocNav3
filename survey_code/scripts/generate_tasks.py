# cd dev/llama.cpp/build
# bin/llama-server --hf-repo hugging-quants/Llama-3.2-3B-Instruct-Q8_0-GGUF --hf-file llama-3.2-3b-instruct-q8_0.gguf -c 2048 --api-key LOL --temp 0.15

import sys
import time
import random

import numpy as np
from tqdm import tqdm

import matplotlib.pyplot as plt

from openai import OpenAI

bYOUR_API_KEY = "LOL"
llama_client = OpenAI(api_key=bYOUR_API_KEY, base_url="http://127.0.0.1:8080")

messages = [
    {
        "role": "user",
        "content": (""),
    },
]


def to_float(string):
    ret = ''.join([s for s in string if s in "1234567890"])
    return float(ret)

def generate_descriptions(how_many=500):
    PLACES = ["a hospital",
              "a restaurant",
              "a school",
              "a home",
              "a library",
              "a store",
              "a museum",
              "a warehouse"]

    descriptions = []

    def bored_task():
        place = random.choice(PLACES).split()[1]
        return f"A robot is moving in {place} to find a user who may want to assign it to a task."
    def going_task():
        place = random.choice(PLACES)
        return f"A robot is performing routine tasks in {place}."
    def delivering_task():
        place = random.choice(PLACES)
        add = " It works with fragile objects." if random.randint(0,1) else ""
        return f"A delivery robot is navigating in {place}." + add
    def collecting_task():
        place = random.choice(PLACES)
        add = " It works with fragile objects." if random.randint(0,1) else ""
        return f"A robot is navigating as part of a collection task in {place}." + add
    def exploring_task():
        place = random.choice(PLACES)
        return f"An idle robot working in {place} explores its environment."
    def cleaning_task():
        place = random.choice(PLACES)
        return f"A cleaning robot working in {place} is looking for dirty spots to clean."
    def recharging_task():
        place = random.choice(PLACES)
        level = random.choice([x for x in range(95)])
        return f"An idle robot working in {place} goes to recharge its battery. It has {level}% battery left."
    def lab_samples_task():
        whatever = random.choice(["contain a deadly virus",
                                  "contain blood samples",
                                  "are part of a routine environmental study to measure pollution"])
        return f"A robot is working with lab samples. The samples {whatever}."
    def universe_task():
        place = random.choice(PLACES).split()[1]
        return f"A {place} robot is looking for a fire extinguisher, as it just detected a fire."

    types = [bored_task, going_task, delivering_task, collecting_task, exploring_task, cleaning_task,
             recharging_task, lab_samples_task]*3 + [universe_task]

    if how_many > 0:
        while len(descriptions) < how_many:
            descriptions.append(random.choice(types)())
    elif how_many < 0:
        RESET_PATIENCE = 1000
        all = set([])
        for function in types:
            print(function)
            these = set([])
            patience = RESET_PATIENCE
            while patience > 0:
                sample = function()
                if sample not in these:
                    these.add(sample)
                    patience = RESET_PATIENCE
                else:
                    patience -= 1
            all = all.union(these)
        descriptions = [x for x in all]
    else:
        print("I cannot generate 0...")
        sys.exit(-1)

    return descriptions



contexts = generate_descriptions(how_many=-1)
contexts = sorted(contexts)

fd = open("all_contexts.txt", "w")
for context in contexts:
    fd.write(context+"\n")
fd.close()

number_of_queries = len(contexts)*3*4 # 3 for the 3 approaches, 4 for the 4 variables

progress_bar = tqdm(total=number_of_queries)

groups = {}
for c in contexts:
    g = c[0]
    text = c[2:]
    try:
        groups[g].append(text)
    except:
        groups[g] = [text]


prompt_values = [
    "the urgency of the task",
    "the importance of the task",
    "the risk involved in the task",
    "the maximum speed the robot should operate",
]
for prompt_value in prompt_values:
    print("=====================================================")
    print("====  M A I N    P R O M P T  =======================")
    print("=====================================================")
    prompt = f"I will give a task description for a robot. I want you to reply with the percentile (a number from 0 to 100) " + \
              "that corresponds to {prompt_value} in comparison with that of other tasks you could imagine. I don't want an " + \
              "explanation, only the percentile. The task is: "
    print(prompt)
    # prompt = "I will give a task description for a robot. I want you to reply with a number from 0 to 100 " + \
    #         "that corresponds to {value} and only a number. I don't want an explanation. " + \
    #         "The task is: "
    print("=====================================================")
    print("======  T A S K S   =================================")
    print("=====================================================")

    urgency = np.zeros((3,101))

    # for index, client in enumerate([ppx_client, llama_client, None]):
    if True:
        index = 1
        client = llama_client
        match index:
            case 0:
                approach = "perplexity"
            case 1:
                approach = "llama"
            case 2:
                approach = "average"
            case _:
                print("???kie")
                sys.exit(1)

        fname = prompt_value+"_"+approach+".txt"
        fd = open(fname, "w")

        for context_a in contexts:
            context = context_a[2:]
            print("===========================")
            print(context)
            sum = 0
            r = []
            for i in range(2):
                messages[0]["content"] = prompt + context
                # What LLM to use? In the first two cases, we'll use a specific one.
                # Otherwise, we'll average the two.
                if index < 2:
                    choice = index
                else:
                    choice = i

                if choice == 0:
                    done = False
                    while done is False:
                        try:
                            response = ppx_client.chat.completions.create(model="llama-3.1-sonar-large-128k-chat", messages=messages,)
                            time.sleep(1)
                            done = True
                        except Exception as e:
                            print("Rate exceeded", e)
                            time.sleep(15)
                else:
                    response = llama_client.chat.completions.create(model="llama-3.2-3b-instruct-q8_0", messages=messages,)

                value = to_float(response.choices[0].message.content)
                r.append(value)
                sum += value
                if index == 0:
                    break # No need to go for a second, as Perplexity will always give the same value
                        
            average = sum/len(r)
            print(r, average)

            if average < 0:
                average = 0
            elif average > 100:
                average = 100

            fd.write(f"{context}: {average}\n")
            urgency[index, int(average)] += 1

            progress_bar.update(1)

        fd.close()

    print(urgency.tolist())

    bin_edges = np.linspace(0, 100, 102)

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], urgency[0,:], width=np.diff(bin_edges), align='edge', edgecolor='black')
    plt.title(f'{prompt_value} histogram: perplexity')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'{prompt_value}_perplexity.png')

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], urgency[1,:], width=np.diff(bin_edges), align='edge', edgecolor='black')
    plt.title(f'{prompt_value} histogram: llama.cpp')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'{prompt_value}_llama.png')

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    plt.bar(bin_edges[:-1], urgency[2,:], width=np.diff(bin_edges), align='edge', edgecolor='black')
    plt.title(f'{prompt_value} histogram: perplexity / llama.cpp')
    plt.xlabel('value')
    plt.ylabel('frequency')
    plt.grid(axis='y', alpha=0.75)
    plt.savefig(f'{prompt_value}_average.png')

    # # Show the plot
    # plt.show()

    # Close the plot to prevent it from displaying
    plt.close()



