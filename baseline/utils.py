import matplotlib.pyplot as plt
import numpy as np


def plot_predictions_vs_expected(predictions, labels):
    """
    Create a line plot comparing predicted vs expected labels, sorted by expected labels.
    Also calculates and prints the Mean Squared Error (MSE) manually.
    
    Args:
        predictions: numpy array of model predictions
        labels: numpy array of true labels
    """
    # Convert to numpy arrays if they aren't already
    predictions = np.array(predictions)
    labels = np.array(labels)
    
    # Sort by expected labels
    sort_idx = np.argsort(labels)
    sorted_labels = labels[sort_idx]
    sorted_predictions = predictions[sort_idx]
    
    # Calculate Mean Squared Error (MSE) and Mean Absolute Error (MAE)
    
    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.plot(sorted_labels, label='Expected', color='blue')
    plt.scatter(range(len(sorted_predictions)), sorted_predictions, label='Predicted', color='red')
    #plt.plot(sorted_predictions, label='Predicted', color='red')
    
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Expected vs Predicted Labels')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    return plt

def plot_qualitative_multiple(predicted_ratings, name, context, color, sample_step = 1., ax=None, indices=None, add_xlabel=True, add_ylabel=True):
    # print(context)
    # if "battery" in context:
    #     bp = context.split("%")[0][-2:]
    #     context = f"{bp} battery"
    # elif "fire" in context:
    #     context = "fire"
    # elif context == "A robot is working with lab samples. The samples contain a deadly virus.":
    #     context = "lab"
    # elif context == "A restaurant robot is looking for a fire extinguisher, as it just detected a fire.":
    #     context = "fire"
    # elif context == "A hospital robot is looking for a fire extinguisher, as it just detected a fire.":
    #     context = "fire"        
    # elif context == "A hospital assistant robot has been asked to go to the goal, with no additional context.":
    #     context = "hospital"
    # elif context == "A delivery robot is navigating in a hospital. It works with fragile objects.":
    #     context = "fragile"
    # elif context == "A robot is navigating as part of a collection task in a library. It works with fragile objects.":
    #     context = "library"
    # elif context == "A museum guide robot has been asked to go to the goal shown, with no additional context.":
    #     context = "museum"      
    # elif context == "A robot is performing routine tasks in a museum.":
    #     context = "museum"      
    # elif  "An idle robot working in a home goes to recharge its battery" in context:
    #     context = "home"
    # elif  "office" in context:
    #     context = "office"
    # elif  "warehouse" in context:
    #     context = "warehouse"
    # elif  "library" in context:
    #     context = "library"

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 7))
    
    if indices is None:
        value = 0.1
        index = 0
        increment = 0.08
        factor = sample_step #1.0
        distances = []
        for i in range(len(predicted_ratings)):
            # print(predicted_rating, filename)
            distances.append(value)
            distances.append(-1*value)
            # print(value)
            value += increment*factor
            # increment *= factor
            index += 2
        # print("Length of distances: ",len(distances))
        distances = distances[:len(distances)//2]
        distances.sort()
    else:
        distances = indices
    
    style =  {'color': color, 'marker': 's', 'linestyle': '-', 'linewidth': 2, 'markersize': 1}

    # Use ax instead of plt
    ax.plot(distances,
            predicted_ratings,
            label=name,
            **style,
            alpha=0.8)
    
    # Use ax methods instead of plt
    # ax.set_title(context, fontsize=16, pad=2)
    ax.set_xlabel("signed distance", fontsize=16)
    extra_ticks = np.arange(-4,4,0.5)
    ax.set_xticks(extra_ticks, minor = True)
    for tick in extra_ticks:
        ax.axvline(x=tick, color='lightgray', linestyle=':', linewidth=0.5)
    ax.set_yticks([0.1,0.3,0.5,0.7,0.9])
    if add_ylabel:
        ax.set_ylabel('score', fontsize=16)
    else:
        # ax.spines['left'].set_visible(False)
        ax.tick_params(axis='y', labelleft=False)  # remove tick labels
        ax.set_ylabel("")                             # remove Y-axis label
        ax.tick_params(axis='y', left=False, labelleft=False, top=False)

    if not add_xlabel:
        ax.tick_params(axis='x', bottom=False, labelbottom=False, top=False)  # remove tick labels
        ax.set_xlabel("")                             # remove Y-axis label

    ax.text(-4, 0.92, "("+context+")", fontsize=16, horizontalalignment='left')


    ax.grid(True, alpha=0.3)
    # ax.legend(loc='upper right', framealpha=0.8, facecolor='white')
   
    # Set y-axis limits between 0 and 1 since ratings are normalized
    ax.set_ylim(0, 1)
   
    return ax

