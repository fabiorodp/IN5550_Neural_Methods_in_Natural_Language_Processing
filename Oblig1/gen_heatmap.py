import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

# get local path
DIR, FILE = os.path.split(os.path.abspath(__file__))

# Static plot titles
htm_titles = [  "Accuracy Score Heatmap",
                "Macro Precision Score Heatmap",
                "Macro Recall Score Heatmap",
                "Macro F1 Score Heatmap",
                "Training elapsed time in seconds"]
def heatmap(htm_arrays, x_title, y_title):
    for array, title in zip(htm_arrays, htm_titles):
        sns.heatmap(
            data=array,
            annot=True,
            fmt=".4g",
            # xticklabels=out_actFcts,
            # yticklabels=hl_actFcts
        )
        plt.title(title)
        plt.xlabel(x_title)
        plt.ylabel(y_title)
        plt.show()


if __name__=='__main__':
    
    # tune parameters score arrays
    act_type = [
        ("Activation Function for output layer.",
        "Activation Function for hidden layers.")
    ]
    bow_type = [
        ("Types of Bag of words.",
        "Vocbulary size in quantity of words.")
    ]
    lrn_rate = []
    momentum = []
    drop_out = []
    num_unit = []
    num_layr = []


    # find all outputs available
    outputs = os.listdir(os.path.join(DIR, 'output'))

    # iterate through output types for data
    for i, output in enumerate(outputs):
        data = pd.read_csv(
                os.path.join(DIR, 'output', output),
                header=0,
                index_col=0
            )
        t = output.split('_')[0]
        if t == 'act':
            # print('activation output')
            act_type.append(data)
        
        elif t == 'bow':
            # print('bag of words output')
            bow_type.append(data)
        
        elif t == 'hlu':
            # print('bag of words output')
            bow_type.append(data)
        
        elif t == 'lrm':
            # print('bag of words output')
            bow_type.append(data)

    # join all tuning arrays
    all_arrays = [
        act_type,
        bow_type,
        lrn_rate,
        momentum,
        drop_out,
        num_unit,
        num_layr
    ]

    # plot non-empty tuning arrays
    for i, htm_arrays in enumerate(all_arrays):
        if len(htm_arrays) > 1:
            x_title, y_title = htm_arrays[0]
            plot(htm_arrays[1:], x_title, y_title)
    

