import argparse
import plotly
import plotly as py
import plotly.graph_objs as go
import pandas as pd
import scipy.signal
import seaborn as sns; sns.set()
import numpy as np
import matplotlib.pyplot as plt
from tbparser.summary_reader import SummaryReader
import matplotlib.patches as mpatches




def smooth_and_plot(col_name, metric_df):

    smoothed_df = metric_df.copy()
    smoothed_df[col_name] = scipy.signal.savgol_filter(metric_df[col_name], 201,3)

    lower_df = smoothed_df.copy()
    lower_df[col_name] = lower_df[col_name] - metric_df[col_name].std()

    upper_df = smoothed_df.copy()
    upper_df[col_name] = upper_df[col_name] + metric_df[col_name].std()





    
    fig, ax = plt.subplots(1)
    ax.plot(smoothed_df['step'], smoothed_df[col_name], lw=2, label='smoothed_reward', color='blue')
    ax.fill_between(lower_df['step'], lower_df[col_name], upper_df[col_name], facecolor='blue', alpha=0.3,
                    label='1 std range')
    ax.legend(loc='upper left')
    ax.set_ylabel(col_name)
    ax.set_xlabel('step')
    ax.grid(True)
    plt.show()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--tb-log', help="Path for tb_log", type=str)
    args = parser.parse_args()
    reader = SummaryReader(args.tb_log)
    rewards = []
    steps = []
    reward_dict = []
    closs_dict = []
    aloss_dict = []
    for item in reader:
        if item.tag == 'episode_reward':
            reward_dict.append({"step":item.step,"episode_reward":item.value})
            rewards.append(item.value)

    print("Finished reading")
    reward_df = pd.DataFrame.from_dict(reward_dict)
    closs_df = pd.DataFrame.from_dict(closs_dict)
    aloss_df = pd.DataFrame.from_dict(aloss_dict)
    smooth_and_plot('episode_reward', reward_df)
    #smooth_and_plot('critic_loss', closs_df)
    #smooth_and_plot('actor_loss', aloss_df)
