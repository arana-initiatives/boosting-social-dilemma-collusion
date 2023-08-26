"""result visualizer script which summarizes social dilemma problem experiments."""
from marl_trainers.common.constants import *
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import OrderedDict
# import statements from matplotlib package font style setup
import matplotlib.font_manager
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
# setting up the plotter function with 'Times New Roman' font style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
# setting up the plotter function with 'Times New Roman' font style
sns.set(font="Times New Roman")
sns.set_style({'font.family': 'Times New Roman'})


def result_processor(progress_path_dict):
    """returns an OrderedDict with reward mean value, confidence intervals, and timesteps for the plotting function"""
    experiment_results_list = []
    for names, paths in progress_path_dict.items():
        reward_nested_list = []
        eps_len_nested_list = []
        for path in paths:
            exp_df = pd.read_csv(path)
            reward_list_json = list(exp_df["hist_stats/episode_reward"])
            eps_len_list_json = list(exp_df["hist_stats/episode_lengths"])
            reward_list = []
            eps_len_list = []
    
            for reward_, eps_ in zip(reward_list_json, eps_len_list_json):
                reward_list.extend(json.loads(reward_))
                eps_len_list.extend(json.loads(eps_))
    
            reward_nested_list.append(reward_list)
            eps_len_nested_list.append(eps_len_list)

        exp_rewards_arr = np.array(reward_nested_list).T
        exp_rewards_mean_arr, exp_rewards_std_arr = np.mean(exp_rewards_arr, axis=1), np.std(exp_rewards_arr, axis=1)
        exp_rewards_mean_arr, exp_rewards_std_arr = exp_rewards_mean_arr / np.array(eps_len_nested_list[0]), \
                                                    exp_rewards_std_arr / np.array(eps_len_nested_list[0])
        eps_len_arr = np.cumsum(eps_len_nested_list[0])
        exp_rewards_ci_lower, exp_rewards_ci_upper = exp_rewards_mean_arr - (6. * exp_rewards_std_arr) / np.sqrt(len(exp_rewards_std_arr)), \
                                                     exp_rewards_mean_arr + (6. * exp_rewards_std_arr) / np.sqrt(len(exp_rewards_std_arr))
        
        eps_len_arr = np.insert(eps_len_arr, 0, 0, axis=0)
        exp_rewards_mean_arr = np.insert(exp_rewards_mean_arr, 0, 0., axis=0)
        exp_rewards_ci_lower = np.insert(exp_rewards_ci_lower, 0, 0., axis=0)
        exp_rewards_ci_upper = np.insert(exp_rewards_ci_upper, 0, 0., axis=0)
        experiment_results_list.append((names, (eps_len_arr, exp_rewards_mean_arr, exp_rewards_ci_lower, exp_rewards_ci_upper)))

    return OrderedDict(experiment_results_list)


def _plotter(experiment_results_dict, title):
    sns.set_style("whitegrid", {'axes.grid' : True,
                            'axes.edgecolor':'black'
                  })
    fig = plt.figure()
    sns.set(font="Times New Roman", rc={'figure.figsize':(19,12.0)})
    sns.set_style({'font.family': 'Times New Roman'})
    plt.clf()
    ax = fig.gca()
    colors = ["forestgreen", "cornflowerblue", "cadetblue"]
    color_patch = []
    for color, (experiment_name, data_tuple) in zip(colors, experiment_results_dict.items()):
        # sns.lineplot(data=data, color=color, linewidth=2.5)
        ax.plot(data_tuple[0], data_tuple[1], color=color)
        ax.fill_between(data_tuple[0], data_tuple[2], data_tuple[3], color=color, alpha=.4)
        color_patch.append(mpatches.Patch(color=color, label=experiment_name))
    
    plt.xlim([0, 12000])
    ax.set_ylim([.5, 1.5])
    plt.xlabel('Timesteps Duration', fontsize=15)
    plt.ylabel('Reward Values', fontsize=15)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    # prop={'size':14}, handles=color_patch, loc="best")
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="best")
    plt.title(title, fontsize=20)
    ax = plt.gca()
    
    # uncomment for adding custom tick values
    # ax.set_xticks([10, 20, 30, 40, 50])
    # ax.set_xticklabels([0.5, 1, 1.5, 2.5, 3.0])

    plt.setp(ax.get_xticklabels(), fontsize=15)
    plt.setp(ax.get_yticklabels(), fontsize=15)
    sns.despine()
    plt.tight_layout()
    plt.show()


def experiment_plotter(experiment_result_paths, title):
    
    experiment_names = ["Baseline PPO Trainer", "Baseline Governed Trainer", "Zero-Mean Governed Trainer"]
    progress_path_list = []
    # generate complete paths to load all the progress.csv result files
    for names, paths in zip(experiment_names, experiment_result_paths):
        result_sub_dirs = [x for x in paths.iterdir() if x.is_dir()]
        result_sub_dirs = [x / PROGRESS_CSV for x in result_sub_dirs]
        progress_path_list.append((names, result_sub_dirs))
    
    progress_path_dict = OrderedDict(progress_path_list)
    experiment_results_dict = result_processor(progress_path_dict)

    _plotter(experiment_results_dict, title)



if __name__ == "__main__":
    experiment_result_paths = [PPO_LARGE_HMG_TRAINER_PATH, PPO_LARGE_HMG_GOV_TRAINER_PATH, PPO_LARGE_HMG_ZERO_MEAN_GOV_TRAINER_PATH]
    # experiment_result_paths = [PPO_LARGE_HTR_TRAINER_PATH, PPO_LARGE_HTR_GOV_TRAINER_PATH, PPO_LARGE_HTR_ZERO_MEAN_GOV_TRAINER_PATH]
    experiment_plotter(experiment_result_paths, "Average Homogeneous Reward Returns")