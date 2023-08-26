"""visualization: impact of different experimentation parameters
   on reward accumulation distribution at various episodes length."""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collusion_dilemma.common.constants import *
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


def _fill_reward_vals(val_arr, num_agents, pbrs_factor, hetero_agents=False, val_type="min", gov_rek=False, zero_mean=False):
    assert val_type in ("min", "max")

    for i in range(val_arr.shape[0]):
        for j in range(val_arr.shape[1]):
            if val_type == "min":
                if gov_rek:
                    if zero_mean:
                        val_arr[i][j] = 0 + -pbrs_factor*num_agents + np.round(np.random.uniform(low=-NOISE_VAL, high=NOISE_VAL), 3) * (i+1)
                    else:
                        val_arr[i][j] = 0 + .25*pbrs_factor*num_agents + np.round(np.random.uniform(low=-NOISE_VAL, high=NOISE_VAL), 3) * (i+1)    
                else:
                    val_arr[i][j] = 0 + np.round(np.random.uniform(low=-NOISE_VAL, high=NOISE_VAL), 3) * (i+1)
            
            if val_type == "max":
                if gov_rek:
                    if hetero_agents:
                        val_arr[i][j] = (1.5+pbrs_factor)*num_agents + np.round(np.random.uniform(low=-NOISE_VAL, high=NOISE_VAL), 3) * (i+1)
                    else:
                        val_arr[i][j] = (1+pbrs_factor)*num_agents + np.round(np.random.uniform(low=-NOISE_VAL, high=NOISE_VAL), 3)  * (i+1)
                else:
                    if hetero_agents:
                        val_arr[i][j] = 1.5*num_agents + np.round(np.random.uniform(low=-NOISE_VAL, high=NOISE_VAL), 3) * (i+1)
                    else:    
                        val_arr[i][j] = num_agents + np.round(np.random.uniform(low=-NOISE_VAL, high=NOISE_VAL), 3) * (i+1)
            
            val_arr[i][j] = val_arr[i][j]*(i+1) # to obtain cummulative rewards

    return val_arr


def _plotter(max_eps_len, rewards_values, reward_ci_values, title):
    sns.set_style("whitegrid", {'axes.grid' : True,
                            'axes.edgecolor':'black'
                  })
    fig = plt.figure()
    sns.set(font="Times New Roman", rc={'figure.figsize':(9.5,6.0)})
    sns.set_style({'font.family': 'Times New Roman'})
    plt.clf()
    ax = fig.gca()
    colors = ["forestgreen", "olive", "cornflowerblue", "cadetblue", \
              
              ]
    color_patch = []
    for color, (label, reward_mean), (label, reward_ci_vals) in zip(colors, rewards_values.items(), reward_ci_values.items()):
        # sns.lineplot(data=data, color=color, linewidth=2.5)
        ax.plot(range(0, max_eps_len), reward_mean, color=color)
        ax.fill_between(
            range(0, max_eps_len), reward_ci_vals[0], reward_ci_vals[1], color=color, alpha=.4)
        color_patch.append(mpatches.Patch(color=color, label=label))
    
    plt.xlim([0, max_eps_len-1])
    ax.set_ylim([-150., 400])
    plt.xlabel('Episode Duration', fontsize=15)
    plt.ylabel('Expected Reward Returns', fontsize=15)
    lgd=plt.legend(
    frameon=True, fancybox=True, \
    # prop={'size':14}, handles=color_patch, loc="best")
    prop={'weight':'bold', 'size':14}, handles=color_patch, loc="upper left")
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


def plot_reward_dist(max_eps_len=16, num_agents=16, hetero_agents=False, zero_mean=False, title=None):
    if title is None:
        title = ""
    
    pbrs_factor =  1 / np.sqrt(max_eps_len)
    if hetero_agents:
        pbrs_factor =  1.5 / np.sqrt(max_eps_len * num_agents)

    max_reward_arr = np.zeros((max_eps_len,5))
    min_reward_arr = np.zeros((max_eps_len,5))
    gov_max_reward_arr = np.zeros((max_eps_len,5))
    gov_min_reward_arr = np.zeros((max_eps_len,5))

    max_reward_arr = _fill_reward_vals(max_reward_arr, num_agents, pbrs_factor, hetero_agents, "max", False, zero_mean)
    min_reward_arr = _fill_reward_vals(min_reward_arr, num_agents, pbrs_factor, hetero_agents, "min", False, zero_mean)
    gov_max_reward_arr = _fill_reward_vals(gov_max_reward_arr, num_agents, pbrs_factor, hetero_agents, "max", True, zero_mean)
    gov_min_reward_arr = _fill_reward_vals(gov_min_reward_arr, num_agents, pbrs_factor, hetero_agents, "min", True, zero_mean)
    

    max_reward_mean, max_reward_std = np.mean(max_reward_arr, axis=1), np.std(max_reward_arr, axis=1)
    min_reward_mean, min_reward_std = np.mean(min_reward_arr, axis=1), np.std(min_reward_arr, axis=1)
    gov_max_reward_mean, gov_max_reward_std = np.mean(gov_max_reward_arr, axis=1), np.std(gov_max_reward_arr, axis=1)
    gov_min_reward_mean, gov_min_reward_std = np.mean(gov_min_reward_arr, axis=1), np.std(gov_min_reward_arr, axis=1)

    rewards_mean_values = OrderedDict([
        ("Maximum Accumulated Rewards", max_reward_mean),
        ("Minimum Accumulated Rewards", min_reward_mean),
        ("Governed Maximum Rewards", gov_max_reward_mean),
        ("Governed Minimum Rewards", gov_min_reward_mean),
    ])

    max_reward_ci_lower, max_reward_ci_upper = max_reward_mean - (1.96 * max_reward_std) / np.sqrt(len(max_reward_std)), \
                                               max_reward_mean + (1.96 * max_reward_std) / np.sqrt(len(max_reward_std))
    min_reward_ci_lower, min_reward_ci_upper = min_reward_mean - (1.96 * min_reward_std) / np.sqrt(len(min_reward_std)), \
                                               min_reward_mean + (1.96 * min_reward_std) / np.sqrt(len(min_reward_std))
    gov_max_reward_ci_lower, gov_max_reward_ci_upper = gov_max_reward_mean - (1.96 * gov_max_reward_std) / np.sqrt(len(gov_max_reward_std)), \
                                                       gov_max_reward_mean + (1.96 * gov_max_reward_std) / np.sqrt(len(gov_max_reward_std))
    gov_min_reward_ci_lower, gov_min_reward_ci_upper = gov_min_reward_mean - (1.96 * gov_min_reward_std) / np.sqrt(len(gov_min_reward_std)), \
                                                       gov_min_reward_mean + (1.96 * gov_min_reward_std) / np.sqrt(len(gov_min_reward_std))

    reward_ci_values = OrderedDict([
        ("Maximum Accumulated Rewards", (max_reward_ci_lower, max_reward_ci_upper)),
        ("Minimum Accumulated Rewards", (min_reward_ci_lower, min_reward_ci_upper)),
        ("Governed Maximum Rewards", (gov_max_reward_ci_lower, gov_max_reward_ci_upper)),
        ("Governed Minimum Rewards", (gov_min_reward_ci_lower, gov_min_reward_ci_upper)),
    ])

    _plotter(max_eps_len, rewards_mean_values, reward_ci_values, title)


if __name__ == "__main__":
    plot_reward_dist(title="Cumulative Rewards for Baseline Kernels")
