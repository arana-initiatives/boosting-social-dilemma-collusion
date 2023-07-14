### Boosted Social Dilemma Collusion

This repository provides a compact multi-agent implementation for the social dilemma problem with homogenous and heterogenous agents having same and different reward pay-offs respectively.
This `N`-player social dilemma problem simulates the oligopolistic collusion in action strategy-wise simpler, but number of agent-wise harder algorithmic collusion settings.
The social dilemma environment is implemented with `PettiingZoo` and the reinforcement learning algorithms are supported through Ray's `rllib` multi-agent implementation.

#### Detailed Functional Description

First, the founding action strategy pay-off matrix/vector is defined by the `collusion_payoffs.py` module. This allows the functionality to add more agents _(num\_agent)_, and also introduce heterogeneity _(hetero\_prob)_ as well.
The `collusion_dilemma_envs.py` implements the collusion dilemma environments in both _ParallelEnv_ and _AECEnv_ formats _(refer `PettingZoo` documentation)_.
And, in addition to _num\_agent_ and _hetero\_prob_ parameters, this environment also supports the episode length with `eps_len` to customize the episode length.
Finally, we have our `dqn_agent_trainer.py` script which trains LSTM policy based agents, and also compares it to simpler policies as well for a more comprehensive comparison.

#### Getting with the Project Implementation

Also, for setting the repository please follow the below listed steps for easy replication.

* First, add the project repository to your `PYTHONPATH` or add it to your virtual environments `.pth` file
* Make sure prerequisite packages are installed for avoiding any package dependency issues. A helpful `requirements.txt` from `pip freeze` command is added as reference.
  - `Known Issue Fix`: Because of some implementation compatibility issues between `PettingZoo` and `rllib`, the `pettinzoo_env.py` wrapper file in `rllib` needs to be updated. Please, remove the `return_info=True,` occurrences from this file for successful execution of the trainer scripts.

After the setup, the model can be simply trained with the execution of the main trainer script `dqn_agent_trainer.py`. Additionally, customization to agent nature, agent numbers, and other hyperparameters can be done through changing the imported config `.yaml` files.

__Note:__ Installing this setup can cause some unexpected issues. Please, feel free to report any dependency related new issues.

#### Initial Results and Discussion

For starters, half of the actions for each agent does not yield any rewards if cooperation/collusion does not happen.
And, it makes the pay-off matrix/vector even more sparse when the number of agents increases which makes the convergence problem harder.
With our experimentation, we found that in this setup with simplicity in agent strategy options _(i.e. action-space/state reformulation design)_ the collusion happens fastest in a decentralized executor based true multi-agent setting.
The collusion also persists for heterogeneous agents as well, and also the simpler strategies are overpowered heavily by more novel strategies.
Also, we observe that collusion easily happens for a relatively larger number of agents as well.
Hence, collusion propensity is increased by a multi-agent implementation with simplistic strategy reformulation design.
And it is highly robust to even more wider oligopolies which can include heterogeneous agents as well.
This simple experimental setup highlights the threat that the automated collusion algorithms have in realistic decentralized executor settings.

#### Citing the Experiment Findings and Accompanying Theoretical Document

If you find any of the theoretical work interesting and useful in your research work, please consider citing it with corresponding portfolio citations:

```
@misc{algorithmic-collusion-analysis,
  author = {Rana, Ashish},
  title = {Exploring Robustness of Automated Pricing Algorithmic Collusion in Financial Markets},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/arana-initiatives/social-dilemma-collusion}},
}
```