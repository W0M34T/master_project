# Project Setup and Configuration
Setting up the environment correctly is essential to avoid errors and ensure smooth execution of the project. The following steps outline the installation and configuration processes required to set up the environments for both Single-Agent Reinforcement Learning (SARL) and Multi-Agent Reinforcement Learning (MARL). 

## Project PC Specs
- Windows
- Single processor from the Intel64 family
- Clock speed: approximately 2095 MHz (or 2.095 GHz)
- 32 GB RAM

## First, download and install Python from the official website:
https://www.python.org/downloads/release/python-3115/

## Set up separate virtual environments for SARL and MARL:
### Create and activate virtual environments for SARL and MARL
```bash
conda create -n sagent
conda create -n magent
conda activate sagent
conda install python=3.10
conda activate magent
conda install python=3.10
```

### PyTorch Installation
Install PyTorch in each virtual environment:

```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### StableBaselines3 Installation
Install StableBaselines3 in each virtual environment:

```bash
pip install stable-baselines3
```

### Install additional required packages:

```bash
pip install SuperSuit
pip install pettingzoo
pip install ipykernel
conda install tensorboard
pip install psutil
pip install py-cpuinfo
```

### Dependency Fixes
Fix specific dependencies that might cause issues. Open command line in administrator mode:

```bash
pip uninstall matplotlib
pip install matplotlib
pip uninstall kiwisolver
pip install kiwisolver
pip uninstall pandas
pip install pandas
```

## Verification
Verify that all necessary packages and their versions are correctly installed:

```bash
Package                 Version
----------------------- -----------
gymnasium               0.29.1
kiwisolver              1.4.5
matplotlib              3.9.1
matplotlib-inline       0.1.7
numpy                   1.26.4
pandas                  2.2.2
pettingzoo              1.24.3
psutil                  6.0.0
stable_baselines3       2.3.2
SuperSuit               3.9.2
tensorboard             2.10.0
tensorboard-data-server 0.6.1
tensorboard-plugin-wit  1.8.1
torch                   2.4.0
torchaudio              2.4.0
torchvision             0.15.2a0
```

## Required Code Changes
### SARL Modifications
To enable SARL functionality, make the following modifications:

File Path: site-packages/stable_baselines3/common/policies.py

```python
def obs_to_tensor(self, observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> Tuple[PyTorchObs, bool]:
    vectorized_env = False
    if isinstance(observation, dict):  # CHANGED SARL
        observation = observation["rogue"]

    if isinstance(observation, dict):
        assert isinstance(
            self.observation_space, spaces.Dict
        ), f"The observation provided is a dict but the obs space is {self.observation_space}"
        observation = copy.deepcopy(observation)
        for key, obs in observation.items():
            obs_space = self.observation_space.spaces[key]
            if is_image_space(obs_space):
                obs_ = maybe_transpose(obs, obs_space)
            else:
                obs_ = np.array(obs)
            vectorized_env = vectorized_env or is_vectorized_observation(obs_, obs_space)
            observation[key] = obs_.reshape((-1, *self.observation_space[key].shape))  # type: ignore[misc]

    elif is_image_space(self.observation_space):
        observation = maybe_transpose(observation, self.observation_space)

    else:
        observation = np.array(observation)

    if not isinstance(observation, dict):
        vectorized_env = is_vectorized_observation(observation, self.observation_space)
        observation = observation.reshape((-1, *self.observation_space.shape))  # type: ignore[misc]

    obs_tensor = obs_as_tensor(observation, self.device)
    return obs_tensor, vectorized_env
```

File Path: site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py

```python
def reset(self) -> VecEnvObs:
    for env_idx in range(self.num_envs):
        maybe_options = {"options": self._options[env_idx]} if self._options[env_idx] else {}
        obs, self.reset_infos[env_idx] = self.envs[env_idx].reset(seed=self._seeds[env_idx], **maybe_options)
        self._save_obs(env_idx, obs["rogue"])  # CHANGED SARL
    self._reset_seeds()
    self._reset_options()
    return self._obs_from_buf()
```

### MARL Modifications
For MARL functionality, apply the following changes:

File Path: site-packages/stable_baselines3/common/vec_env/dummy_vec_env.py

```python
def step_wait(self) -> VecEnvStepReturn:
    for env_idx in range(self.num_envs):
        obs, self.buf_rews[env_idx], terminated, truncated, infos = self.envs[env_idx].step(
            self.actions[env_idx]
        )
        self.buf_dones[env_idx] = terminated or truncated
        self.buf_infos[env_idx]["TimeLimit.truncated"] = truncated and not terminated

        if self.buf_dones[env_idx]:
            self.buf_infos[env_idx]["terminal_observation"] = obs
            obs, self.reset_infos[env_idx] = self.envs[env_idx].reset()
        self._save_obs(env_idx, obs[0])  # CHANGED
    return (self._obs_from_buf(), np.copy(self.buf_rews), 
            np.copy(self.buf_dones), deepcopy(self.buf_infos))
```

File Path: site-packages/supersuit/vector/markov_vector_wrapper.py

```python
def reset(self, seed=None, options=None):
    _observations, infos = self.par_env.reset(seed=seed, options=options)
    observations = self.concat_obs(_observations)
    infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]
    return observations[0], infs  # CHANGED

def step(self, actions):  # CHANGED
    act_dict = {
        "rogue": actions,
        "fighter": actions,
        "wizard": actions,
        "cleric": actions
    }

    observations, rewards, terms, truncs, infos = self.par_env.step(act_dict)

    terminations = np.fromiter(terms.values(), dtype=bool)
    truncations = np.fromiter(truncs.values(), dtype=bool)
    env_done = (terminations or truncations).all()
    
    if env_done:
        for agent, obs in observations.items():
            infos[agent]["terminal_observation"] = obs

    rews = rewards["rogue"]
    tms = terms["rogue"]
    tcs = truncs["rogue"]

    infs = [infos.get(agent, {}) for agent in self.par_env.possible_agents]

    if env_done:
        observations, reset_infs = self.reset()
    else:
        observations = self.concat_obs(observations)
        reset_infs = [{} for _ in range(len(self.par_env.possible_agents))]

    infs = [{**inf, **reset_inf} for inf, reset_inf in zip(infs, reset_infs)]

    assert (
        self.black_death or self.par_env.agents == self.par_env.possible_agents
    ), 
    """MarkovVectorEnv does not support environments with varying numbers 
    of active agents unless black_death is set to True"""
    
    return observations, rews, tms, tcs, infs[0]  # CHANGED
```
This README provides detailed steps to set up the environments and make the necessary code modifications for both SARL and MARL. By following these instructions, you can ensure that your project is properly configured and ready for development or experimentation.
