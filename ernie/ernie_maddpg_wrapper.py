import torch
import numpy as np
from gymnasium import Wrapper

class ERNIEAdversarialWrapper(Wrapper):
    def __init__(self, env, ernie_model, epsilon=0.1):
        """
        Wrapper to apply ERNIE adversarial perturbations to observations.

        :param env: The Multi-Agent PettingZoo environment.
        :param ernie_model: The ERNIE adversarial model that generates perturbations.
        :param epsilon: Strength of adversarial noise.
        """
        super().__init__(env)
        self.ernie_model = ernie_model
        self.epsilon = epsilon  # Controls perturbation strength

    def observation(self, obs):
        """
        Converts observations from NumPy to PyTorch, applies ERNIE perturbations, and converts back.
        """
        perturbed_obs = {}
        for agent, agent_obs in obs.items():  # Iterate through each agent's observation
            obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)  # Convert NumPy to Tensor
            perturbation = self.ernie_model.generate_adversarial_noise(obs_tensor)  # Apply ERNIE perturbation
            perturbed_obs[agent] = (obs_tensor + self.epsilon * perturbation).numpy()  # Convert back to NumPy
        return perturbed_obs  # Return as a dictionary

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self.observation(obs), info

    def step(self, action):
        next_obs, reward, termination, truncation, info = self.env.step(action)
        return self.observation(next_obs), reward, termination, truncation, info 
