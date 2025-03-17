import torch
import torch.nn as nn

class ERNIE(nn.Module):
    def __init__(self, obs_dim, epsilon=0.1):
        """
        A simple ERNIE adversarial model that generates perturbations.
        
        :param obs_dim: The dimension of the observation space.
        :param epsilon: Strength of the perturbation.
        """
        super(ERNIE, self).__init__()
        self.epsilon = epsilon
        self.noise_generator = nn.Linear(obs_dim, obs_dim)  # Simple MLP

    def generate_adversarial_noise(self, obs_tensor):
        """
        Generate adversarial noise for the given observation.
        
        :param obs_tensor: PyTorch tensor representing the observation.
        :return: Perturbation tensor to be added to the observation.
        """
        with torch.no_grad():
            noise = torch.randn_like(obs_tensor) * self.epsilon  # Gaussian noise
        return noise

# Example usage:
if __name__ == "__main__":
    dummy_obs = torch.randn(10)  # Example observation with 10 dimensions
    ernie = ERNIE(obs_dim=10)
    perturbation = ernie.generate_adversarial_noise(dummy_obs)
    print("Original Observation:", dummy_obs)
    print("Perturbation:", perturbation)
    print("Perturbed Observation:", dummy_obs + perturbation)
