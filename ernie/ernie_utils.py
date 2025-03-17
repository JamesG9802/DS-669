import torch

def perturb_observation(obs, perturb_alpha):
    """Apply adversarial perturbation to observations."""
    if isinstance(obs, dict):
        return {agent: perturb_observation(o, perturb_alpha) for agent, o in obs.items()}

    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs, dtype=torch.float32)

    perturbed_obs = obs + torch.normal(
        mean=torch.zeros_like(obs),
        std=torch.ones_like(obs) * perturb_alpha
    )
    
    return perturbed_obs


def adversarial_regularization_loss(agent, obs, perturb_alpha, perturb_steps=3):
    """Compute ERNIE adversarial loss by optimizing perturbations iteratively."""
    losses = []
    for agent_id, agent_obs in obs.items():
        if not isinstance(agent_obs, torch.Tensor):
            agent_obs = torch.tensor(agent_obs, dtype=torch.float32, requires_grad=True)

        perturbed_obs = agent_obs.clone().detach().requires_grad_()

        for _ in range(perturb_steps):
            original_action_dict, _ = agent.get_action({agent_id: agent_obs.detach().cpu().numpy()}, training=False)
            perturbed_action_dict, _ = agent.get_action({agent_id: perturbed_obs.detach().cpu().numpy()}, training=False)

            original_action = torch.tensor(original_action_dict[agent_id], dtype=torch.float32)
            perturbed_action = torch.tensor(perturbed_action_dict[agent_id], dtype=torch.float32)

            distance_loss = torch.norm(original_action - perturbed_action, p="fro")

            grad = torch.autograd.grad(
                outputs=distance_loss, inputs=perturbed_obs,
                grad_outputs=torch.ones_like(distance_loss), retain_graph=True, create_graph=True
            )[0]

            if grad is None:
                raise RuntimeError("Gradient computation failed: grad is None.")

            perturbed_obs = perturbed_obs + perturb_alpha * grad * torch.abs(agent_obs)

        loss = torch.norm(agent_obs - perturbed_obs, p=2)
        losses.append(loss)

    return sum(losses) / len(losses)