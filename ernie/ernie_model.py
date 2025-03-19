import copy

import torch
import torch.nn as nn

from agilerl.algorithms.maddpg import MADDPG

# class ERNIE(nn.Module):
#     def __init__(self, obs_dim, epsilon=0.1):
#         """
#         A simple ERNIE adversarial model that generates perturbations.
        
#         :param obs_dim: The dimension of the observation space.
#         :param epsilon: Strength of the perturbation.
#         """
#         super(ERNIE, self).__init__()
#         self.epsilon = epsilon
#         self.noise_generator = nn.Linear(obs_dim, obs_dim)  # Simple MLP

#     def generate_adversarial_noise(self, obs_tensor):
#         """
#         Generate adversarial noise for the given observation.
        
#         :param obs_tensor: PyTorch tensor representing the observation.
#         :return: Perturbation tensor to be added to the observation.
#         """
#         with torch.no_grad():
#             noise = torch.randn_like(obs_tensor) * self.epsilon  # Gaussian noise
#         return noise

# # Example usage:
# if __name__ == "__main__":
#     dummy_obs = torch.randn(10)  # Example observation with 10 dimensions
#     ernie = ERNIE(obs_dim=10)
#     perturbation = ernie.generate_adversarial_noise(dummy_obs)
#     print("Original Observation:", dummy_obs)
#     print("Perturbation:", perturbation)
#     print("Perturbed Observation:", dummy_obs + perturbation)

def weight_regularization_loss(model, lambda_reg=1e-5):
    weight_loss = 0
    for param in model.parameters():
        # L2 norm of the weights
        weight_loss += torch.norm(param, p=2)
    return lambda_reg * weight_loss

def ernie_learn(self, experiences):
    """
    The custom overriden learning function for the MADDPG algorithm to account for the
    ERNIE regularization loss.
    
    Updates agent network parameters to learn from experiences.

    :param experience: Tuple of dictionaries containing batched states, actions, rewards, next_states,
    dones in that order for each individual agent.
    :type experience: Tuple[Dict[str, torch.Tensor]]
    """
####
#   Taking into account the global obs
####
    states, actions, rewards, next_states, dones, old_global_obs = experiences
    if self.one_hot:
        states = {
            agent_id: nn.functional.one_hot(state.long(), num_classes=state_dim[0])
            .float()
            .squeeze(1)
            for (agent_id, state), state_dim in zip(states.items(), self.state_dims)
        }
        next_states = {
            agent_id: nn.functional.one_hot(
                next_state.long(), num_classes=state_dim[0]
            )
            .float()
            .squeeze(1)
            for (agent_id, next_state), state_dim in zip(
                next_states.items(), self.state_dims
            )
        }
####
#   One hot encoding global obs
####
        old_global_obs = {
            agent_id: nn.functional.one_hot(
                obs.long(), num_classes=state_dim[0]
            )
            .float()
            .squeeze(1)
            for (agent_id, obs), state_dim in zip(
                old_global_obs.items(), self.state_dims
            )
        }
    next_actions = []
    with torch.no_grad():
        if self.arch == "mlp":
            for i, agent_id_label in enumerate(self.agent_ids):
                unscaled_actions = self.actor_targets[i](
                    next_states[agent_id_label]
                )
                if not self.discrete_actions:
                    scaled_actions = torch.where(
                        unscaled_actions > 0,
                        unscaled_actions * self.max_action[i][0],
                        unscaled_actions * -self.min_action[i][0],
                    )
                    next_actions.append(scaled_actions)
                else:
                    next_actions.append(unscaled_actions)
            action_values = list(actions.values())
            state_values = list(states.values())
            input_combined = torch.cat(state_values + action_values, 1)
        elif self.arch == "cnn":
            for i, agent_id_label in enumerate(self.agent_ids):
                unscaled_actions = self.actor_targets[i](
                    next_states[agent_id_label].unsqueeze(2)
                )
                if not self.discrete_actions:
                    scaled_actions = torch.where(
                        unscaled_actions > 0,
                        unscaled_actions * self.max_action[i][0],
                        unscaled_actions * -self.min_action[i][0],
                    )
                    next_actions.append(scaled_actions)
                else:
                    next_actions.append(unscaled_actions)
            stacked_states = torch.stack(list(states.values()), dim=2)
            stacked_actions = torch.cat(list(actions.values()), dim=1)
            stacked_next_states = torch.stack(list(next_states.values()), dim=2)

    if self.arch == "mlp":
        next_input_combined = torch.cat(
            list(next_states.values()) + next_actions, 1
        )
    elif self.arch == "cnn":
        stacked_next_actions = torch.cat(next_actions, dim=1)

    loss_dict = {}
    for idx, (
        agent_id,
        actor,
        critic,
        critic_target,
        actor_optimizer,
        critic_optimizer,
    ) in enumerate(
        zip(
            self.agent_ids,
            self.actors,
            self.critics,
            self.critic_targets,
            self.actor_optimizers,
            self.critic_optimizers,
        )
    ):
        loss_dict[f"{agent_id}"] = self._learn_individual(
            idx,
            agent_id,
            actor,
            critic,
            critic_target,
            actor_optimizer,
            critic_optimizer,
            input_combined if self.arch == "mlp" else None,
            stacked_states if self.arch == "cnn" else None,
            stacked_actions if self.arch == "cnn" else None,
            next_input_combined if self.arch == "mlp" else None,
            stacked_next_states if self.arch == "cnn" else None,
            stacked_next_actions if self.arch == "cnn" else None,
            states,
            actions,
            rewards,
            dones,
            old_global_obs,
        )

    for actor, actor_target, critic, critic_target in zip(
        self.actors, self.actor_targets, self.critics, self.critic_targets
    ):
        self.soft_update(actor, actor_target)
        self.soft_update(critic, critic_target)

    return loss_dict

def ernie_learn_individual(
        self,
        idx,
        agent_id,
        actor,
        critic,
        critic_target,
        actor_optimizer,
        critic_optimizer,
        input_combined,
        stacked_states,
        stacked_actions,
        next_input_combined,
        stacked_next_states,
        stacked_next_actions,
        states,
        actions,
        rewards,
        dones,
        old_global_obs,
    ):
        """
        The custom overriden learning function for the MADDPG algorithm to account for the
        ERNIE regularization loss.
        Inner call to each agent for the learning/algo training
        steps, up until the soft updates. Applies all forward/backward props
        """
        if self.arch == "mlp":
            if self.accelerator is not None:
                with critic.no_sync():
                    q_value = critic(input_combined)
            else:
                q_value = critic(input_combined)
        elif self.arch == "cnn":
            if self.accelerator is not None:
                with critic.no_sync():
                    q_value = critic(stacked_states, stacked_actions)
            else:
                q_value = critic(stacked_states, stacked_actions)

        with torch.no_grad():
            if self.arch == "mlp":
                if self.accelerator is not None:
                    with critic_target.no_sync():
                        q_value_next_state = critic_target(next_input_combined)
                else:
                    q_value_next_state = critic_target(next_input_combined)
            elif self.arch == "cnn":
                if self.accelerator is not None:
                    with critic_target.no_sync():
                        q_value_next_state = critic_target(
                            stacked_next_states, stacked_next_actions
                        )
                else:
                    q_value_next_state = critic_target(
                        stacked_next_states, stacked_next_actions
                    )

        y_j = (
            rewards[agent_id] + (1 - dones[agent_id]) * self.gamma * q_value_next_state
        )

        critic_loss = self.criterion(q_value, y_j)

        # critic loss backprop
        critic_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(critic_loss)
        else:
            critic_loss.backward()
        critic_optimizer.step()

        # update actor and targets
        if self.arch == "mlp":
            if self.accelerator is not None:
                with actor.no_sync():
                    action = actor(states[agent_id])
            else:
                action = actor(states[agent_id])
            if not self.discrete_actions:
                action = torch.where(
                    action > 0,
                    action * self.max_action[idx][0],
                    action * -self.min_action[idx][0],
                )
            detached_actions = copy.deepcopy(actions)
            detached_actions[agent_id] = action
            input_combined = torch.cat(
                list(states.values()) + list(detached_actions.values()), 1
            )
            if self.accelerator is not None:
                with critic.no_sync():
                    actor_loss = -critic(input_combined).mean()
####
#   Adding the ERNIE LOSS to the actor
####
                _obs = old_global_obs[agent_id].clone().detach()
                perturbed_tensor = _obs + torch.normal(torch.zeros_like(_obs), torch.ones_like(_obs) * 1e-3)
                perturbed_tensor.requires_grad = True
                
                #   TODO! setup configuration for perturbations, for now hardcode 
                # for k in range(self.config.alg.perturb_num_steps):
                for k in range(2):
                    distance_loss = torch.norm(actor(_obs) - actor(perturbed_tensor), p="fro")

                    # Compute gradient
                    grad = torch.autograd.grad(outputs=distance_loss, 
                                            inputs=perturbed_tensor, 
                                            grad_outputs=torch.ones_like(distance_loss), 
                                            retain_graph=True, 
                                            create_graph=True)[0]

                    # Apply perturbation
                    perturbation = 1e-3 * grad * torch.abs(_obs.detach())
                    
                    # Ensure perturbation remains small to avoid instability
                    perturbed_tensor = perturbed_tensor + perturbation
                
                adv_reg_loss = torch.norm(actor(_obs) - actor(perturbed_tensor), p="fro")
                actor_loss += adv_reg_loss
                
                #   Add L2 regularization loss
                actor_loss += weight_regularization_loss(actor)

            else:
                actor_loss = -critic(input_combined).mean()
####
#   Adding the ERNIE LOSS to the actor
####
                _obs = old_global_obs[agent_id].clone().detach()
                perturbed_tensor = _obs + torch.normal(torch.zeros_like(_obs), torch.ones_like(_obs) * 1e-3)
                perturbed_tensor.requires_grad = True

                #   TODO! setup configuration for perturbations, for now hardcode 
                # for k in range(self.config.alg.perturb_num_steps):
                for k in range(2):
                    distance_loss = torch.norm(actor(_obs) - actor(perturbed_tensor), p="fro")

                    # Compute gradient
                    grad = torch.autograd.grad(outputs=distance_loss, 
                                            inputs=perturbed_tensor, 
                                            grad_outputs=torch.ones_like(distance_loss), 
                                            retain_graph=True, 
                                            create_graph=True)[0]

                    # Apply perturbation
                    perturbation = 1e-3 * grad * torch.abs(_obs.detach())
                    
                    # Ensure perturbation remains small to avoid instability
                    perturbed_tensor = perturbed_tensor + perturbation
                
                adv_reg_loss = torch.norm(actor(_obs) - actor(perturbed_tensor), p="fro")
                actor_loss += adv_reg_loss

                #   Add L2 regularization loss
                actor_loss += weight_regularization_loss(actor)

        #   TODO! we aren't using a CNN, but I mean we could implement the loss for this
        elif self.arch == "cnn":
            if self.accelerator is not None:
                with actor.no_sync():
                    action = actor(states[agent_id].unsqueeze(2))
            else:
                action = actor(states[agent_id].unsqueeze(2))
            if not self.discrete_actions:
                action = torch.where(
                    action > 0,
                    action * self.max_action[idx][0],
                    action * -self.min_action[idx][0],
                )
            detached_actions = copy.deepcopy(actions)
            detached_actions[agent_id] = action
            stacked_detached_actions = torch.cat(list(detached_actions.values()), dim=1)
            if self.accelerator is not None:
                with critic.no_sync():
                    actor_loss = -critic(
                        stacked_states, stacked_detached_actions
                    ).mean()
            else:
                actor_loss = -critic(stacked_states, stacked_detached_actions).mean()

        # actor loss backprop
        actor_optimizer.zero_grad()
        if self.accelerator is not None:
            self.accelerator.backward(actor_loss)
        else:
            actor_loss.backward()
        actor_optimizer.step()

        return actor_loss.item(), critic_loss.item()

print("Irreversibly transforming MADDPG into ERNIE.")
MADDPG.learn = ernie_learn
MADDPG._learn_individual = ernie_learn_individual