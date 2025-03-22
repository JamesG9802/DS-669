import copy

import numpy as np
import torch
import torch.nn as nn

from agilerl.algorithms.maddpg import MADDPG
from wandb import agent

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

def ernie_adv_reg_loss(self, obs, perturbed_obs, actions, actor):
    '''
    Function to get the regularization part of the loss function based on adversarial perturbation

    Parameters
    ----------
    - obs : the non-perturbed obs tensor [batch_size, num_lights, obs_shape]
    - perturbed_obs : the perturbed obs tensor [batch_size, num_lights, obs_shape]
    - actions : the actions taken by each agent [batch_size, num_lights]
    - actor : the actor that we are applying the regularizer to [nn.Module]
    Returns
    -------
    - reg_loss : the regularization loss
    '''
    # print(obs.shape, perturbed_obs.shape)
    perturbed_obs = perturbed_obs.detach()
    # if network == self.critic:
    #     obs_shape = obs.size()[2]
    #     actions = actions.view(-1, 1)
    #     normal = network(obs.view(-1, obs_shape), actions)
    #     perturbed = network(perturbed_obs.view(-1, obs_shape), actions)
    # elif network == self.actor:
    normal = actor(obs)
    perturbed = actor(perturbed_obs)

    reg_loss = torch.norm(normal - perturbed, p="fro")
    
    return reg_loss

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
#   Taking into account the local obs
####
    states, actions, rewards, next_states, dones = experiences
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
            # detached_actions = copy.deepcopy(actions)
            # detached_actions[agent_id] = action
            # input_combined = torch.cat(
            #     list(states.values()) + list(detached_actions.values()), 1
            # )            
            detached_actions = copy.deepcopy(actions)
            detached_actions[agent_id] = action
            _states = {}
            for key, state in states.items():
                if key == agent_id:
                    # Make sure the agent's state is differentiable
                    state.requires_grad = True
                    _states[key] = state
                else:
                    _states[key] = state.detach()
            input_combined = torch.cat(
                list(_states.values()) + list(detached_actions.values()), 1
            )
            
            if self.accelerator is not None:
                with critic.no_sync():
                    actor_loss = -critic(input_combined).mean()
                    
                    # Get the regularization loss for a smooth policy
                    perturbed_tensor = torch.normal(states[agent_id].detach(), torch.ones_like(states[agent_id].detach()) * 1e-3)
                    perturbed_tensor.requires_grad = True
                    # for i in range(self.config.alg.perturb_num_steps):
                    for i in range(2):
                        # Gradient of loss wrt the old observation

                        obs_grad = torch.autograd.grad(outputs=actor_loss, inputs=states[agent_id], grad_outputs=torch.ones_like(actor_loss), retain_graph=True)[0]
                        
                        # project gradient onto ball
                        # obs_grad = torch.clamp(input=obs_grad, min=-self.config.alg.perturb_radius,
                        #                         max=self.config.alg.perturb_radius)
                        obs_grad = torch.clamp(input=obs_grad, min=-1,
                                                max=1)
                        # perturbed_tensor = perturbed_tensor + self.config.alg.perturb_alpha * obs_grad
                        perturbed_tensor = perturbed_tensor + 1e-1 * obs_grad

                    adv_reg_loss = self.get_adv_reg_loss(states[agent_id], perturbed_tensor, actions, actor)
                    # actor_loss = actor_loss + self.config.alg.lam * adv_reg_loss
                    actor_loss = actor_loss + 1e-1 * adv_reg_loss

            else:
                actor_loss = -critic(input_combined).mean()
                
                # Get the regularization loss for a smooth policy
                perturbed_tensor = torch.normal(states[agent_id].detach(), torch.ones_like(states[agent_id].detach()) * 1e-3)
                perturbed_tensor.requires_grad = True
                # for i in range(self.config.alg.perturb_num_steps):
                for i in range(2):
                    # Gradient of loss wrt the old observation

                    obs_grad = torch.autograd.grad(outputs=actor_loss, inputs=states[agent_id], grad_outputs=torch.ones_like(actor_loss), retain_graph=True)[0]
                    
                    # project gradient onto ball
                    # obs_grad = torch.clamp(input=obs_grad, min=-self.config.alg.perturb_radius,
                    #                         max=self.config.alg.perturb_radius)
                    obs_grad = torch.clamp(input=obs_grad, min=-1,
                                            max=1)
                    # perturbed_tensor = perturbed_tensor + self.config.alg.perturb_alpha * obs_grad
                    perturbed_tensor = perturbed_tensor + 1e-1 * obs_grad

                adv_reg_loss = self.get_adv_reg_loss(states[agent_id], perturbed_tensor, actions, actor)
                # actor_loss = actor_loss + self.config.alg.lam * adv_reg_loss
                actor_loss = actor_loss + 1e-1 * adv_reg_loss

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
MADDPG.get_adv_reg_loss = ernie_adv_reg_loss
MADDPG.learn = ernie_learn
MADDPG._learn_individual = ernie_learn_individual