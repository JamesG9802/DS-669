from multiprocessing.reduction import steal_handle
import os
import glob
from typing import Dict

import imageio
import numpy as np
import pettingzoo
import pettingzoo.mpe
import pettingzoo.mpe.simple_push.simple_push
import pettingzoo.mpe.simple_spread.simple_spread
import torch
from pettingzoo.mpe import simple_speaker_listener_v4, simple_tag_v3, simple_spread_v3, simple_push_v3, simple_adversary_v3, simple_crypto_v3
from PIL import Image, ImageDraw

from agilerl.algorithms.maddpg import MADDPG

from get_args import get_args
import pettingzoo.mpe.simple_push

import pettingzoo.mpe.simple_spread

# Define function to return image
def _label_with_episode_number(frame, episode_num):
    im = Image.fromarray(frame)

    drawer = ImageDraw.Draw(im)

    if np.mean(frame) < 128:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 0, 0)
    drawer.text(
        (im.size[0] / 20, im.size[1] / 18), f"Episode: {episode_num+1}", fill=text_color
    )

    return im


if __name__ == "__main__":
    args = get_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = None
    if args.env == "simple_tag":
        env = simple_tag_v3
    elif args.env == "simple_speaker_listener":
        env = simple_speaker_listener_v4
    elif args.env == "simple_spread":
        env = simple_spread_v3

        # For some reason, the benchmark_data function just isn't properly working??????????
        def simple_spread_benchmark_data(self, agent, world):
            rew = 0
            collisions = 0
            occupied_landmarks = 0
            min_dists = 0
            for lm in world.landmarks:
                dists = [
                    np.sqrt(np.sum(np.square(a.state.p_pos - lm.state.p_pos)))
                    for a in world.agents
                ]
                min_dists += min(dists)
                rew -= min(dists)
                if min(dists) < 0.1:
                    occupied_landmarks += 1
            if agent.collide:
                for a in world.agents:
                    #   an agent shouldn't care if it collides with itself...
                    #   this reflects the correct and intended behaviour as the reward function ignores self-collisions 
                    if self.is_collision(a, agent) and a != agent:
                        rew -= 1
                        collisions += 1
            return (rew, collisions, min_dists, occupied_landmarks)
    
        pettingzoo.mpe.simple_spread.simple_spread.Scenario.benchmark_data = simple_spread_benchmark_data

    elif args.env == "simple_push":
        env = simple_push_v3
        # For some reason, there is no benchmark_data function defined for simple_push???

        def simple_push_benchmark_data(self, agent, world):
            # only for adversary
            if not agent.adversary:
                return 0.0
            # distance from adversary to its goal
            dist = np.linalg.norm(agent.state.p_pos - agent.goal_a.state.p_pos)
            # threshold: “on goal” if within agent size
            return 1.0 if dist < agent.size else 0.0
        pettingzoo.mpe.simple_push.simple_push.Scenario.benchmark_data = simple_push_benchmark_data
    elif args.env == "simple_adversary":
        env = simple_adversary_v3
    elif args.env == "simple_crypto":
        env = simple_crypto_v3

    #   Checks if you have the petting zoo update before they introduced dynamic rescaling
    try:
        env = env.parallel_env(continuous_actions=True, render_mode="rgb_array", dynamic_rescaling=True)
    except Exception:
        env = env.parallel_env(continuous_actions=True, render_mode="rgb_array")

    env.reset()

    try:
        state_dim = [env.observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.observation_space(agent).shape for agent in env.agents]
        one_hot = False
    try:
        action_dim = [env.action_space(agent).n for agent in env.agents]
        discrete_actions = True
        max_action = None
        min_action = None
    except Exception:
        action_dim = [env.action_space(agent).shape[0] for agent in env.agents]
        discrete_actions = False
        max_action = [env.action_space(agent).high for agent in env.agents]
        min_action = [env.action_space(agent).low for agent in env.agents]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = env.num_agents
    agent_ids = env.agents

    models: Dict[str, MADDPG] = {}

    for agent_id in agent_ids:
        # Load the saved agent
        model_dir = f"./models/MADDPG/"
        if args.model_num is None:
            # Find the latest model file if no specific number is given
            model_pattern = f"trained_agent_{args.env}_*.pt"

            if (args.agent_setup == "e_m" and str(agent_id).startswith("adversary")) or \
                (args.agent_setup == "m_e" and str(agent_id).startswith("agent")):
                model_pattern = f"ernie_{model_pattern}" 

            model_files = glob.glob(os.path.join(model_dir, model_pattern))

            if not model_files:
                raise FileNotFoundError(f"No trained MADDPG model found for {args.env} in {model_dir}")

            model_files.sort(key=os.path.getmtime, reverse=True)
            model_path = model_files[0]  # Load the latest model
        else:
            # Load the specified model number
            model_file_name = f"trained_agent_{args.env}_{args.model_num}.pt"

            if (args.agent_setup == "e_m" and str(agent_id).startswith("adversary")) or \
                (args.agent_setup == "m_e" and str(agent_id).startswith("agent")):
                model_pattern = f"ernie_{model_pattern}"

            model_path = os.path.join(model_dir, model_file_name)

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Specified model {model_path} does not exist.")
        print(f"Loading {model_path} for {agent_id}")
        models[agent_id] = MADDPG.load(model_path, device)

    # Define test loop parameters
    episodes = 1000  # Number of episodes to test agent on
    max_steps = 100  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    rewards = []  # List to collect total episodic reward
    infos = []  # List of each agent's info for each episode

    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    def save_function():
        frame = env.render()
        frames.append(_label_with_episode_number(frame, episode_num=ep))

    frame_save = save_function if args.save else lambda _ : None

    epsilon = args.noise

    def perturbed_observation(obs):
        """
        Converts observations from NumPy to PyTorch, applies ERNIE perturbations, and converts back.
        """
        perturbed_obs = {}
        for agent, agent_obs in obs.items():  # Iterate through each agent's observation
            obs_tensor = torch.tensor(agent_obs, dtype=torch.float32)  # Convert NumPy to Tensor
            perturbation = torch.randn_like(obs_tensor) * epsilon  # Gaussian noise
            perturbed_obs[agent] = (obs_tensor + epsilon * perturbation).numpy()  # Convert back to NumPy
        return perturbed_obs  # Return as a dictionary

    #   Get the perturbed observation if using ernie, otherwise use the default obs
    get_obs = perturbed_observation if epsilon != None else lambda obs: obs

    
    # Record benchmark data
    scenario = env.unwrapped.scenario
    world    = env.unwrapped.world

    # Test loop for inference
    episode_info = { agent_id: [] for agent_id in agent_ids }
    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            # Get next action from agent
            state = get_obs(state)

            action = {}

            for agent_id, model in models.items():
                agent_state = state[agent_id]
                agent_infos = info.get(agent_id, {})
                
                cont_actions, discrete_action = model.get_action(
                    state, training=False, infos=info
                )
                if model.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

            # Save the frame for this step and append to frames list
            # frame = env.render()
            # frames.append(_label_with_episode_number(frame, episode_num=ep))
            save_function()

            # Take action in environment
            state, reward, termination, truncation, info = env.step(
                {agent: a.squeeze() for agent, a in action.items()}
            )

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            for agent in world.agents:
                data = scenario.benchmark_data(agent, world)
                episode_info[agent.name].append(data)

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)
        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        # Record benchmark data
        scenario = env.unwrapped.scenario
        world    = env.unwrapped.world

        infos.append(episode_info)

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    print(f"Agent's average reward over {episodes} episodes:")
    for agent_id, reward_list in indi_agent_rewards.items():
        average_reward = sum(reward_list) / len(reward_list)
        min_reward = min(reward_list)
        max_reward = max(reward_list)
        std_reward = np.std(reward_list)
        print(f"{agent_id} average reward: {average_reward}")
        print(f"{agent_id} min reward: {min_reward}")
        print(f"{agent_id} max reward: {max_reward}")
        print(f"{agent_id} std reward: {std_reward}")
        print(average_reward, min_reward, max_reward, std_reward)

    #   Simple push
    print(f"Agent's info over {episodes} episodes:")
    total_frames = 0
    count = 0

    for ep in infos:
        for agent_id, records in ep.items():
            if not str(agent_id).startswith("adversary"):
                break
            for step_info in records:
                total_frames += step_info
            count += len(records)
    
    print("adversary average frame occupancy", total_frames / count)

    #   Simple spread
    # print(f"Agent's infos over {episodes} episodes:")
    # total_collisions = {}
    # total_distances = {}
    # counts = {}

    # for ep in infos:
    #     for agent_id, records in ep.items():
    #         for step_info in records:
    #             # sum collisions
    #             total_collisions[agent_id] = total_collisions.get(agent_id, 0) + step_info[1]
    #             # sum distances
    #             total_distances[agent_id] = total_distances.get(agent_id, 0) + step_info[2]
    #             # count samples
    #             counts[agent_id] = counts.get(agent_id, 0) + 1
    # 
    # for agent_id in counts:
    #     print(agent_id, "average collisions", total_collisions[agent_id] / counts[agent_id])
    #     print(agent_id, "average distances", total_distances[agent_id] / counts[agent_id])

    if not args.save:
        exit(0) 

    # Save the gif to specified path
    gif_path = "./videos/"
    base_filename = f"{args.agent_setup}_{args.env}"

    os.makedirs(gif_path, exist_ok=True)

    # Find existing GIF files that match the pattern
    existing_files = glob.glob(os.path.join(gif_path, f"{base_filename}_*.gif"))

    # Determine the next iteration number
    if existing_files:
        existing_numbers = [
            int(f.split("_")[-1].split(".")[0])  # Extract number from filename
            for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()
        ]
        next_number = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        next_number = 1

    gif_filename = f"{base_filename}_{next_number}.gif"
    save_path = os.path.join(gif_path, gif_filename)

    imageio.mimwrite(save_path, frames, duration=10)