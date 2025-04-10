import os
import glob

import imageio
import numpy as np
import torch
from pettingzoo.mpe import simple_speaker_listener_v4, simple_tag_v3, simple_spread_v3, simple_push_v3, simple_adversary_v3, simple_crypto_v3
from PIL import Image, ImageDraw

from agilerl.algorithms.maddpg import MADDPG

from get_args import get_args

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
    elif args.env == "simple_push":
        env = simple_push_v3
    elif args.env == "simple_adversary":
        env = simple_adversary_v3
    elif args.env == "simple_crypto":
        env = simple_crypto_v3
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

    # Load the saved agent
    model_dir = "./models/RARL/"
    if args.model_num is None:
        # Find the latest model file if no specific number is given
        model_pattern = f"RARL_trained_agent_{args.env}_*.pt"
        model_files = glob.glob(os.path.join(model_dir, model_pattern))

        if not model_files:
            raise FileNotFoundError(f"No trained RARL model found for {args.env} in {model_dir}")

        model_files.sort(key=os.path.getmtime, reverse=True)
        model_path = model_files[0]  # Load the latest model
    else:
        # Load the specified model number
        model_path = os.path.join(model_dir, f"RARL_trained_agent_{args.env}_{args.model_num}.pt")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Specified model {model_path} does not exist.")

    print(f"Loading model: {model_path}")
    maddpg = MADDPG.load(model_path, device)

    # Define test loop parameters
    episodes = 10  # Number of episodes to test agent on
    max_steps = 100  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):
            # Get next action from agent
            cont_actions, discrete_action = maddpg.get_action(
                state, training=False, infos=info
            )
            if maddpg.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Save the frame for this step and append to frames list
            frame = env.render()
            frames.append(_label_with_episode_number(frame, episode_num=ep))

            # Take action in environment
            state, reward, termination, truncation, info = env.step(
                {agent: a.squeeze() for agent, a in action.items()}
            )

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
    env.close()

    # Save the gif to specified path
    gif_path = "./videos/"
    base_filename = "RARL_{}".format(args.env)

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