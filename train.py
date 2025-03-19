"""This tutorial shows how to train an MADDPG agent on the space invaders atari environment.

Authors: Michael (https://github.com/mikepratt1), Nick (https://github.com/nicku-a)
"""
import copy
import os
import glob
import numpy as np
import torch
from tqdm import trange
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

def train_algorithm(env, env_name, NET_CONFIG, INIT_HP, num_envs, max_steps, use_ernie: bool, device=None):
    if use_ernie:
        #   Load the ERNIE monkey-patch
        import ernie.ernie_model

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Set up the environment
    env = env.parallel_env(continuous_actions=True)
    env = AsyncPettingZooVecEnv([lambda: env for _ in range(num_envs)])
    env.reset()

    # Configure the multi-agent algo input arguments
    try:
        state_dim = [env.single_observation_space(agent).n for agent in env.agents]
        one_hot = True
    except Exception:
        state_dim = [env.single_observation_space(agent).shape for agent in env.agents]
        one_hot = False

    try:
        action_dim = [env.single_action_space(agent).n for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = True
        INIT_HP["MAX_ACTION"] = None
        INIT_HP["MIN_ACTION"] = None
    except Exception:
        action_dim = [env.single_action_space(agent).shape[0] for agent in env.agents]
        INIT_HP["DISCRETE_ACTIONS"] = False
        INIT_HP["MAX_ACTION"] = [
            env.single_action_space(agent).high for agent in env.agents
        ]
        INIT_HP["MIN_ACTION"] = [
            env.single_action_space(agent).low for agent in env.agents
        ]

    # Not applicable to MPE environments, used when images are used for observations (Atari environments)
    if INIT_HP["CHANNELS_LAST"]:
        state_dim = [
            (state_dim[2], state_dim[0], state_dim[1]) for state_dim in state_dim
        ]

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["N_AGENTS"] = env.num_agents
    INIT_HP["AGENT_IDS"] = env.agents

    # Create a population ready for evolutionary hyper-parameter optimisation
    pop = create_population(
        INIT_HP["ALGO"],
        state_dim,
        action_dim,
        one_hot,
        NET_CONFIG,
        INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    # Configure the multi-agent replay buffer
    
    if use_ernie:
        field_names = ["old_global_obs", "state", "action", "reward", "next_state", "done"]
    else:
        field_names = ["state", "action", "reward", "next_state", "done"]

    memory = MultiAgentReplayBuffer(
        INIT_HP["MEMORY_SIZE"],
        field_names=field_names,
        agent_ids=INIT_HP["AGENT_IDS"],
        device=device,
    )

    # Instantiate a tournament selection object (used for HPO)
    tournament = TournamentSelection(
        tournament_size=2,  # Tournament selection size
        elitism=True,  # Elitism in tournament selection
        population_size=INIT_HP["POPULATION_SIZE"],  # Population size
        eval_loop=1,  # Evaluate using last N fitness scores
    )

    # Instantiate a mutations object (used for HPO)
    mutations = Mutations(
        algo=INIT_HP["ALGO"],
        no_mutation=0.2,  # Probability of no mutation
        architecture=0.2,  # Probability of architecture mutation
        new_layer_prob=0.2,  # Probability of new layer mutation
        parameters=0.2,  # Probability of parameter mutation
        activation=0,  # Probability of activation function mutation
        rl_hp=0.2,  # Probability of RL hyperparameter mutation
        rl_hp_selection=[
            "lr",
            "learn_step",
            "batch_size",
        ],  # RL hyperparams selected for mutation
        mutation_sd=0.1,  # Mutation strength
        agent_ids=INIT_HP["AGENT_IDS"],
        arch=NET_CONFIG["arch"],
        rand_seed=1,
        device=device,
    )

    # Define training loop parameters
    learning_delay = 0  # Steps before starting learning
    evo_steps = 1000  # Evolution frequency
    eval_steps = None  # Evaluation steps per episode - go until done
    eval_loop = 1  # Number of evaluation episodes
    elite = pop[0]  # Assign a placeholder "elite" agent
    
    total_steps = 0

    # TRAINING LOOP
    print("Training...")
    pbar = trange(max_steps, unit="step")
    while np.less([agent.steps[-1] for agent in pop], max_steps).all():
        pop_episode_scores = []
        for agent in pop:  # Loop through population
            state, info = env.reset()  # Reset environment at start of episode
            scores = np.zeros(num_envs)
            completed_episode_scores = []
            steps = 0
            old_global_obs = copy.deepcopy(state) #    initialize the ernie observation

            if INIT_HP["CHANNELS_LAST"]:
                state = {
                    agent_id: np.moveaxis(s, [-1], [-3])
                    for agent_id, s in state.items()
                }

            for idx_step in range(evo_steps // num_envs):

                # Get next action from agent
                cont_actions, discrete_action = agent.get_action(
                    states=state, training=True, infos=info
                )
                if agent.discrete_actions:
                    action = discrete_action
                else:
                    action = cont_actions

                # Act in environment
                try:
                    next_state, reward, termination, truncation, info = env.step(action)
                except Exception:
                    print("Crashed")
                    print(action)
                    for i, actor in enumerate(agent.actors):
                        # Sum all the weights of the actor
                        actor_weight_sum = sum(p.sum() for p in actor.parameters())
                        print(f"Sum of actor weights for agent {i}: {actor_weight_sum.item()}")
                        for p in actor.parameters():
                            print(p)
                    exit()
                scores += np.sum(np.array(list(reward.values())).transpose(), axis=-1)
                total_steps += num_envs
                steps += num_envs

                # Image processing if necessary for the environment
                if INIT_HP["CHANNELS_LAST"]:
                    next_state = {
                        agent_id: np.moveaxis(ns, [-1], [-3])
                        for agent_id, ns in next_state.items()
                    }

                # Save experiences to replay buffer

                if use_ernie:
                    memory.save_to_memory(
                        state,
                        cont_actions,
                        reward,
                        next_state,
                        termination,
                        old_global_obs,
                        is_vectorised=True,
                    )
                else:
                    memory.save_to_memory(
                        state,
                        cont_actions,
                        reward,
                        next_state,
                        termination,
                        is_vectorised=True,
                    )

                # Learn according to learning frequency
                # Handle learn steps > num_envs
                if agent.learn_step > num_envs:
                    learn_step = agent.learn_step // num_envs
                    if (
                        idx_step % learn_step == 0
                        and len(memory) >= agent.batch_size
                        and memory.counter > learning_delay
                    ):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)
                # Handle num_envs > learn step; learn multiple times per step in env
                elif (
                    len(memory) >= agent.batch_size and memory.counter > learning_delay
                ):
                    for _ in range(num_envs // agent.learn_step):
                        # Sample replay buffer
                        experiences = memory.sample(agent.batch_size)
                        # Learn according to agent's RL algorithm
                        agent.learn(experiences)

                state = next_state
                old_global_obs = copy.deepcopy(state)

                # Calculate scores and reset noise for finished episodes
                reset_noise_indices = []
                term_array = np.array(list(termination.values())).transpose()
                trunc_array = np.array(list(truncation.values())).transpose()
                for idx, (d, t) in enumerate(zip(term_array, trunc_array)):
                    if np.any(d) or np.any(t):
                        completed_episode_scores.append(scores[idx])
                        agent.scores.append(scores[idx])
                        scores[idx] = 0
                        reset_noise_indices.append(idx)
                agent.reset_action_noise(reset_noise_indices)

            pbar.update(evo_steps // len(pop))

            agent.steps[-1] += steps
            pop_episode_scores.append(completed_episode_scores)

        # Evaluate population
        fitnesses = [
            agent.test(
                env,
                swap_channels=INIT_HP["CHANNELS_LAST"],
                max_steps=eval_steps,
                loop=eval_loop,
            )
            for agent in pop
        ]
        mean_scores = [
            (
                np.mean(episode_scores)
                if len(episode_scores) > 0
                else "0 completed episodes"
            )
            for episode_scores in pop_episode_scores
        ]

        print(f"--- Global steps {total_steps} ---")
        print(f"Steps {[agent.steps[-1] for agent in pop]}")
        print(f"Scores: {mean_scores}")
        print(f'Fitnesses: {["%.2f"%fitness for fitness in fitnesses]}')
        print(
            f'5 fitness avgs: {["%.2f"%np.mean(agent.fitness[-5:]) for agent in pop]}'
        )

        # Tournament selection and population mutation
        elite, pop = tournament.select(pop)
        pop = mutations.mutation(pop)

        # Update step counter
        for agent in pop:
            agent.steps.append(agent.steps[-1])

    # Save the trained algorithm
    algo_name = str(INIT_HP["ALGO"])
    path = f"./models/{algo_name}"
    base_filename = "trained_agent_{}".format(env_name)

    if use_ernie:
        base_filename = f"ernie_{base_filename}" 

    os.makedirs(path, exist_ok=True)

    # Find existing files that match
    existing_files = glob.glob(os.path.join(path, f"{base_filename}_*"))

    # Determine the next iteration number
    if existing_files:
        existing_numbers = [
            int(f.split("_")[-1].split(".")[0])  # Extract number from filename
            for f in existing_files if f.split("_")[-1].split(".")[0].isdigit()
        ]
        next_number = max(existing_numbers) + 1 if existing_numbers else 1
    else:
        next_number = 1

    # Create new filename
    filename = f"{base_filename}_{next_number}.pt"

    save_path = os.path.join(path, filename)
    elite.save_checkpoint(save_path)

    pbar.close()
    env.close()