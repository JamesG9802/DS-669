import argparse


def get_args():
    parser = argparse.ArgumentParser()

    # Method
    parser.add_argument('-env', type=str, choices=[
            'simple_speaker_listener',  #   Cooperative communication
            'simple_spread',            #   Cooperative navigation
            'simple_push',              #   Keep-away
            'simple_adversary',         #   Physical deception
            'simple_tag',               #   Predator-prey
            'simple_crypto',            #   Covert communcation
        ],
        default='simple_speaker_listener', 
        help='Pick an environment to run.')

    parser.add_argument('-use_ernie', action='store_true', 
                    help='Enable ERNIE adversarial perturbations.')
    
    # Parameters
    parser.add_argument('-seeds', type=int, default=1,
                        help='random seeds, in range [0, seeds)')
    
    parser.add_argument('-model_num', type=int, default=None,
                    help='Specify the model number to load (default: latest).')

    parser.add_argument('-max_steps', type=int, default=10000,
                    help='The maximum number of training steps.')

    parser.add_argument('-save', action="store_true",
                    help="Whether to save the video.")

    parser.add_argument('-noise', type=float, default=None,
                    help="The episilon noise strength to add to inputs during evaluation.")

    parser.add_argument('-agent_setup', type=str, choices=[
        "m_m",
        "m_e",
        "e_m",
        "e_e"
        ],
        default="m_m",
        help="How models are assigned during viewing. m_m = MADDPG vs MADDPG, m_e = MADDPG vs ERNIE, e_m = ERNIE vs MADDPG, and e_e = ERNIE vs ERNIE"
    )

    return parser.parse_args()
