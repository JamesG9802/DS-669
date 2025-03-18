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


    parser.add_argument('-algo', type=str, choices=[
            'MADDPG',
        ],
        default='MADDPG', 
        help='Pick an algorithm to use.')

    parser.add_argument('-use_ernie', action='store_true', 
                    help='Enable ERNIE adversarial perturbations.')
    
    # Parameters
    parser.add_argument('-seeds', type=int, default=1,
                        help='random seeds, in range [0, seeds)')
    
    parser.add_argument('-model_num', type=int, default=None,
                    help='Specify the model number to load (default: latest).')

    parser.add_argument('-max_steps', type=int, default=10000,
                    help='The maximum number of training streps.')

    return parser.parse_args()
