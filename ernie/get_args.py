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

    # Parameters
    parser.add_argument('-seeds', type=int, default=1,
                        help='random seeds, in range [0, seeds)')
    
    parser.add_argument('-use_ernie', action='store_true', 
                    help='Enable ERNIE adversarial perturbations.')
    
    parser.add_argument('-model_num', type=int, default=None,
                    help='Specify the model number to load (default: latest).')

    return parser.parse_args()
