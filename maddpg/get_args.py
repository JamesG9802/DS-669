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

    return parser.parse_args()
