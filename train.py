from argparse import ArgumentParser
from prog.trainer import Trainer

def parse_args():
    parser = ArgumentParser()

    # crucial arguments
    parser.add_argument('-s', '--seed', default=0, type=int,
                        help='random seed for torch and gym')
    parser.add_argument('-l', '--lr', default=0.001, type=float,
                        help='learning rate')
    parser.add_argument('-b', '--betas', default=(0.9, 0.99), type=tuple,
                        help='hyper-parameter for Adam optimizer')
    parser.add_argument('-g', '--gamma', default=0.99, type=float,
                        help='discount factor for future reward')
    parser.add_argument('-e', '--environment', default="Boxoban-Train-v0", type=str,
                        help='environment used for training')

    parser.add_argument('-c', '--eps_clip', default=0.2, type=float,
                        help='epsilon clip co-efficient for ppo')
    parser.add_argument('-k', '--k_epochs', default=5, type=float,
                        help='n updates in PPO')



    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    prog = Trainer(args)
    prog.train()

if __name__ == '__main__':
    main()
