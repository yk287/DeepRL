import argparse

class options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        # experiment specifics

        # Training Options
        self.parser.add_argument('--epochs', type=int, nargs='?', default=2500, help='total number of training episodes')
        self.parser.add_argument('--time_step', type=int, nargs='?', default=1000, help='total number of training episodes')


        self.parser.add_argument('--show_every', type=int, nargs='?', default=10, help='How often to show images')
        self.parser.add_argument('--print_every', type=int, nargs='?', default=10, help='How often to print scores')

        self.parser.add_argument('--seed', type=int, nargs='?', default=66, help='seeds')

        self.parser.add_argument('--num_workers', type=int, nargs='?', default=16, help='number of cpu cores to use')

        self.parser.add_argument('--save_model', type=bool, nargs='?', default=True, help='Whether to save model or not')
        self.parser.add_argument('--save_model_every', type=int, nargs='?', default=50,
                                 help='how often to save the model')

        self.parser.add_argument('--lr_decay', type=float, nargs='?', default=0.5, help='decay rate for lr')
        self.parser.add_argument('--gamma', type=float, nargs='?', default=0.99, help='discount rate')

        self.parser.add_argument('--ppo_epochs', type=int, nargs='?', default=4, help='how many frames to stack')
        self.parser.add_argument('--epsilon', type=float, nargs='?', default=0.1, help='how many frames to stack')
        self.parser.add_argument('--lambdas', type=float, nargs='?', default=2.0, help='weight for combining loss')
        self.parser.add_argument('--lambda_', type=float, nargs='?', default=0.99, help='lambda for GAE')

        self.parser.add_argument('--env_name', type=str, nargs='?', default='LunarLander-v2', help='environment names')
        self.parser.add_argument('--win_condition', type=int, nargs='?', default=200, help='Winning Condition')
        self.parser.add_argument('--render', type=bool, nargs='?', default=False, help='whether to render')

        # model configs
        self.parser.add_argument('--features_dim', type=int, nargs='?', default=128, help='dimensions of input to the policy network')
        self.parser.add_argument('--depth', type=int, nargs='?', default=5, help='depth of the visual feature extractor')
        self.parser.add_argument('--first_dim', type=int, nargs='?', default=8, help='dimension of the first layer of feature extractor')
        self.parser.add_argument('--i_stacks', type=int, nargs='?', default=4, help='how many frames to stack')

        self.parser.add_argument('--horizon_multiplier', type=int, nargs='?', default=16, help='horizon length = batch * horizon_multiplier')

        self.parser.add_argument('--stacked_image_size', type=int, nargs='?', default=96, help='size of the image tensor')
        self.parser.add_argument('--image_size', type=int, nargs='?', default=96, help='size of the input image')
        self.parser.add_argument('--action_size', type=int, nargs='?', default=3, help='size of the action size')

        self.parser.add_argument('--action_repeat', type=int, nargs='?', default=4, help='how many times to repeat an action')

        self.parser.add_argument('--layer_1', type=int, nargs='?', default=64,
                                 help='number of nodes in the first hidden layer')
        self.parser.add_argument('--layer_2', type=int, nargs='?', default=32,
                                 help='number of nodes in the second hidden layer')

        # Agent Options
        self.parser.add_argument('--batch', type=int, nargs='?', default=64, help='batch size to be used')
        self.parser.add_argument('--memory_size', type=int, nargs='?', default=1000000, help='size of replay memory')

        self.parser.add_argument('--lr', type=float, nargs='?', default=0.00001, help='learning rate')
        self.parser.add_argument('--beta1', type=float, nargs='?', default=0.5, help='values for beta1')
        self.parser.add_argument('--beta2', type=float, nargs='?', default=0.999, help='values for beta2')

        self.parser.add_argument('--update_freq', type=int, nargs='?', default=4, help='how often to update the model')
        self.parser.add_argument('--discount_rate', type=int, nargs='?', default=0.9, help='rewards discount rate')
        self.parser.add_argument('--transfer_rate', type=int, nargs='?', default=0.001,
                                 help='transfer rate for soft update')

        self.parser.add_argument('--num_episodes', type=int, nargs='?', default=3000,
                                 help='total number of training episodes')
        self.parser.add_argument('--max_iteration', type=int, nargs='?', default=1000,
                                 help='max number of iterations per episodes')
        self.parser.add_argument('--min_epsilon', type=int, nargs='?', default=0.1, help='min value for epsilon')
        self.parser.add_argument('--decay', type=int, nargs='?', default=0.995,
                                 help='decay rate of epsilon per episode')
        self.parser.add_argument('--win_cond', type=int, nargs='?', default=200,
                                 help='Condition where the env is considered solved')

        #RAY Options
        self.parser.add_argument('--cpu_use', type=int, nargs='?', default=3, help='Number of CPUs to use')
        self.parser.add_argument('--gpu_use', type=float, nargs='?', default=.125, help='Fraction of GPUs to use')
        self.parser.add_argument('--tune_iter', type=int, nargs='?', default=300, help='number of tuning steps')
        self.parser.add_argument('--num_samples', type=int, nargs='?', default=8, help='number of samples')
        self.parser.add_argument('--perturb_iter', type=int, nargs='?', default=5, help='number of perturb iterations')
        self.parser.add_argument('--train_iterations_per_step', type=int, nargs='?', default=10, help='number of train iter per STEP()')
        self.parser.add_argument('--model_path', type=str, nargs='?', default='/home/youngwook/.ray/models/mnist_cnn.pt', help='directory where inception model gets saved')


    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()

        return self.opt