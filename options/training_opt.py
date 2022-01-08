from options.base_opt import BaseOptions

class TrainOptions(BaseOptions):
    """This class includes training options.
    It also includes shared options defined in BaseOptions.
    """

    def init_experiment_params(self, parser):
        parser = BaseOptions.init_experiment_params(self, parser)

        # visdom and HTML visualization parameters
        parser.add_argument('--print_freq', type=int, default=20, help='frequency of showing training results on console')

        # network saving and loading parameters
        parser.add_argument('--save_weights_freq', type=int, default=5000, help='frequency of saving the training weights')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--optimizer', type=str, default="Adam", help='Optimizer to use')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate for adam')
        parser.add_argument('--weight_decay', type=float, default=0.001, help='The weight decay, if using [AdamW | SGD]')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='exp', help='learning rate policy. [linear | step | plateau | cosine | exp]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--gamma', type=float, default=0.99, help='gamma for the lr decay')
        parser.add_argument('--warmup_period', type=int, default=100, help='Warmup period (# iterations)')

        # Loss functions
        parser.add_argument('--lambda_G', type=float, default=1, help='weight of loss_G.')
        parser.add_argument('--lambda_D', type=float, default=1, help='weight of loss_D.')
        parser.add_argument('--lambda_L1', type=float, default=0, help='weight of loss_L1.')
        parser.add_argument('--lambda_MSE', type=float, default=250, help='weight of loss_MSE.')
        parser.add_argument('--lambda_PCP', type=float, default=0.85, help='weight of loss_PCP.')
        parser.add_argument('--lambda_AGE', type=float, default=0, help='weight of loss_AGE.')
        parser.add_argument('--lambda_ID', type=float, default=0, help='weight of loss_ID.')
        parser.add_argument('--lambda_cycle', type=float, default=10, help='weight of loss_ID.')
        parser.add_argument('--lambda_ID_cycle', type=float, default=0.5, help='weight of identity loss of cycleGAN.')

        self.is_train = True

        return parser
