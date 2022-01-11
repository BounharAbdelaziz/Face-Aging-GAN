
import argparse
import os
from utils import helpers
import torch
import model
import datasets as data
from pathlib import Path

class BaseOptions():
    
    def __init__(self) -> None:
        self.initialized = False
        self.parser = None


    def init_experiment_params(self, parser):

        """Define the common options that are used in both training and test."""
        # basic parameters
        parser.add_argument('--img_dir', required=False, default='../datasets/UTKFace/', help='path to training image dataset')
        parser.add_argument('--experiment_name', type=str, default='train_dnn', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./check_points/', help='models are saved here')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')

        # model parameters
        parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use.')
        parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels: 3 for RGB and 1 for grayscale')
        parser.add_argument('--init_type', type=str, default='kaiming', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        parser.add_argument('--norm_type', type=str, default='bn2d', help='Normalization [bn1d | bn2d | in2d | in2d | none]')

        # dataset parameters
        parser.add_argument('--process_ffhq', action='store_true', help='Adds labels to FFHQ filenames to facilitate data loading')
        parser.add_argument('--process_utkface', action='store_true', help='Adds labels to UTKFace filenames to facilitate data loading')
        parser.add_argument('--num_threads', default=4, type=int, help='# threads for loading data')
        parser.add_argument('--batch_size', type=int, default=5, help='input batch size')
        parser.add_argument('--img_size', type=int, default=256, help='scale images to this size')
        parser.add_argument('--n_ages_classes', type=int, default=5, help='Number of age classes. This specify the input shape as follow: 3 + n_ages_classes ( which makes an input of 8 channels by default). It also depends on the age classifier.')

        # additional parameters
        parser.add_argument('--epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--load_iter', type=int, default='0', help='which iteration to load? if load_iter > 0, the code will load models by iter_[load_iter]; otherwise, the code will load models by [epoch]')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{load_size}')
        parser.add_argument('--seed', default=42, type=int, help='Random seed value.')

        self.initialized = True
        self.is_train = False

        return parser

    def gather_options(self):
        """ Initialize our parser with basic options(only once).
            Add additional model-specific and dataset-specific options.
            These options are defined in the <modify_commandline_options> function in model and dataset classes.
        """
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.init_experiment_params(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()

        # save and return the parser
        self.parser = parser
        return parser.parse_args()

    def parse(self):
        """Parse our options, create checkpoints directory suffix, and set up gpu device."""
        opt = self.gather_options()
        opt.is_train = self.is_train   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0]) # set data device

        self.opt = opt
        return self.opt

    def print_options(self, opt):
        """ Print and save options on txt file. It will print both current options and default values(if different).
            It will save options into a text file / [checkpoints_dir] / opt.txt
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            # print("k=",k)
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save 
        experiment_dir = opt.checkpoints_dir / Path(opt.experiment_name)
        experiment_dir = Path.cwd() / experiment_dir

        try :
            experiment_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f'[INFO] experiment_dir directory already exists')
        else:
            print(f'[INFO] experiment_dir directory has been created')

        phase = 'train' if self.is_train else 'test'
        file_name = os.path.join(experiment_dir, '{}_opt.txt'.format(phase))

        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')