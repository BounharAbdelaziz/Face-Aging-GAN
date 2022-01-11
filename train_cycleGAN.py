import torch

from datasets import add_label_to_img, split_domains
from datasets.UTKFace import UTKFaceData
from torch.utils.data import DataLoader
from model.hyperparameters import Hyperparameters
from model.cycle_gan.cycle_gan import CycleGAN

from pathlib import Path

from utils.helpers import fix_seed, create_checkpoints_dir
from options.training_opt import TrainOptions


if __name__ == "__main__":



    ## Init Models
    options = TrainOptions().parse()

    if options.process_ffhq:
        # add_label_to_img(is_debug=False, delete_when_enmpty_features=True)
        split_domains( path_dataset="../datasets/ffhq_mini/images/", domain_1_age=25)
        exit()

    if options.process_utkface:
        split_domains( path_dataset=options.img_dir, domain_1_age=39)

     # fix seeds for reproducibility
    fix_seed(options.seed)

    ## Checkpoints directory
    path = Path.cwd() / "check_points" / options.experiment_name
    create_checkpoints_dir(path)

    hyperparams = Hyperparameters(  show_advance=options.print_freq, 
                                    lr=options.lr, 
                                    batch_size=options.batch_size, 
                                    n_epochs=options.n_epochs,
                                    save_weights=options.save_weights_freq,
                                    lambda_disc=options.lambda_D,
                                    lambda_gen=options.lambda_G,
                                    lambda_pcp=options.lambda_PCP,
                                    lambda_id=options.lambda_ID,
                                    lambda_mse=options.lambda_MSE,
                                    num_threads=options.num_threads)

    # dumping hyperparams to keep track of them for later comparison/use
    path_dump_hyperparams = path / "train_options.txt"
    hyperparams.dump(path_dump_hyperparams)

    ## Init Dataloader
    dataset = UTKFaceData(options, domain_1_name="domain_1", domain_2_name="domain_2", do_transform=True)

    # init dataloarder
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)

    if options.batch_size > 1 :
        norm_type='bn2d'
    else :
        norm_type='none'


    network = CycleGAN(hyperparams, options)

    ## Start Training
    network.train(  dataloader=dataloader)