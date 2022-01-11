import torch
from datasets.CACD2000 import CACD2000Data
from datasets.FFHQ import FFHQData
from torch.utils.data import DataLoader
from model.hyperparameters import Hyperparameters
from model.idfgan.id_face_gan import IDFaceGAN
from model.idfgan.generator import IDFGANGenerator
from model.idfgan.discriminator import IDFGANDiscriminator
from model.optimization import init_weights, define_network

from pathlib import Path

from utils.helpers import fix_seed, compute_nbr_parameters, create_checkpoints_dir
from options.training_opt import TrainOptions

if __name__ == "__main__":

    ## Init 
    options = TrainOptions().parse()

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
    path_dump_hyperparams = path / "train_options_.txt"
    hyperparams.dump(path_dump_hyperparams)

    ## Dataloader
    if options.dataset_name.upper() == "FFHQ":
        dataset = FFHQData(options, hyperparams, do_transform=True)
    elif options.dataset_name.upper() == "CACD2000":
        dataset = CACD2000Data(n_classes=hyperparams.n_ages_classes, h=options.img_size, w=options.img_size)
    else:
        print(f"[ERROR] Dataset name {options.dataset_name} unkown.")
        raise
    dataloader = DataLoader(dataset, batch_size=options.batch_size, shuffle=True)#, num_workers=hyperparams.num_threads) 

    # Generator
    generator = IDFGANGenerator(  norm_type=options.norm_type,
                            norm_before=True, 
                            activation='relu', 
                            alpha_relu=0.2, 
                            use_bias=True,
                            min_features = 32, 
                            max_features=256,
                            n_inputs=3, 
                            n_output_first = 64,                
                            n_ages_classes=hyperparams.n_ages_classes, 
                            down_steps=3, 
                            bottleneck_size=9, 
                            up_steps=3,
                            use_pad=True, 
                            interpolation_mode='bicubic', 
                            kernel_size=3,
                            use_UNet_archi=0,
                            is_debug=options.verbose,
                        ) # need to fix shapes problems for UNet archi

    # Discriminator
    discriminator = IDFGANDiscriminator(norm_type=options.norm_type, down_steps=4, max_features=256, in_size=options.img_size, n_ages_classes=hyperparams.n_ages_classes, activation='lk_relu', is_debug=options.verbose)

    print("-----------------------------------------------------")
    print(f"[INFO] Number of trainable parameters for the Generator : {compute_nbr_parameters(generator)}")
    print(f"[INFO] Number of trainable parameters for the Discriminator : {compute_nbr_parameters(discriminator)}")
    print("-----------------------------------------------------")
    print(f"[INFO] Initializing the networks...")
    generator=init_weights(generator, init_type=options.init_type)
    discriminator=init_weights(discriminator, init_type=options.init_type)
    print("-----------------------------------------------------")
    device_ids = [i for i in range(torch.cuda.device_count())]
    print(f"[INFO] Setting up {len(device_ids)} GPU(s) for the networks...")
    print(f"[INFO] .... using GPU(s) device_ids : {device_ids} ...")
    print("-----------------------------------------------------")

    generator=define_network(generator, hyperparams.device, device_ids)
    discriminator=define_network(discriminator, hyperparams.device, device_ids)
    print("-----------------------------------------------------")

    num_steps = len(dataloader) * hyperparams.n_epochs
    print(options.gan_type)

    network = IDFaceGAN(generator, discriminator, options, hyperparams, device_ids)

    ## Start Training
    network.train(dataloader)