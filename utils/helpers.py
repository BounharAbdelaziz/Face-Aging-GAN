from torch.utils.tensorboard import SummaryWriter
import numpy as np

def write_logs_tb(tb_writer_loss, tb_writer_fake, tb_writer_real, img_fake, img_real, age_fake, losses, step, epoch, hyperparams, with_print_logs=True):

    for k,v in losses.items():
        tb_writer_loss.add_scalar(
            k, v, global_step=step
        )
        
    # Adding generated images to tb
    age_fake=age_fake.cpu().numpy()
    age_fake = (np.argmax(age_fake)+1)*10

    label_fake = "Fake images of age "+str(age_fake)
    tb_writer_fake.add_image(
        label_fake, img_fake, global_step=step
    )

    tb_writer_real.add_image(
        "Real images", img_real, global_step=step
    )

    if with_print_logs :
        print(f"Epoch [{epoch}/{hyperparams.n_epochs}]", sep=' ')
        for k,v in losses.items():
            print(f"{k} : [{v:.4f}]", sep=' - ', end=' - ')
            
    
def compute_nbr_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)