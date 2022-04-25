from pathlib import Path
import shutil
import os
from tqdm import tqdm
def split_domains( path_dataset="../datasets/UTKFace/", domain_1_age=35):

    domain_1_path = Path(path_dataset) / "domain_1"
    domain_2_path = Path(path_dataset) / "domain_2"

    try :
        domain_1_path.mkdir(parents=True, exist_ok=False)
        domain_2_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'[INFO] Dataset already splited.')
    else:
        print(f'[INFO] Dataset is being splited...')
        
    for filename in tqdm(os.listdir(path_dataset)):

        if filename.split('_')[0] != 'domain':
            if int(filename.split('_')[0])<domain_1_age:
                shutil.move(path_dataset+"/"+filename, str(domain_1_path)+"/"+filename)
            else:
                shutil.move(path_dataset+"/"+filename, str(domain_2_path)+"/"+filename)

if __name__ == "__main__":
    split_domains()