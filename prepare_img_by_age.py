from pathlib import Path
import shutil
import os
from tqdm import tqdm


def prepare_images_by_age( path_dataset="../datasets/ffhq_mini/images/", age_min=20, age_max=70):
    dataset_name = "age_"+str(age_min)+"_to_"+str(age_max)
    print(f'[INFO] Dataset name: {dataset_name}.')


    new_path = Path(path_dataset) / dataset_name
    try :
        new_path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        print(f'[INFO] Dataset already created.')
    else:
        print(f'[INFO] Dataset is being created...')
        
    for filename in tqdm(os.listdir(path_dataset)):

        if filename.split('_')[0] != 'age':

            in_age_min = int(filename.split('_')[0]) >= age_min
            in_age_max = int(filename.split('_')[0]) <= age_max

            if  in_age_min and in_age_max :
                shutil.move(path_dataset+"/"+filename, str(new_path)+"/"+filename)

if __name__ == "__main__":
    prepare_images_by_age()