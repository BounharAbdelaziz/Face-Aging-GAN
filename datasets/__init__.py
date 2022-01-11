import os
from pathlib import Path
from tqdm import tqdm
import json

import shutil

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def add_label_to_img(list_features=['age','gender'], list_features_key=['faceAttributes','faceAttributes'], img_dir="../datasets/ffhq_mini/images/", labels_dir="../datasets/ffhq_mini/features/", is_debug=False, delete_when_enmpty_features=False):
    """
    This function adds labels to filename to facilitate data loading.
    @params:
        list_features : a list of features name to extract from the json features file. they will be added in order to the filename and separated with '_' 
        img_dir : path to the images
        labels_dir : path to the json files containing the features and labels.
    """

    print("-------------------------------------------")
    print(f'[INFO] Start processing FFHQ images...')
    print(f'[INFO] Total number of images : {len(os.listdir(img_dir))}')
    print(f'[INFO] Total number of features files : {len(os.listdir(labels_dir))}')
    print(f'[INFO] Features to be added : {list_features}')
    print("-------------------------------------------")

    # we do reverse to add the values in the original order of the selected features
    list_features = list_features[::-1]
    list_features_key = list_features_key[::-1] 

    unprocessed_files = []

    assert len(os.listdir(img_dir)) <= len(os.listdir(labels_dir)), f'[ERROR] The number of images should be less or equal than the number of files. Ther must be one feature file per image. Expected {len(os.listdir(img_dir))} files but found {len(os.listdir(labels_dir))}'

    for filename in tqdm(os.listdir(img_dir)):
        if is_debug:
            print(f'[INFO] filename : {filename}')
        
        # pass if already preprocessed (for now, for later; extract only filename without labels os.path.splitext(filename)[-1]...)
        if '_' in filename:
            continue


        with open(os.path.join(labels_dir, os.path.splitext(filename)[0]+".json")) as json_file:

            # Loading json metadata
            json_data = json.load(json_file)

            # avoid files where we don't have data
            if json_data == []:
                unprocessed_files.append(filename)
                if is_debug:
                    print(f'[WARNING] Found empty metadata file for image: {filename}')

                if delete_when_enmpty_features:
                    # we delete image and its json file
                    os.remove(os.path.join(img_dir,filename))
                    os.remove(os.path.join(labels_dir,os.path.splitext(filename)[0]+".json"))
                continue

            new_file_name = filename

            for feature_key, feature_name in zip(list_features_key, list_features):
                if feature_name != '':

                    # accessing features values
                    feature_value = json_data[0][feature_key][feature_name]

                    if feature_name == 'age':
                        feature_value = int(feature_value)
                    if feature_name == 'gender':
                        feature_value = feature_value[0].upper()

                    new_file_name = str(feature_value) + '_' + new_file_name

            # renaming the file
            os.rename(src=os.path.join(img_dir,filename), dst=os.path.join(img_dir,new_file_name))

            if is_debug:
                print(f'[INFO] new file name : {new_file_name}')
                print("-------------------------------------------")

    unprocessed_policy = 'deleted' if delete_when_enmpty_features else 'kept'
    
    print(f'[INFO] unprocessed_files : {unprocessed_files}')
    print(f'[INFO] Total unprocessed files : {len(unprocessed_files)}')
    print(f'[INFO] Unprocessed files has been {unprocessed_policy}.')
    print("-------------------------------------------")

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#

def split_domains( path_dataset="../datasets/UTKFace/", domain_1_age=39):

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
                shutil.copy(path_dataset+"/"+filename, str(domain_1_path)+"/"+filename)
            else:
                shutil.copy(path_dataset+"/"+filename, str(domain_2_path)+"/"+filename)

# -----------------------------------------------------------------------------#
# -----------------------------------------------------------------------------#