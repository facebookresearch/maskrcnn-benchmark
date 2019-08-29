import subprocess
import warnings
from typing import List

import numpy as np
import pandas as pd
import torch
from skimage.io import imread


def format_box(bbox):
    """
    This is a helper function because geometry's are in the format [{'x':x,'y':y}]
    :param bbox: list of dicts
    :return: returns list of lists [[x,y],...]
    """
    box_points = [bbox[0]['x'],
                  bbox[0]['y'],
                  bbox[2]['x'],
                  bbox[2]['y']]
    return box_points


def label_transform(some_label):
    """
    takes in a label and if that label contains the word in the keys of the transform dict it maps it to that value. If
    the label is not in the key it outputs the original label
    :param some_label: input label which is a string
    :return: New mapped label or the original input label
    """
    transform_dict = {
        'Defective Restoration': 'Defective Restoration',
        'Non-ideal Restoration': 'Defective Restoration',
    }

    for key in transform_dict.keys():
        if key in some_label:
            return transform_dict[key]
    return some_label


def parse_dataframe(df, label_types, transform=None):
    temp = {idx: {label: [] for label in label_types} for idx in range(len(df))}

    for idx in range(len(df)):
        # this gives all the labels present (Tooth, Crown, Filling, etc.)
        for label in df.label[idx]:
            # list of property dicts (each one has a geometry associated with it)
            # this is a list of dicts that all have geometry and other keys
            for property_dict in df.label[idx][label]:
                # Create new columns as needed for types
                if type(property_dict) == str:
                    try:
                        df.at[idx, property_dict] = 1
                    except:
                        df[property_dict] = 0
                        df.at[idx, property_dict] = 1

                else:
                    # we check to see if there are any other sub properties besides geometry
                    if len(property_dict) == 1 and 'geometry' in property_dict.keys():
                        if len(property_dict['geometry']) > 0:
                            # there is only one property and it is geometry
                            # add polygon (list of lists)
                            if transform:
                                label = transform(label)
                            if label in label_types:
                                temp[idx][label].append(format_box(property_dict['geometry']))

                    # Elseif we have a sublabel
                    elif len(property_dict) > 1:
                        # for all the subproperties except geometry we get the sublabel
                        for subproperty in property_dict.keys():  # subproperty would be geometry or 'tooth_number_palmer'
                            if subproperty != 'geometry':
                                #                                 subproperty_label = property_dict[subproperty] # e.g. Filling, confidence, should restoration be redone
                                #                                 label_and_subproperty = label + '_' + str(subproperty_label)
                                if transform:
                                    #                                     label_and_subproperty = transform(label_and_subproperty)
                                    label = transform(label)
                                # we format the sublabel with the label
                                #                                 if label_and_subproperty in label_types:
                                #                                     temp[idx][label_and_subproperty].append(format_box(property_dict['geometry']))

                                if label in label_types:
                                    temp[idx][label].append(format_box(property_dict['geometry']))

                    else:
                        print('something went wrong, property dict.keys had only one key and it was not geometry')
                        break

    temp_df = pd.DataFrame(temp).transpose()
    df = pd.concat([df, temp_df], axis=1)

    return df


def clean_dataframe(df):
    # Combine similar columns
    try:
        df['no_finding'] = df['no_finding'].fillna(df['has_no_finding'])
        df = df.drop(columns=['has_no_finding'])
    except KeyError:
        # Already dropped or does not exist
        pass

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    dataset_names = df['dataset_name'].unique()
    print(f"Datasets within json: {dataset_names}")

    # Initialize empty columns
    df['patient_id'] = np.nan
    df['patient'] = np.nan
    df['image_id'] = np.nan
    df['clinic_id'] = np.nan

    # Extract patient, image, clinic IDs (for ranges, the first number is used for all patients)
    for name in dataset_names:
        if name == 'disease_data_fp_7_23_2019':
            # I'm sure there is a clever way of setting this but I couldn't figure it out
            temp = df.loc[df['dataset_name'] == name]['external_id'].str.extractall(
                r'([0-9]+)~([0-9]+).jpg.[0-9]+.([0-9]+)')

            # Patient ID is keyed to the clinic
            df.loc[df['dataset_name'] == name, ['patient_id']] = np.array(temp)[:, 2] + '_' + np.array(temp)[:, 0]

            # Image ID requires clinic and patient IDs to be unique
            df.loc[df['dataset_name'] == name, ['image_id']] = np.array(temp)[:, 2] + '_' + np.array(temp)[:,
                                                                                            0] + '_' + np.array(temp)[:,
                                                                                                       1]

            # Not entirely necessary but might as well save it
            df.loc[df['dataset_name'] == name, ['clinic_id']] = np.array(temp)[:, 2]

        else:
            warnings.warn(f"{name} not parsed! Assumed to be one giant patient. Write a new case in clean_dataframe")

    print(f'Number of unique patients: {df["patient_id"].nunique()}')

    return df


def balance_dataset(df, class_names, split=0.3, continuing=False):
    if continuing:
        # Make sure to use the same dataset if continuing to train on the same model
        return df
    else:
        patients = list(df['patient_id'].unique())
        num_unique_total = len(patients)  # Number of unique patients in entire dataset

        # Creating data indices for training and validation splits:
        n_val = int(num_unique_total * split)  # number of test/val elements
        n_train = num_unique_total - n_val

        n_val_images = int(len(df) * split)
        n_train_images = len(df) - n_val_images

        # Create phase column as we will be shuffling and splitting the dataframe
        df['phase'] = np.nan

        # Just use the presence of classes within the image to determine balance for now.
        pos_weight = torch.Tensor([df[cl].notna().sum() / df.shape[0] for cl in class_names])
        desired_class_balance = pos_weight.numpy()

        # Keep shuffling until I get a reasonably-sized train-test split
        shuffle_dataset = True
        random_seed = 42

        # The actual number of images in each subset
        true_n_val = 1
        true_n_train = 1

        train_patients = patients[:n_train]
        train_indices = df.loc[df['patient_id'].isin(train_patients)]

        val_patients = patients[n_train:]
        val_indices = df.loc[df['patient_id'].isin(val_patients)]

        # Each subset needs to have relatively balanced classes
        val_fraction = [val_indices[cl].notna().sum() / true_n_val for cl in class_names]
        train_fraction = [train_indices[cl].notna().sum() / true_n_train for cl in class_names]

        # Make sure there are enough images in the unaugmented validation set
        while (true_n_val < n_val_images
               or np.abs(val_fraction - desired_class_balance).mean() > 0.1
               or np.abs(train_fraction - desired_class_balance).mean() > 0.1):
            if shuffle_dataset:
                np.random.seed(random_seed)
                np.random.shuffle(patients)

            train_patients = patients[:n_train]
            train_indices = df[df['patient_id'].isin(train_patients)]

            val_patients = patients[n_train:]
            val_indices = df[df['patient_id'].isin(val_patients)]

            true_n_val = len(val_indices)
            true_n_train = len(train_indices)

            val_fraction = [val_indices[cl].notna().sum() / true_n_val for cl in class_names]
            train_fraction = [train_indices[cl].notna().sum() / true_n_train for cl in class_names]

        print(class_names)
        print('Val fraction:\n', [np.round(x, 2) for x in val_fraction])
        print('Train fraction:\n', [np.round(x, 2) for x in train_fraction])
        print('Patient split:\n', len(val_patients), len(train_patients))
        print('Image split:\n', true_n_val, true_n_train)

        train_indices.loc[:, 'phase'] = 'train'
        val_indices.loc[:, 'phase'] = 'val'

        # Recombine the dataframe
        df = pd.concat([train_indices, val_indices])

        return df


def check_downloaded(df, directory):
    """
    this function gets the images in the download directory and checks to see what is in the json. It returns a list
    of images that need to be downloaded with the urls that host them. Cleaned data is a dictionary that comes out of
    parse_json and raw_json is the json before it is cleaned up. the reason we do this is because raw_json has the label
    in it where as the cleaned data only has the labels.
    :param df: DataFrame containing image labels and paths
    :param directory: this is the top-level the directory you want to store your images
    :return: this is a pandas DataFrame
    """
    all_files = []
    for i in directory.rglob('*.*'):
        all_files.append((i.name, i.parent, i))

    columns = ["external_id", "parent", "image_path"]
    temp = pd.DataFrame.from_records(all_files, columns=columns)

    # Save filepath of existing images, download missing
    df = pd.merge(df, temp, how='left', on='external_id')

    return df


def download_imgs(df, download_dir):
    """
    This is a dataframe with the img_filename and url stored
    it adds the download directory to the image filename and called get_img_from_url
    :param download_dict: dict
    :param download_dir: file path
    :return:
    """
    count = 0

    df['image_path'] = (df['dataset_name'] + '_' + df['patient_id'] + '.jpg').apply(lambda x: Path(download_dir, x))

    for i in range(len(df)):
        # Get download url of image
        get_img_from_url(df.at[i, 'image_path'], df.at[i, 'labeled_data'])
        if (count + 1) % 10 == 0:
            print(f'Downloaded {count + 1} / {len(df)}')
        count += 1

    return df


def get_img_from_url(filename, url):
    """
    This takes in a full image path name and a url and downloads it using WGET
    :param filename: This is the full path to the image you want to save
    :param url: URL holding an image
    :return: Nothing
    """
    try:
        subprocess.check_output(['wget', '-O', filename, url])
        imread(filename)
    except Exception as E:
        print(E)
        print('File Downloaded FAIL: {}'.format(filename))


def massage_labels(df: object, class_names: List) -> object:

    df['all_boxes'] = np.nan
    df['all_boxes'] = df['all_boxes'].astype(object)

    df['all_labels'] = np.nan
    df['all_labels'] = df['all_labels'].astype(object)

    for i in range(len(df)):
        full = []
        labels = []
        for j, cl in enumerate(class_names):
            to_add = df.at[i, cl]
            if type(to_add) == list:
                if type(to_add[0]) == int:
                    full.append(to_add)
                    labels.append(j)
                elif type(to_add[0]) == list:

                    for j in range(len(to_add)):
                        full.append(to_add[j])
                        labels.append(j)

        df.at[i, 'all_boxes'] = full
        df.at[i, 'all_labels'] = labels

    return df


def create_disease_dataset(json_name, dataroot):
    label_types = ['Calculus', 'Primary Decay', 'Secondary Decay', 'Defective Restoration']
    class_names = [name.strip().lower().replace(' ', '_') for name in label_types]

    # import the json data from the json file. We will use this data to generate the label masks
    with open(Path(dataroot, f'{json_name}.json')) as f:
        raw_data = json.load(f)

    # Load into a dataframe
    df = pd.DataFrame(raw_data)

    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # For now
    df['dataset_name'] = json_name

    # we need to clean the json data to create a dataframe where we can get the label by using the External ID
    df = parse_dataframe(df, label_types, label_transform)

    # Clean up labels and extract patient IDs
    df = clean_dataframe(df)

    df = balance_dataset(df, class_names, split=0.3, continuing=False)

    # directory for downloaded images
    download_dir = Path(dataroot, 'imgs')

    # Create directory structure if it does not exist
    download_dir.mkdir(parents=True,
                       exist_ok=True)

    #     # get a list of images that have not been downloaded
    #     to_download = check_downloaded(df, download_dir)

    # download images and update dataframe
    df = download_imgs(df, download_dir)

    df = massage_labels(df)

    return df, label_types
