import os
import cv2
import pandas as pd
from typing import List, Tuple
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

def normalize_save_image(in_filepath: str,
                    out_filepath: str,
                    size: int)-> None:
    # Read image
    image = cv2.imread(in_filepath)

    # Center Crop Resize
    image = cv2.resize(image, (size, size), interpolation=cv2.INTER_CUBIC)
 
    # Save image
    cv2.imwrite(out_filepath, image)

def extract_dir_to_df(root_dir: str)-> pd.DataFrame:
    """Collecting the file address of images from image directories 
    and store the 'class' and 'filepath' of the images in a DataFrame 
    format. This function doesn't load any image dataset, only extract
    class ID and file address.

    The image directories must be organized as follows:
        root_dir/
            001.class_name/
                class_name_images_01
                class_name_images_02
                ...
            002.class_name/
                class_name_images_01
                class_name_images_02
                ...
            {class_id}.class_name/
                class_name_images_01
                ...
            ...

    # Arguments:
        root_dir:  str. A path to root directory of the dataset.

    # Returns:
        dataset: Pandas DataFrame. 
        The first column is 'class', indicates the class ID.
        The second column is 'filepath', indicates the file address.
    """
    dirnames = os.listdir(root_dir)
    dirnames = [dire for dire in dirnames 
                if os.path.isdir(os.path.join(root_dir, dire))]
    data = {'class': [],
            'filepath': []}

    for dir_ in sorted(dirnames):
        clsname = dir_.split('.')[0]
        list_dir = os.listdir(os.path.join(root_dir, dir_))
        for fname in list_dir:
            fullpath = os.path.join(root_dir, dir_, fname)
            if not os.path.isfile(fullpath):
                continue
            data['class'].append(clsname)
            data['filepath'].append(fullpath)
    return pd.DataFrame(data)

def stratified_shuffle_split(data: pd.DataFrame,
                             test_size: float,
                             feature: List[str],
                             label: List[str],
                             )-> List[pd.DataFrame]:

    """Split dataset into train and test dataset. This is also can be 
    used to split dataset into train and validation dataset.
    
    `stratified_shuffle_split` uses `StratifiedShuffleSplit` from 
    scikit-learn.

    # Arguments:
        data: DataFrame. A dataset in Pandas DataFrame format.
        test_size: float. A fraction of test dataset.
        feature: List[str]. A list of column names.
        label: List[str]. A list of column names.

    # Returns:
        splitted dataset: List[DataFrame].
        A list of train and test dataset 
        [X_train, X_test, y_train, y_test]
    
    
    """
    FEATURE = feature
    LABEL = label

    spliter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=3)
    splitted = list(spliter.split(data[FEATURE], data[LABEL]))
    train_index = splitted[0][0]
    test_index = splitted[0][1]

    X_train = data[FEATURE].iloc[train_index].reset_index(drop=True)
    X_test = data[FEATURE].iloc[test_index].reset_index(drop=True)
    y_train = data[LABEL].iloc[train_index].reset_index(drop=True)
    y_test = data[LABEL].iloc[test_index].reset_index(drop=True)

    return X_train, X_test, y_train, y_test

def build_dataset(in_dir: str,
                  out_dir: str,
                  image_size: int)-> None:

    FEATURE = ['filepath']
    LABEL = ['class']
    
    # Extract filepath and class from raw CUB_200_2011 dataset
    df_data = extract_dir_to_df(in_dir)

    # Directory for train, test, and validation dataset
    train_data_dir = os.path.join(out_dir, 'train')
    test_data_dir = os.path.join(out_dir, 'test')
    val_data_dir = os.path.join(out_dir, 'test')

    # Test size is 20% from total dataset
    # Validation size is 20% from training set (16% from total dataset)
    test_size = 0.2
    val_size = 0.2

    # Split train and test dataset
    
    X_train, X_test, y_train, y_test = stratified_shuffle_split(df_data,
                                    test_size=test_size,
                                    feature=FEATURE,
                                    label=LABEL)
    
    df_data = pd.concat([X_train, y_train], axis=1)

    X_train, X_val, y_train, y_val = stratified_shuffle_split(df_data,
                                    test_size=test_size,
                                    feature=FEATURE,
                                    label=LABEL)

    in_data_mode = {'train': X_train,
                  'val': X_val,
                  'test': X_test}

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    else:
        print(f"Warning: output dir {out_dir} already exists.")


    out_data_mode = {'train': [],
                  'val': [],
                  'test': []}


    for split_mode in ['train', 'val', 'test']:
        print(f"Processing {split_mode} dataset.")
        out_dir_split = os.path.join(out_dir, split_mode)
        if not os.path.exists(out_dir_split):
            os.makedirs(out_dir_split)
        else:
            print(f"Warning: dir {out_dir_split} already exists.")
            print(f"This will lead to potential overwriting.")

        for filepath in tqdm(in_data_mode[split_mode].values):
            in_filepath = filepath[0]
            filename = os.path.basename(in_filepath)

            out_filepath = os.path.join(out_dir_split, filename)
            normalize_save_image(in_filepath, out_filepath, image_size)
            out_data_mode[split_mode].append(out_filepath)

    tmp_series = pd.Series(out_data_mode['train']).rename('filepath')
    out_train = pd.concat([tmp_series, y_train], 
                        axis=1)
    tmp_series = pd.Series(out_data_mode['val']).rename('filepath')
    out_val = pd.concat([tmp_series, y_val],
                        axis=1)
    tmp_series = pd.Series(out_data_mode['test']).rename('filepath')
    out_test = pd.concat([tmp_series, y_test],
                        axis=1)
    
    out_train.to_csv(os.path.join(out_dir, 'train.csv'))
    out_val.to_csv(os.path.join(out_dir, 'val.csv'))
    out_test.to_csv(os.path.join(out_dir, 'test.csv'))

    print("Done building dataset")

def build_dataset_label(in_dir: str,
                  out_dir: str,
                  image_size: int)-> None:

    FEATURE = ['filepath']
    LABEL = ['class']
    
    # Extract filepath and class from raw CUB_200_2011 dataset
    df_data = extract_dir_to_df(in_dir)

    # Directory for train, test, and validation dataset
    train_data_dir = os.path.join(out_dir, 'train')
    test_data_dir = os.path.join(out_dir, 'test')
    val_data_dir = os.path.join(out_dir, 'test')

    # Test size is 20% from total dataset
    # Validation size is 20% from training set (16% from total dataset)
    test_size = 0.2
    val_size = 0.2

    # Split train and test dataset
    
    X_train, X_test, y_train, y_test = stratified_shuffle_split(df_data,
                                    test_size=test_size,
                                    feature=FEATURE,
                                    label=LABEL)
    
    df_data = pd.concat([X_train, y_train], axis=1)

    X_train, X_val, y_train, y_val = stratified_shuffle_split(df_data,
                                    test_size=test_size,
                                    feature=FEATURE,
                                    label=LABEL)

    in_data_mode = {'train': (X_train, y_train),
                  'val': (X_val, y_val),
                  'test': (X_test, y_test)}

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    out_data_mode = {'train': [],
                  'val': [],
                  'test': []}

    for split_mode in ['train', 'val', 'test']:
        print(f"Processing {split_mode} dataset.")
        out_dir_split = os.path.join(out_dir, split_mode)
        if not os.path.exists(out_dir_split):
            os.makedirs(out_dir_split)

        for filepath, label in tqdm(zip(in_data_mode[split_mode][0].values,
                             in_data_mode[split_mode][1].values)):
            in_filepath = filepath[0]
            in_label = label[0]
            filename = os.path.basename(in_filepath)
            if not os.path.exists(os.path.join(out_dir_split, in_label)):
                os.makedirs(os.path.join(out_dir_split, in_label))

            out_filepath = os.path.join(out_dir_split, in_label, filename)
            
            normalize_save_image(in_filepath, out_filepath, image_size)
            out_data_mode[split_mode].append(out_filepath)

    tmp_series = pd.Series(out_data_mode['train']).rename('filepath')
    out_train = pd.concat([tmp_series, y_train], 
                        axis=1)
    tmp_series = pd.Series(out_data_mode['val']).rename('filepath')
    out_val = pd.concat([tmp_series, y_val],
                        axis=1)
    tmp_series = pd.Series(out_data_mode['test']).rename('filepath')
    out_test = pd.concat([tmp_series, y_test],
                        axis=1)
    
    out_train.to_csv(os.path.join(out_dir, 'train.csv'))
    out_val.to_csv(os.path.join(out_dir, 'val.csv'))
    out_test.to_csv(os.path.join(out_dir, 'test.csv'))

    print("Done building dataset")
