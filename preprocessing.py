import pandas as pd
from imblearn.under_sampling import RandomUnderSampler


def load_dataset(path):

    HEADER = ['Timestamp', 'CanId', 'DLC', 'D0', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'Label']
    dataset = pd.read_csv(path, names=HEADER, low_memory=False)

    dataset = dataset.drop(dataset[dataset['DLC'] != 8].index)

    label = dataset.columns[dataset.shape[1] - 1]

    # non-malicious 1 malicious -1
    dataset[label] = [1 if x == 'R' else -1 for x in dataset[label].values]

    return dataset


def hex_to_int(val):
    """
    Convert the values of specific columns into decimal form
    :param val: value to be converted
    :return: value into decimal form
    """
    return int(str(val), 16)


def convert_can_hex_to_dec(dataset):

    dataset['CanId'] = dataset['CanId'].apply(hex_to_int)

    return dataset


def convert_data_hex_to_dec(dataset):

    for i in range(0, 8):
        dataset[f"D{int(i)}"] = dataset[f"D{int(i)}"].apply(hex_to_int)

    return dataset


def drop_features(dataset: pd.DataFrame, features: list):
    """
    List of features to be dropped
    :param dataset: dataset to be processed
    :param features: List of features
    :return: dataset without the features
    """

    return dataset.drop(columns=features)


def init_preprocessing(path):

    dataset = load_dataset(path)

    label = dataset.columns[dataset.shape[1] - 1]

    dataset = convert_data_hex_to_dec(dataset)

    dataset = convert_can_hex_to_dec(dataset)

    dataset = dataset.drop(columns=['Timestamp', 'DLC'])

    # rus = RandomUnderSampler(random_state=42)
    # X, y = rus.fit_resample(dataset.drop(columns=[label]), dataset[label])
    #
    # df = pd.concat([X, y], axis=1)

    return dataset
