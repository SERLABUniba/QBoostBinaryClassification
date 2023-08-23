from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


def split_dataset(X, y):
    return train_test_split(X, y, test_size=0.3, random_state=42)


def scale_dataset(x, scaler=None):

    if scaler is None:
        mms = MinMaxScaler()
        return mms.fit_transform(x), scaler
    else:
        return scaler.transform(x)
