import numpy as np
from sklearn.preprocessing import StandardScaler

def split_df_by_asset(df, drop_label_column = True, num_columns_per_asset = 3):
    if(drop_label_column):
        df = df.drop(columns = ['label'])
    asset_dfs = []
    for i in range(0, len(df.columns), num_columns_per_asset):
        asset_dfs.append(df[df.columns[i : i + num_columns_per_asset]])
    return asset_dfs

def train_test_split_lstm(stocks, prediction_time=1, test_data_size=450, unroll_length=50):
    """
        Split the data set into training and testing feature for Long Short Term Memory Model
        :param stocks: whole data set containing ['Open','Close','Volume'] features
        :param prediction_time: no of days
        :param test_data_size: size of test data to be used
        :param unroll_length: how long a window should be used for train test split
        :return: X_train : training sets of feature
        :return: X_test : test sets of feature
        :return: y_train: training sets of label
        :return: y_test: test sets of label
    """
    # training data
    test_data_cut = test_data_size + unroll_length + 1

    x_train = stocks[0:-prediction_time - test_data_cut]
    #y_train = stocks[prediction_time:-test_data_cut]['Close'].as_matrix()
    y_train = stocks[prediction_time:-test_data_cut]

    # test data
    x_test = stocks[0 - test_data_cut:-prediction_time]
    #y_test = stocks[prediction_time - test_data_cut:]['Close'].as_matrix()
    y_test = stocks[prediction_time - test_data_cut:]

    return x_train, x_test, y_train, y_test


def unroll(data, sequence_length=24):
    """
    use different windows for testing and training to stop from leak of information in the data
    :param data: data set to be used for unrolling
    :param sequence_length: window length
    :return: data sets with different window.
    """
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
    return np.asarray(result)


def standardise_df(df):
    scalers = [] #save the scalers so we can inverse_transform later
    for column in df.columns:
        x = np.array(df[column])
        scaler = StandardScaler()
        scaler.fit(x.reshape(len(x), 1))
        df[column] = scaler.transform(x.reshape(len(x), 1))
        scalers.append(scaler)
    return df, scalers