from keras import Sequential
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from keras.losses import mean_squared_error
from pandas import DataFrame
import numpy as np

from models.print_save_callback import PrintSaveCallback


def train_models(training_data: DataFrame, tournament_data: DataFrame,
                         identifier: str, cluster_label: str):
    """Example function with types documented in the docstring.

    Args:
        training_data (DataFrame): The training data, including only features and target. (only use features though...)
        tournament_data (DataFrame): The testing data, including only features and target. (only use features though...)
        cluster_label: the name of the column with cluster membership data.
        identifier: identifying name for output files.

    Returns:
        bool: The return value. True for success, False otherwise

    """
    print("Training model")

    target_name = f"target_{identifier}"
    prediction_name = f"prediction_{identifier}"

    clusters = set(training_data[cluster_label])
    feature_names = [f for f in training_data.columns if f.startswith("feature")]

    for cluster in clusters:
        cluster_train_data: DataFrame = training_data.loc[training_data[cluster_label] == cluster]
        cluster_tournament_data: DataFrame = tournament_data.loc[tournament_data[cluster_label] == cluster]
        cluster_validation_data = tournament_data[tournament_data.data_type == "validation"]
        cluster_validation_data = cluster_validation_data.loc[cluster_validation_data[cluster_label] == cluster]

        train_data, train_target = cluster_train_data[feature_names].values, cluster_train_data[target_name].values
        test_data, test_target = cluster_tournament_data[feature_names].values, cluster_tournament_data[target_name].values
        validation_data, validation_target = cluster_validation_data[feature_names].values, cluster_validation_data[target_name].values

        train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
        test_data = np.reshape(test_data, (test_data.shape[0], 1, test_data.shape[1]))
        validation_data = np.reshape(validation_data, (validation_data.shape[0], 1, validation_data.shape[1]))

        model = train_model(train_data, train_target, validation_data, validation_target, cluster, target_name)

        training_data.loc[training_data[cluster_label] == cluster, prediction_name], tournament_data.loc[tournament_data[cluster_label] == cluster, prediction_name] = model.predict(train_data), model.predict(test_data)

    tournament_data[prediction_name].to_csv(identifier + "_submission.csv", header=True)


def train_model(train_data: np.ndarray, train_target: np.ndarray,
                test_data: np.ndarray, test_target: np.ndarray, cluster, target_name):
    """
    :param cluster:
    :param target_name:
    :param train_data:
    :param train_target:
    :param test_data:
    :param test_target:
    :return: ndarray. the predictions for the given group.
    """

    model = Sequential()

    model.add(LSTM(units=310, input_shape=(train_data.shape[1], 310), return_sequences=True, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Bidirectional(LSTM(units=310, return_sequences=True)))
    model.add(Dropout(0.3))

    model.add(Bidirectional(LSTM(units=155, return_sequences=True)))
    model.add(Dropout(0.2))

    model.add(Bidirectional(LSTM(units=50)))
    model.add(Dropout(0.1))

    model.add(Dense(units=1))

    model.compile(loss=mean_squared_error, optimizer='adam')

    callback = PrintSaveCallback(test_data, test_target, target_name, cluster)
    print("Beginning training for cluster ", cluster)
    model.fit(train_data, train_target, epochs=50, batch_size=128, verbose=1, callbacks=[callback], validation_data=(test_data, test_target))
    model.save("lstm_model_final_" + str(cluster))

    return callback.load_best_model()

