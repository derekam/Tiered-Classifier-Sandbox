from keras import Sequential, optimizers, models, backend
from keras.layers import Bidirectional, LSTM, Dropout, Dense
from keras.losses import mean_squared_error
from pandas import DataFrame
import numpy as np
import gc

from models.print_save_callback import PrintSaveCallback


def train_models(training_data: DataFrame, tournament_data: DataFrame,
                         identifier: str, cluster_label: str):
    """Example function with types documented in the docstring.
    TODO don't use tes data for print_save_callback.
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

    model = None

    for cluster in clusters:
        gc.collect()
        train_data, train_target = extract_data(training_data, cluster_label, cluster, feature_names, target_name)
        test_data, test_target = extract_data(tournament_data, cluster_label, cluster, feature_names, target_name)
        validation_data, validation_target = extract_data(tournament_data[tournament_data.data_type == "validation"], cluster_label, cluster, feature_names, target_name)

        if test_data is not None and validation_data is not None and train_data is not None and len(validation_target) != 0:
            model_name = train_model(train_data, train_target, validation_data, validation_target, cluster, target_name)
            print("Loading winner model from file ", model_name)
            model = models.load_model(model_name)
            training_data.loc[training_data[cluster_label] == cluster, prediction_name], tournament_data.loc[tournament_data[cluster_label] == cluster, prediction_name] = model.predict(train_data), model.predict(test_data)
            del model
            backend.clear_session()

    tournament_data[prediction_name].to_csv(identifier + "_submission.csv", header=True)

def extract_data(training_data, cluster_label, cluster, feature_names, target_name):
    if len(training_data) == 0:
        return None, None
    cluster_train_data: DataFrame = training_data.loc[training_data[cluster_label] == cluster]
    train_data, train_target = cluster_train_data[feature_names].values, cluster_train_data[target_name].values
    train_data = np.reshape(train_data, (train_data.shape[0], 1, train_data.shape[1]))
    return train_data, train_target

def train_model(train_data: np.ndarray, train_target: np.ndarray,
                test_data: np.ndarray, test_target: np.ndarray, cluster, target_name) -> str:
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

    model.add(Bidirectional(LSTM(units=310, return_sequences=True)))

    model.add(Bidirectional(LSTM(units=155, return_sequences=True)))

    model.add(Bidirectional(LSTM(units=50)))


    model.add(Dense(units=1))

    opt = optimizers.Adam(learning_rate=0.00001)

    model.compile(loss=mean_squared_error, optimizer=opt)

    callback = PrintSaveCallback(test_data, test_target, target_name, cluster)
    print("Beginning training for cluster ", cluster)
    model.fit(train_data, train_target, epochs=50, batch_size=128, verbose=1, callbacks=[callback], validation_data=(test_data, test_target))
    model.save("lstm_model_final_" + str(cluster))

    del model

    return callback.best_model()

