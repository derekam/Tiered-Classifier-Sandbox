from numpy import corrcoef
from pandas import DataFrame
from filehandlers.input_handler import read_csv_predictions

def score_from_file(file_name: str, tournament_data: DataFrame, prediction_name: str, target_name: str):
    # Generate predictions on both training and tournament data
    print("Loading predictions...")
    predictions = read_csv_predictions(file_name)
    tournament_data[prediction_name] = predictions[prediction_name]
    score_tournament_frame(tournament_data, prediction_name, target_name)


def score_from_frames(training_data: DataFrame, tournament_data: DataFrame, prediction_name: str, target_name: str):
    # Check the per-era correlations on the training set (in sample)
    train_correlations = training_data.groupby("era").apply(lambda x: score(x, prediction_name, target_name))
    print(f"On training the correlation has mean {train_correlations.mean()} and std {train_correlations.std()}")
    print(f"On training the average per-era payout is {payout(train_correlations).mean()}")
    score_tournament_frame(tournament_data, prediction_name, target_name)


def score_tournament_frame(tournament_data: DataFrame, prediction_name: str, target_name: str):
    # Check the per-era correlations on the validation set (out of sample)
    validation_data = tournament_data[tournament_data.data_type == "validation"]
    validation_correlations = validation_data.groupby("era").apply(lambda x: score(x, prediction_name, target_name))
    print(f"On validation the correlation has mean {validation_correlations.mean()} and std {validation_correlations.std()}")
    print(f"On validation the average per-era payout is {payout(validation_correlations).mean()}")


# Submissions are scored by spearman correlation
def score(df, prediction_name: str, target_name: str):
    # method="first" breaks ties based on order in array
    pct_ranks = df[prediction_name].rank(pct=True, method="first")
    targets = df[target_name]
    return corrcoef(targets, pct_ranks)[0, 1]


# Payout is just the score cliped at +/-25%
def payout(scores):
    return scores.clip(lower=-0.25, upper=0.25)