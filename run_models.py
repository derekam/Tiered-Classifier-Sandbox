import argparse
import sys
from models.lstm import train_models
from scoring.scorer import score_from_file, score_from_frames
from pandas import DataFrame
from filehandlers.input_handler import read_csv_full
from models.clustering import classify_by_clusters

sys.path.append('../')
sys.path.append('./')

"""
 - load data to and score example submission then trash the data to free memory
 - load just the data you need in wanted format
 - create clustering models and train in separate process that then
    - classifies data according to model and returns the classified data somehow
 - train each subset on own lstm model in own process and save final submodels to disk
 - generate final answers for all data and score
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run defined model on supplied data.')
    parser.add_argument('-training_file', dest="training_file", type=str, help='the full filename of the csv training file',
                        default='numerai_training_data.csv')
    parser.add_argument('-validation_file', dest="validation_file", type=str, help='the full filename of the csv testing file',
                        default='numerai_tournament_data.csv')
    parser.add_argument('-example_predictions', dest="example_predictions", type=str, help='the full filename of the csv example predictions file',
                        default=None)
    parser.add_argument('-identifier', dest="identifier", type=str, help='identifier to use in the output data', default='kazutsugi')
    parser.add_argument('-cluster_label', dest="cluster_label", type=str, help='identifier to use in the output data', default='CLUSTER')
    parser.add_argument('-n_clusters', dest="n_clusters", type=int, help='number of clusters to cluser into', default=1)
    parser.add_argument('-cluster_sensitivity', dest="cluster_sensitivity", type=int, help='number of clusters to cluser into', default=1)

    args: argparse.Namespace = parser.parse_args()

    print(args.n_clusters)

    print("Loading data...")

    training_data: DataFrame = read_csv_full(args.training_file)
    tournament_data: DataFrame = read_csv_full(args.validation_file)

    target_name = f"target_{args.identifier}"
    prediction_name = f"prediction_{args.identifier}"
    print("Example score to beat: ")
    if args.example_predictions is not None:
        score_from_file(args.example_predictions, tournament_data, prediction_name, target_name)

    #  - create clustering models and train in separate process that then
    #     - classifies data according to model and returns the classified data somehow
    classify_by_clusters(training_data, tournament_data, args.n_clusters, args.cluster_label)

    train_models(training_data, tournament_data, args.identifier, args.cluster_label)

    print("Achieved score: ")
    score_from_frames(training_data, tournament_data, target_name, prediction_name)

