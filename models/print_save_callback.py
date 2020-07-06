import os
from heapq import heappushpop, heappush, heappop
from keras import models
from keras.callbacks import Callback
from numpy import corrcoef, ndarray
from pandas import DataFrame

class PrintSaveCallback(Callback):

    def __init__(self, test_data: ndarray, test_target: ndarray, target_name: str, cluster):
        super().__init__()
        self.test_data = test_data
        self.test_target = test_target
        self.target_name = target_name
        self.cluster = cluster
        self.pq = []

    def on_epoch_end(self, epoch, logs=None):
        output = self.model.predict(self.test_data)
        df = DataFrame()
        df[self.target_name] = self.test_target
        df["pred"] = output

        score = corrcoef(
            df[self.target_name],
            df["pred"].rank(pct=True, method="first")
        )[0, 1]

        print("Score ", score)
        filename = "lstm_model_" + str(epoch) + "_" + str(self.cluster)
        to_save = (score, filename)

        if len(self.pq) > 2:
            lowest_score, to_delete = heappushpop(self.pq, to_save)
            if filename is to_delete:
                print("Score is too low to save.")
            else:
                print("Deleting model ", to_delete, " which has lowest score ", lowest_score)
                os.remove(to_delete)
                self.model.save(filename)
        else:
            heappush(self.pq, to_save)
            self.model.save(filename)

    def load_best_model(self):
        while len(self.pq) > 1:
            lower_score, to_delete = heappop(self.pq)
            print("Runner-up score: ", lower_score)
            os.remove(to_delete)
        winner_score, winner = heappop(self.pq)
        print("Winning score for cluster ", self.cluster, " is ", winner_score)
        print("Loading winner model from file ", winner)
        return models.load_model(winner)

