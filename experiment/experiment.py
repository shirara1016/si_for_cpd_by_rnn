#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

# print(sys.path)

from abc import ABCMeta, abstractmethod
from concurrent.futures import ProcessPoolExecutor

# import mlflow
from tqdm import tqdm

import numpy as np
import tensorflow as tf
from source.models import one_step_predict_model
from source.nn import ChangesForLinearTrend, ChangesForMeanShift
from sicore import SelectiveInferenceNorm, InfiniteLoopError


def create_cov_matrix(size, rho=0.5):
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(abs(i - j))
        matrix.append(row)
    cov = np.power(rho, matrix)
    return cov


class PararellExperiment(metaclass=ABCMeta):
    def __init__(self, num_iter: int, num_results: int, num_worker: int):
        self.num_iter = num_iter
        self.num_results = num_results
        self.num_worker = num_worker

    @abstractmethod
    def iter_experiment(self, args) -> tuple:
        """Run each iteration of the experiment

        Args:
            args (tuple): Tuple of data for each iteration

        Returns:
            tuple: Tuple of results of each iteration
        """
        pass

    def experiment(self, dataset: list) -> list:
        """Execute all iterations of the experiment

        Args:
            dataset (list): List of args for iter_experiment method, the size of which must be equal to num_iter

        Returns:
            list: List of results from iter_experiment method
        """

        with ProcessPoolExecutor(max_workers=self.num_worker) as executor:
            results = list(
                tqdm(executor.map(self.iter_experiment, dataset), total=self.num_iter)
            )
            results = [result for result in results if result is not None]
            return results[: self.num_results]

    @abstractmethod
    def run_experiment(self):
        pass


class ChangesExperiment(PararellExperiment):
    def __init__(
        self,
        num_results: int,
        num_worker: int,
        signal: float,
        d: int,
        mode: str,
        noise: str,
    ):
        super().__init__(int(num_results * 1.02), num_results, num_worker)
        self.signal = signal
        self.d = d
        self.mode = mode
        self.noise = noise

        if self.noise == "ar":
            self.cov = create_cov_matrix(self.d)
        elif self.noise == "iid":
            self.cov = 1.0
        else:
            raise Exception('noise must be "ar" or "iid".')

    def iter_experiment(self, args) -> tuple:
        data = tf.constant(args, dtype=tf.float64)

        model = one_step_predict_model()
        model.load_weights(f"model/{self.mode}2.h5")

        if self.mode == "ms":
            predictor = ChangesForMeanShift(model)
        elif self.mode == "lt":
            predictor = ChangesForLinearTrend(model)
        else:
            raise Exception('mode must be "ms" or "lt".')

        etas = predictor.construct_eta(data)

        try:
            para_res = []
            oc_res = []
            for eta in etas:
                si = SelectiveInferenceNorm(data, self.cov, eta, use_tf=True)

                result = si.inference(
                    predictor.algorithm,
                    predictor.model_selector,
                    termination_criterion="decision",
                    significance_level=0.05,
                )
                para_res.append(result)

                result = si.inference(
                    predictor.algorithm,
                    predictor.model_selector,
                    over_conditioning=True,
                )
                oc_res.append(result)

        except InfiniteLoopError:
            return None
        except Exception as e:
            print(e)
            return None

        return [para_res, oc_res]

    def run_experiment(self):
        with open(
            f"dataset/{self.mode}_sig{self.signal}_d{self.d}_{self.noise}.pkl", "rb"
        ) as f:
            dataset = pickle.load(f)

        results = self.experiment(dataset)
        results = [result for result in results if result is not None]
        results = results[: self.num_results]

        self.results = {}
        self.results["proposed"] = [result[0] for result in results]
        self.results["oc"] = [result[1] for result in results]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_results", type=int, default=1000)
    parser.add_argument("--num_worker", type=int, default=32)
    parser.add_argument("--signal", type=float, default=0.0)
    parser.add_argument("--d", type=int, default=60)
    parser.add_argument("--mode", type=str, default="ms")
    parser.add_argument("--noise", type=str, default="iid")
    args = parser.parse_args()

    experiment = ChangesExperiment(
        args.num_results, args.num_worker, args.signal, args.d, args.mode, args.noise
    )
    experiment.run_experiment()

    result_path = "results"
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    file_name = f"{args.mode}_sig{args.signal}_d{args.d}_{args.noise}.pkl"
    file_path = os.path.join(result_path, file_name)

    print(file_name)

    with open(file_path, "wb") as f:
        pickle.dump(experiment.results, f)
