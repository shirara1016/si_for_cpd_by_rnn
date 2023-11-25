import numpy as np
import pickle
from source.models import one_step_predict_model
from source.nn import ChangesForMeanShift, ChangesForLinearTrend
from tqdm import tqdm

num_iter = 1020


def create_cov_matrix(size, rho=0.5):
    matrix = []
    for i in range(size):
        row = []
        for j in range(size):
            row.append(abs(i - j))
        matrix.append(row)
    cov = np.power(rho, matrix)
    return cov


def make_null_data(d, rng, predictor, mode, noise):
    dataset = []
    cov = create_cov_matrix(d)
    for _ in tqdm(range(2 * num_iter)):
        if noise == "iid":
            data = rng.normal(0, 1, d)
        elif noise == "ar":
            data = rng.multivariate_normal(np.zeros(d), cov)
        eta = predictor.construct_eta(data)
        if eta is None:
            continue
        dataset.append(data)
        if len(dataset) == num_iter:
            break
    with open(f"dataset/{mode}_sig{0.0}_d{d}_{noise}.pkl", "wb") as f:
        pickle.dump(dataset, f)


def make_alter_data(signal, rng, predictor, mode, noise):
    dataset = []
    cov = create_cov_matrix(60)
    for _ in tqdm(range(100 * num_iter)):
        if noise == "iid":
            data = rng.normal(0, 1, 60)
        elif noise == "ar":
            data = rng.multivariate_normal(np.zeros(60), cov)

        if mode == "ms":
            data[20:] += signal
            data[40:] += signal
        elif mode == "lt":
            data[20:40] += np.linspace(signal, signal * 20, 20, endpoint=True)
            data[40:] += signal * 20

        eta = predictor.construct_eta(data)
        cps = sorted(list(predictor.cps))
        if eta is None or len(cps) != 2:
            continue
        if cps[0] not in {17, 18, 19, 20, 21} or cps[1] not in {37, 38, 39, 40, 41}:
            continue
        dataset.append(data)
        if len(dataset) == num_iter:
            break
    with open(f"dataset/{mode}_sig{signal}_d{60}_{noise}.pkl", "wb") as f:
        pickle.dump(dataset, f)


def make_ms_null_dataset(ds=[40, 60, 80, 100]):
    rng = np.random.default_rng(0)
    model = one_step_predict_model()
    model.load_weights("model/ms2.h5")
    for noise in ["iid", "ar"]:
        for d in ds:
            make_null_data(d, rng, ChangesForMeanShift(model), "ms", noise)


def make_ms_alter_dataset(signals=[1.0, 2.0, 3.0, 4.0]):
    rng = np.random.default_rng(0)
    model = one_step_predict_model()
    model.load_weights("model/ms2.h5")
    for noise in ["iid", "ar"]:
        for signal in signals:
            make_alter_data(signal, rng, ChangesForMeanShift(model), "ms", noise)


def make_lt_null_dataset(ds=[40, 60, 80, 100]):
    rng = np.random.default_rng(0)
    model = one_step_predict_model()
    model.load_weights("model/lt2.h5")
    for noise in ["iid", "ar"]:
        for d in ds:
            make_null_data(d, rng, ChangesForLinearTrend(model), "lt", noise)


def make_lt_alter_dataset(signals=[0.1, 0.2, 0.3, 0.4]):
    rng = np.random.default_rng(0)
    model = one_step_predict_model()
    model.load_weights("model/lt2.h5")
    for noise in ["iid", "ar"]:
        for signal in signals:
            make_alter_data(signal, rng, ChangesForLinearTrend(model), "lt", noise)


if __name__ == "__main__":
    make_ms_null_dataset()
    make_ms_alter_dataset()
    make_lt_null_dataset()
    make_lt_alter_dataset()
