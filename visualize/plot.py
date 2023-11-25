#!/usr/bin/python
# -*- coding: utf-8 -*-

import pickle
import argparse
import itertools

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt

import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, ".."))

plt.rcParams["font.size"] = 18

with open("results/lasso_results.pkl", "rb") as f:
    lasso_dict = pickle.load(f)


def get_reject_rates(mode, signal, d, noise="iid"):
    file_name = f"{mode}_sig{signal}_d{d}_{noise}.pkl"
    file_path = os.path.join("results", file_name)

    with open(file_path, "rb") as f:
        results_dict = pickle.load(f)

    results = results_dict["proposed"]
    prop_plist = []
    for result in results:
        for r in result:
            prop_plist.append(r.p_value)
    prop_reject = np.mean([p < 0.05 for p in prop_plist])

    oc_results = results_dict["oc"]
    oc_plist = []
    for result in oc_results:
        for r in result:
            oc_plist.append(r.p_value)
    oc_reject = np.mean([p < 0.05 for p in oc_plist])

    naive_plist = []
    for result in results:
        for r in result:
            naive_plist.append(2 * norm.cdf(-np.abs(r.stat)))
    naive_reject = np.mean([p < 0.05 for p in naive_plist])

    key = f"{mode}_d{d}_{noise}" if signal == 0.0 else f"{mode}_sig{signal}_{noise}"
    lasso_results = lasso_dict[key]
    lasso_reject = np.mean([r.p_value < 0.05 for r in lasso_results])

    if signal == 0.0:
        return prop_reject, oc_reject, lasso_reject, naive_reject
    return prop_reject, oc_reject, lasso_reject


def null_plot(mode, noise, save=True):
    naive_rejects = []
    prop_rejects = []
    oc_rejects = []
    lasso_rejects = []

    for d in (labels := [40, 60, 80, 100]):
        prop, oc, lasso, naive = get_reject_rates(mode, 0.0, d, noise)
        prop_rejects.append(prop)
        oc_rejects.append(oc)
        lasso_rejects.append(lasso)
        naive_rejects.append(naive)

    plt.plot(labels, prop_rejects, label="proposed", marker="x")
    plt.plot(labels, oc_rejects, label="oc", marker="x")
    plt.plot(labels, lasso_rejects, label="lasso", marker="x")
    plt.plot(labels, naive_rejects, label="naive", marker="x")
    plt.plot(labels, 0.05 * np.ones(len(labels)), linestyle="--", color="red", lw=0.5)
    plt.xticks(labels)
    plt.ylim(0, 0.5)

    plt.xlabel("Data Length")
    plt.ylabel("Type I Error Rate")
    # mode_literal = "linear trend" if mode == "lt" else "mean shift"
    # plt.title(f'{mode_literal} with {noise} noise under null')
    plt.legend(frameon=False, loc="upper left")
    if save:
        plt.savefig(
            f"images/{mode}_{noise}_null.pdf",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()


def alter_plot(mode, noise, save=True):
    prop_rejects = []
    oc_rejects = []
    lasso_rejects = []

    labels = [0.1, 0.2, 0.3, 0.4] if mode == "lt" else [1.0, 2.0, 3.0, 4.0]
    for signal in labels:
        try:
            prop, oc, lasso = get_reject_rates(mode, signal, 60, noise)
        except:
            prop, oc, lasso = None, None, None
        prop_rejects.append(prop)
        oc_rejects.append(oc)
        lasso_rejects.append(lasso)

    plt.plot(labels, prop_rejects, label="proposed", marker="x")
    plt.plot(labels, oc_rejects, label="oc", marker="x")
    plt.plot(labels, lasso_rejects, label="lasso", marker="x")
    # plt.plot(labels, 0.05 * np.ones(len(labels)), linestyle="--", color="red", lw=0.5)
    plt.xticks(labels)
    plt.ylim(0, 1.0)

    plt.xlabel("signal")
    plt.ylabel("Power")
    # mode_literal = "linear trend" if mode == "lt" else "mean shift"
    # plt.title(f'{mode_literal} with {noise} noise under alternative')
    if mode == "lt":
        plt.legend(frameon=False, loc="upper left")
    else:
        plt.legend(frameon=False, loc="center right")

    if save:
        plt.savefig(
            f"images/{mode}_{noise}_alter.pdf",
            transparent=True,
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=1000)
    args = parser.parse_args()

    keys = itertools.product(["lt", "ms"], ["iid", "ar"])

    if 0 <= args.num < 4:
        key = list(keys)[args.num]
        null_plot(*key, save=True)

    if 4 <= args.num < 8:
        key = list(keys)[args.num - 4]
        alter_plot(*key, save=True)
