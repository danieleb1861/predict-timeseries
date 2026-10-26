"""Exploratory data analysis plotting workflow."""

import numpy as np
import matplotlib.pyplot as plt


def run_eda_analysis(TT, IMGS_DIR, selector, numGens):
    # %% -- Dataset: bins and distributions
    plt.figure(figsize=(7, 6))
    if np.isnan(numGens):
        plt.subplot(2, 1, 1)
        plt.hist(TT["act_pow_pu"], bins=100, density=False, alpha=0.5, label="original")
        plt.grid(True)
        plt.xlabel("Active power (PU)")
        plt.ylabel("NumSamples")
        plt.yticks()

        plt.subplot(2, 1, 2)
        plt.hist(TT["react_pow_pu"], bins=100, density=False, alpha=0.5, label="original")
        plt.grid(True)
        plt.xlabel("Reactive power (PU)")
        plt.ylabel("NumSamples")
