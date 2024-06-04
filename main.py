from matplotlib import pyplot as plt
import numpy as np
import math

from src.ConfidenceIntervalsMaker import ConfidenceIntervalMaker
from src.IterativeEulerMaruyama import IterativeEulerMaruyamaMultivariate
from src.IterativeEulerMaruyama import IterativeEulerMaruyama

if __name__ == '__main__':
    mu = 0.75
    sigma = 0.30
    dt = 0.01
    times = [i * dt for i in range(math.floor(3/dt))]
    X0 = 300

    # Solve n independent Langevin equations
    n = 20
    langevinModel = IterativeEulerMaruyamaMultivariate(
        lambda t, X: -mu * X,
        lambda t, X: 50 * sigma * np.identity(n),
        times, n
    )

    langevinSol = langevinModel.getSolution(n * [X0])

    fig, ax = plt.subplots(1)
    fig.suptitle("Langevin equation")

    for sol in langevinSol:
        ax.plot(times, sol)
    plt.show()

    # Solve m independent Geometric Brownian Motions
    m = 50
    gbmModel = IterativeEulerMaruyamaMultivariate(
        lambda t, X: mu * X,
        lambda t, X: sigma * np.diag(X),
        times, m
    )

    gbmModel1d = IterativeEulerMaruyama(
        lambda t, X: mu * X,
        lambda t, X: sigma * X,
        times
    )
    pdf = gbmModel1d.getPDF(-10, 10, 0.1, 10, 0.1, 1)
    for p in pdf:
        plt.plot([0.1 * i for i in range(-100, 98)], p)
    plt.show()

    gbmSol = gbmModel.getSolution(m * [X0])

    fig, axs = plt.subplots(2)
    fig.suptitle("Geometric Brownian motion + 95% CI")

    for sol in gbmSol:
        axs[0].plot(times, sol)

    mean = ConfidenceIntervalMaker.getMeanSeries(gbmSol)
    ci = ConfidenceIntervalMaker.getConfidenceInterval(gbmSol, 0.95)

    plt.plot(times, mean)
    for i in ci:
        axs[1].plot(times,i)
    plt.show()