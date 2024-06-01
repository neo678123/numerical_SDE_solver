from matplotlib import pyplot as plt
import numpy as np
import math

from src.ConfidenceIntervalsMaker import ConfidenceIntervalMaker
from src.IterativeEulerMaruyama import IterativeEulerMaruyamaMultivariate

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

    for sol in langevinSol:
        plt.plot(times, sol)
    plt.show()

    # Solve m independent Geometric Brownian Motions
    m = 50
    gbmModel = IterativeEulerMaruyamaMultivariate(
        lambda t, X: mu * X,
        lambda t, X: sigma * np.diag(X),
        times, m
    )

    gbmSol = gbmModel.getSolution(m * [X0])

    for sol in gbmSol:
        plt.plot(times, sol)
    plt.show()

    mean = ConfidenceIntervalMaker.getMeanSeries(gbmSol)
    ci = ConfidenceIntervalMaker.getConfidenceInterval(gbmSol, 0.95)

    plt.plot(times, mean)
    for i in ci:
        plt.plot(times,i)
    plt.show()