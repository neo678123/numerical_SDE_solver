import numpy as np
import math
import matplotlib.pyplot as plt


class IterativeEulerMaruyama:
    """
    Framework for solving SDE of the form
        dX = a(t,X)dt + b(t,X)dW
    using the Euler-Maruyama method
    """

    # a(t: float, X: float) -> float
    # b(t: float, X: float) -> float
    # times: [float]

    def __init__(self, a, b, times):
        """
        :param a: a(t,X)
        :param b: b(t,X)
        :param times: array of time values
        """
        self.a = a
        self.b = b
        self.times = times

    def getSolution(self, Xt0):
        """
        Solve the SDE using an iterative Euler-Maruyama scheme
        :param Xt0:
        :return: the tuple "(t, X_t)"
        """
        solution = [Xt0]
        dt = np.diff(self.times)
        dW = np.multiply(np.vectorize(math.sqrt)(dt), np.random.normal(0, 1, len(dt)))

        for i in range(len(self.times) - 1):
            solution.append(solution[i] + self.a(self.times[i + 1], solution[i]) * dt[i]
                            + self.b(self.times[i + 1], solution[i]) * dW[i])

        return solution


class IterativeEulerMaruyamaMultivariate:
    """
    Framework for solving SDE of the form
        dX = a(t,X)dt + b(t,X)dW
    using the Euler-Maruyama method
    """

    # a(t: float, X: float) -> float
    # b(t: float, X: float) -> [[float],[float]]
    # times: [float]

    def __init__(self, a, b, times, dim):
        """
        :param a: a(t,X)
        :param b: b(t,X)
        :param times: array of time values
        """
        self.a = a
        self.b = b
        self.times = times
        self.dim = dim

    def getSolution(self, Xt0):
        """
        Solve the SDE using an iterative Euler-Maruyama scheme
        :param Xt0:
        :return: the tuple "(t, X_t)"
        """
        dt = np.diff(self.times)
        solution = np.empty([self.dim, len(times)])
        dW = np.empty([self.dim, len(times)-1])

        # TODO: only works for dim=1, need to do matrix stuff
        for i in range(self.dim):
            dW[i] = np.multiply(np.vectorize(math.sqrt)(dt), np.random.normal(0, 1, len(dt)))

            solution[i][0] = Xt0[i]
            for j in range(len(self.times) - 1):
                solution[i][j+1] = solution[i][j] + self.a(self.times[j + 1], solution[i][j]) * dt[j] \
                                    + self.b(self.times[j + 1], solution[i][j]) * dW[i][j]

        return solution


if __name__ == '__main__':
    mu = 10
    sigma = 1
    dt = 0.01
    times = [i * dt for i in range(301)]
    X0 = 0

    # Attempt at solving a path for the Langevin equation
    langevin = IterativeEulerMaruyamaMultivariate(
        lambda t, X: -mu * X,
        lambda t, X: sigma,
        times, 1
    )

    solution = langevin.getSolution([X0])

    plt.plot(times, *solution)
    plt.show()
