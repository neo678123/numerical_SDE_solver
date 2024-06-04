import numpy as np
import math
from src.FiniteDifferenceSolver import FiniteDifferenceSolver


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

    # a(t: float, X: [float]) -> [float]
    # b(t: float, X: [float]) -> [[float],[float]]
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
        :param Xt0: [float]
        :return: The solution X_t as [[float]]
        """
        dt = np.diff(self.times)
        dW = np.transpose(np.multiply(np.vectorize(math.sqrt)(dt),
                          np.reshape(np.random.normal(0, 1, self.dim * len(dt)), [self.dim, len(dt)])))

        solution = np.empty([len(self.times), self.dim])

        solution[0] = Xt0
        for i in range(len(self.times) - 1):
            solution[i+1] = solution[i] + self.a(self.times[i+1], solution[i]) * dt[i] \
                            + np.matmul(self.b(self.times[i+1], solution[i]), dW[i])

        return np.transpose(solution)

    def getPDF(self, minValues: [float], maxValues: [float], meshSize: float, maxTime: float, timeSize: float):
        """
        Solves Fokker-Planck equation for X with X_i in [minValues_i, maxValues_i]
        for 0 <= t <= maxTime with a mesh size meshSize

        :param minValues
        :param maxValues
        :param meshSize
        :param maxTime
        :param timeSize
        """

        # make a mesh
        mesh = np.empty([self.dim, 1])
        for i in range(self.dim):          # this looks dumb
            mesh[i][0] = x = minValues[i]  # TODO: make numpy haiku or use njit(?)
            while x < maxValues[i]:
                x += meshSize
                mesh[i].append(x)

        D = lambda t, X: 0.5 * np.matmul(self.b(t, X), np.transpose(self.b(t, X)))
        solver = FiniteDifferenceSolver(self.dim, meshSize)

        p = []  # TODO: initial conditions

        for t in range(0, maxTime, timeSize):  # temporary: range doesn't support floats
            pass  # TODO: solve the equation
