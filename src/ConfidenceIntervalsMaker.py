import numpy as np


class ConfidenceIntervalMaker:
    @staticmethod
    def getMeanSeries(solutions: [[float]]) -> [float]:
        return np.mean(solutions, axis=0)

    @staticmethod
    def getVarSeries(solutions: [[float]]) -> [float]:
        return np.var(solutions, axis=0)

    @staticmethod
    def getStdSeries(solutions: [[float]]) -> [float]:
        return np.std(solutions, axis=0)

    @staticmethod
    def getConfidenceInterval(solutions: [[float]], percent: float) -> [[float]]:
        mean = ConfidenceIntervalMaker.getMeanSeries(solutions)
        std = ConfidenceIntervalMaker.getStdSeries(solutions)

        return np.stack((np.subtract(mean,percent * std), np.add(mean, percent * std)))

