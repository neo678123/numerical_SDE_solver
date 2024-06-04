import numpy as np


class FiniteDifferenceSolver:
    def __init__(self, dim, meshSize):
        self.dim = dim
        self.meshSize = meshSize
        self.invMeshSize = 1/meshSize

    def getForwardDifferences(self, f, xValues, direction):
        """
        Calculates array of forward differences
        (Define h := meshSize * direction)
            (f(x+h) - f(x))/meshSize
        """
        h = self.meshSize * direction
        g = lambda x: f(x + h) - f(x)

        return self.invMeshSize * np.map(g, xValues)
