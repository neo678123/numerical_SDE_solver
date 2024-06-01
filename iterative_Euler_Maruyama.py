import numpy as np
import math
import matplotlib.pyplot as plt

'''
Dumb solution, probably better vectorized way using numpy
'''

class iterative_Euler_Maruyama:
    '''
    Framework for solving SDE of the form
        dX = a(t,X)dt + b(t,X)dW
    using the Euler-Maruyama method
    '''

    def a(t: float, X: float) -> float: pass
    def b(t: float, X: float) -> float: pass
    t_max: float
    dt: float

    def __init__(self, a, b, t_max, dt):
        self.a = a
        self.b = b
        self.t_max = t_max
        self.dt = dt

    def nextPoint(self, prevPoint, t):
        '''
        Does 1 iteration of the Euler-Maruyama scheme
        :return: next data point "X_i"
        '''

        dW = np.random.normal(0, 1, 1) * math.sqrt(self.dt)
        return prevPoint + self.a(t, prevPoint) * self.dt + self.b(t, prevPoint) * dW

    def iterateSolution(self, X0):
        '''
        Solve the SDE using an iterative Euler-Maruyama scheme
        :param X0:
        :return: the tuple "(t, X_t)"
        '''
        solution = [X0]
        t = 0
        n = 0
        while t < self.t_max:
            solution.append(*self.nextPoint(solution[-1], t))
            t += self.dt
            n += 1

        t_axis = [i * self.dt for i in range(n + 1)]
        return t_axis, solution


if __name__ == '__main__':
    mu = 10
    sigma = 1
    dt = 0.01
    t_max = 3
    X0 = 0

    # Attempt at solving a path for the Langevin equation
    langevin = iterative_Euler_Maruyama(
        lambda t, X: -mu * X,
        lambda t, X: sigma,
        t_max, dt
    )

    t_axis, solution = langevin.iterateSolution(X0)

    print(t_axis)
    print(solution)

    plt.plot(t_axis, solution)
    plt.show()
