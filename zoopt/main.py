import numpy as np
import matplotlib.pyplot as plt
from zoopt import Dimension, Objective, Parameter, Opt


def ackley(solution):
    x = solution.get_x()
    bias = 0.2
    value = -20 * np.exp(-0.2 * np.sqrt(sum([(i - bias) * (i - bias) for i in x]) / len(x))) - \
            np.exp(sum([np.cos(2.0*np.pi*(i-bias)) for i in x]) / len(x)) + 20.0 + np.e
    return value


dim = 100  # dimension
obj = Objective(ackley, Dimension(dim, [[-1, 1]] * dim, [True] * dim))
# perform optimization
solution = Opt.min(obj, Parameter(budget=100 * dim))
# print result
solution.print_solution()


plt.plot(obj.get_history_bestsofar())
plt.savefig('figure.png')