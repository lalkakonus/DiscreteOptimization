import os, sys
directory = os.path.join(os.path.dirname(__file__), '../include')
sys.path.insert(0, directory)

from task import LinearProblem
from solver import Solver
import numpy as np
import random

N = 4
f = np.random.randint(0, 50, N)
A = np.random.randint(0, 50, N)
b = random.randint(0, A.sum())

problem = LinearProblem(f, A, b)
print(problem)

solution = Solver(problem).run()
print('Solution: (' + ', '.join(map(lambda x: str(x), solution)) + ')')
print('Optimal value: ', problem.value(solution))
