import os, sys
directory = os.path.join(os.path.dirname(__file__), 'include')
sys.path.insert(0, directory)

import numpy as np
from task import LinearProblem
from solver import Solver

def main():
    try:
        # Problem description 
        f = np.array([5, 3, 4, 4])
        A = np.array([2, 2, 1, 2])
        b = 4

        problem = LinearProblem(f, A, b)
        print('\nLinear problem :\n', problem)

        solution = Solver(problem).run()
        print('Solution: (' + ', '.join(map(lambda x: str(x), solution)) + ')')
        print('Optimal value: ', problem.value(solution))
    except Exception as exception:
        print('Error occured, program stopped: {}'.format(exception))

if __name__ == '__main__':
    main()
