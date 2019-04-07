import numpy as np
from utils import inverse_permutation
import logging.config, logging

logging.config.fileConfig(fname='include/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('logger.problem')

class LinearProblem:

    def __init__(self, optimization_function: np.array,
                 delimitations_A: np.array,
                 delimitations_b: np.array):
        
        logger.debug('f : {}'.format(optimization_function))
        logger.debug('A : {}'.format(delimitations_A))
        logger.debug('b : {}'.format(delimitations_b))

        if not isinstance(optimization_function, np.ndarray) or\
           not isinstance(delimitations_A, np.ndarray) or\
           not isinstance(delimitations_b, int):
            raise Exception('Type error')

        if not len(optimization_function.shape) == 1 or\
           not len(delimitations_A.shape) == 1 or\
           not optimization_function.shape[0] == delimitations_A.shape[0]:
            raise Exception('Shape error')

        if not (optimization_function > 0).all() or\
           not (delimitations_A >= 0).all(): 
            raise ValueError('Value error')

        self._optimization_function = optimization_function
        self._delimitations_A = delimitations_A
        self._delimitations_b = delimitations_b
        self._size = optimization_function.size

    # Get indefinite region which correspond current problem shape
    def get_indef_region(self):
        return -np.ones(self._size)

    # Get function value in point (x)
    def value(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError('Input argument has wrong type : {}'.format(type(x)))
        if not x.shape == self.optimization_function.shape:
            raise ValueError('Input argument has wrong shape')

        return np.dot(self.optimization_function, x)

    # Dichotomy spliting region. Result have 1 and 0 coordinate at idx position
    def split(self, position, region):
        if not isinstance(position, int):
            raise TypeError('Input argument must be int, not {}'.format(type(position)))
        if not (position >= 0 and position < self._optimization_function.size):
            raise ValueError('Input position must be in [0, {})'.format(self._optimization_function.size))

        point_1 = np.copy(region)
        point_1[position] = 1
        
        point_2 = np.copy(region)
        point_2[position] = 0

        return point_1.astype(np.int), point_2.astype(np.int)

    @property
    def optimization_function(self):
        return np.copy(self._optimization_function)
    
    @property
    def delimitations_A(self):
        return np.copy(self._delimitations_A)
    
    @property
    def delimitations_b(self):
        return self._delimitations_b

    # Dantzing rule described in pdf attachement
    def dantzing_rule(self, region):

        mask_indef = np.argwhere(region < 0).reshape(-1)
        mask_true = np.argwhere(region == 1).reshape(-1)
   
        indef_A = self.delimitations_A[mask_indef].astype(np.float32)
        indef_A[indef_A == 0] = 1e-5
        lambda_array = self.optimization_function[mask_indef] / indef_A
    
        order = np.argsort(-lambda_array)
        inverse_order = inverse_permutation(order)
        
        delimitations_A = self.delimitations_A[mask_indef][order]
        
        if mask_true.size == 0:
            correction = 0
        else:
            correction = np.dot(region[mask_true], self.delimitations_A[mask_true])

        logger.debug('A com_sum: {}'.format(np.cumsum(delimitations_A)))
        logger.debug('b : {}'.format(self.delimitations_b - correction))

        cum_sum, s_position = 0, -1
        for i, a in enumerate(delimitations_A):
            if cum_sum <= self.delimitations_b - correction and\
               self.delimitations_b - correction < cum_sum + a:
                s_position = i
                break
            else:
                cum_sum += a
        logger.debug('S position: {}'.format(s_position))
        

        x = np.ones(mask_indef.size)
        if s_position != -1:
            x[s_position + 1:] = 0
            x[s_position] = (self.delimitations_b - correction - delimitations_A[:s_position].sum()) / delimitations_A[s_position]
        
        estimate_point = np.copy(region).astype(np.float32)
        estimate_point[mask_indef] = x[inverse_order]
       
        if s_position != -1:
            if x[s_position] != round(x[s_position]):
                x[s_position] = 0
        feasible_point= np.copy(region)
        feasible_point[mask_indef] = x[inverse_order]
        
        if s_position == -1:
            s_position = mask_indef.size - 1
        idx = int(mask_indef[np.argwhere(inverse_order == s_position)])

        return feasible_point.astype(np.int), estimate_point, idx
    
    def __str__(self):
        string = ''
        for idx, coeff in enumerate(self.optimization_function, 1):
            string += str(coeff) + '*x' + str(idx) + ' + '
        string = string[:-2] + '-> max'
       
        string += '\n'
        tmp = ''
        for idx, coeff in enumerate(self.delimitations_A, 1):
            tmp += str(coeff) + '*x' + str(idx) + ' + '
        string += tmp[:-2] + '<= ' + str(self.delimitations_b)

        string += '\n'
        for i in range(1, self.optimization_function.size + 1):
            string += 'x' + str(i) + ' in {0, 1}\n'
        return string

