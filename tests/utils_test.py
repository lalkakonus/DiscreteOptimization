import os, sys
directory = os.path.join(os.path.dirname(__file__), '../include')
sys.path.insert(0, directory)

import utils
import numpy as np

low_bound = 0
high_bound = 100
N_items = 20

data = np.random.randint(low_bound, high_bound, N_items)
permutation = np.random.permutation(N_items)
shuffled_data = data[permutation]

inverse_permutation = utils.inverse_permutation(permutation)
predicate = (data == shuffled_data[inverse_permutation]).all()

if predicate:
    print('Test success')
else:
    print('Test failed')
    
