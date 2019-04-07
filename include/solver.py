from queue import Queue
import logging.config, logging

logging.config.fileConfig(fname='include/logging.conf', disable_existing_loggers=False)
logger = logging.getLogger('logger.solver')

class Solver():

    def __init__(self, problem):
        self.problem = problem
        self.queue = Queue()
        self.queue.put(self.problem.get_indef_region())
        
        MIN_CONST = -1
        self.record_point = None
        self.record_value = MIN_CONST


    def run(self):
        while not self.queue.empty():
            region = self.queue.get()
            logger.debug('Get region: {}'.format(region))
            feasible_point, estimate_point, idx = self.problem.dantzing_rule(region)
            estimate_value = self.problem.value(estimate_point)
            feasible_value = self.problem.value(feasible_point)
            
            logger.debug('Feasible point : {}, feasible value {}'.format(feasible_point, feasible_value))
            logger.debug('Estimate point : {}, estimate value {}'.format(estimate_point, estimate_value))

            if estimate_value <= self.record_value:
                logger.debug('Estimate value less or equal record value')
                continue

            if feasible_value > self.record_value:
                self.record_point = feasible_point
                self.record_value = feasible_value 
                logger.debug('New record point')
            
            if idx == -1:
                continue
            region_1, region_2 = self.problem.split(idx, region)
            if self.problem.delimitations_A[region_1 == 1].sum() <= self.problem.delimitations_b and\
               (region_1 < 0).any():
                logger.debug('Put region {}'.format(region_1))
                self.queue.put(region_1)
            if self.problem.delimitations_A[region_2 == 1].sum() <= self.problem.delimitations_b and\
               (region_2 < 0).any():
                logger.debug('Put region {}'.format(region_2))
                self.queue.put(region_2)

        return self.record_point
