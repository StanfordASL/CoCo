from abc import ABC, abstractmethod

class Problem(ABC):
    """Abstract class to handle all problem instances"""
    
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def init_problem(self): 
        """Abstract method to implement problem in cvxpy."""
        raise NotImplementedError 
        
    @abstractmethod
    def solve_micp(self,params):
        """Abstract method to solve mixed-integer problem.
        
        Args:
            params: (list of) numpy array(s) of parameter values for specific 
                problem instance.
            
        Returns:
            solve_time: time required to solve problem by cvxpy.
            cost: objective value of optimal solution.
            x_star: (list) of optimal continuous variable values.
            y_star: optimal integer assignments.
        """
        raise NotImplementedError
        
    @abstractmethod
    def solve_pinned(self,params,strat):
        """Abstract method to solve pinned MICP.
        
        Args:
            params: (list of) numpy array(s) of parameter values for specific
                problem instance.
            strat: numpy array of ints, assignment for binary variables.
            
        Returns:
            feasible: boolean, if problem was feasible with this strategy.
            solve_time: time required to solve problem by cvxpy.
            cost: objective value of optimal solution.
            x_star: (list) of optimal continuous variable values.
            y_star: optimal integer assignments.
        """
        raise NotImplementedError
        
class Solver(ABC):
    """Abstract class to handle all solver instances."""
    
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, params):
        """Abstract method mapping parameter values to predicted strategy.
        
        Args:
            params: (list of) numpy array(s) of parameter values for specific
                problem instance.
            
        Returns:
            y_hat: numpy array of ints, estimate of optimal strategy, of shape
                [self.n_evals, self.n_y].  
        """
        raise NotImplementedError