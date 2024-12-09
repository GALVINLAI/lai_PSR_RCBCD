from qiskit.algorithms.optimizers import GradientDescent
import numpy as np
from qiskit.algorithms.optimizers.steppable_optimizer import AskData, TellData, OptimizerState, SteppableOptimizer

class OptimizerLog:
    """Log to store optimizer's intermediate results"""
    def __init__(self):
        self.evaluations = []
        self.parameters = []
        self.costs = []
    def update(self, evaluation, parameter, cost, _stepsize):
        """Save intermediate results. Optimizer passes five values
        but we ignore the last two."""
        self.evaluations.append(evaluation)
        self.parameters.append(parameter)
        self.costs.append(cost)

log = OptimizerLog()

def f(x):
    return (np.linalg.norm(x) - 1) ** 2

initial_point = np.array([1, 0.5, -0.2])

optimizer = GradientDescent(maxiter=100, learning_rate=0.1, callback=log.update)

result = optimizer.minimize(fun=f, x0=initial_point)

print(f"Found minimum {result.x} at a value"
    f"of {result.fun} using {result.nfev} evaluations.")

print(len(log.costs))


import numpy as np
import random

class RandomCoordinateDescent(GradientDescent):
    """The Random Coordinate Descent minimization routine.

    This optimizer inherits from the GradientDescent optimizer.
    The main difference is that, in each iteration, it randomly selects a coordinate
    and applies the gradient descent update rule to that coordinate only.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._selected_coordinate = None

    def ask(self) -> AskData:
        """Randomly selects a coordinate.

        Returns an object with the data needed to evaluate the gradient at the selected coordinate.
        """
        # Randomly select a coordinate
        self._selected_coordinate = random.choice(range(len(self.state.x)))

        return AskData(
            x_jac=self.state.x,
        )

    def tell(self, ask_data: AskData, tell_data: TellData) -> None:
        """
        Updates the selected coordinate by an amount proportional to the learning rate 
        and the value of the gradient at that point.

        Args:
            ask_data: The data used to evaluate the function.
            tell_data: The data from the function evaluation.

        Raises:
            ValueError: If the gradient passed doesn't have the right dimension.
        """
        if np.shape(self.state.x) != np.shape(tell_data.eval_jac):
            raise ValueError("The gradient does not have the correct dimension")

        # Update the selected coordinate according to the gradient descent update rule
        self.state.x[self._selected_coordinate] -= next(self.state.learning_rate) * tell_data.eval_jac[self._selected_coordinate]

        self.state.stepsize = np.abs(tell_data.eval_jac[self._selected_coordinate])
        self.state.nit += 1


optimizer = RandomCoordinateDescent(maxiter=150, learning_rate=0.1, callback=log.update)

result = optimizer.minimize(fun=f, x0=initial_point)

print(f"Found minimum {result.x} at a value"
    f"of {result.fun} using {result.nfev} evaluations.")

print(len(log.costs))