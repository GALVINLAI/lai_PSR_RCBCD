from qiskit.utils import algorithm_globals
algorithm_globals.random_seed = 3142
from typing import Iterator, Optional, Union, Callable, Dict, Any

from hpman.m import _
import hpargparse
import argparse

from sysflow.utils.common_utils.file_utils import make_dir, load, dump
import numpy as np
import random

np.random.seed(algorithm_globals.random_seed)

from qiskit_machine_learning.datasets import ad_hoc_data
TRAIN_DATA, TRAIN_LABELS, TEST_DATA, TEST_LABELS = (
    ad_hoc_data(training_size=20,
                test_size=5,
                n=2,
                gap=0.3,
                one_hot=False)
)

parser = argparse.ArgumentParser()
_.parse_file(__file__)
hpargparse.bind(parser, _)
parser.parse_args()

from qiskit.circuit.library import ZZFeatureMap, TwoLocal
FEATURE_MAP = ZZFeatureMap(feature_dimension=2, reps=2)
VAR_FORM = TwoLocal(2, ['ry', 'rz'], 'cz', reps=2)

AD_HOC_CIRCUIT = FEATURE_MAP.compose(VAR_FORM)
AD_HOC_CIRCUIT.measure_all()
AD_HOC_CIRCUIT.decompose().draw()

def circuit_instance(data, variational):
    """Assigns parameter values to `AD_HOC_CIRCUIT`.
    Args:
        data (list): Data values for the feature map
        variational (list): Parameter values for `VAR_FORM`
    Returns:
        QuantumCircuit: `AD_HOC_CIRCUIT` with parameters assigned
    """
    parameters = {}
    for i, p in enumerate(FEATURE_MAP.ordered_parameters):
        parameters[p] = data[i]
    for i, p in enumerate(VAR_FORM.ordered_parameters):
        parameters[p] = variational[i]
    return AD_HOC_CIRCUIT.assign_parameters(parameters)

def parity(bitstring):
    """Returns 1 if parity of `bitstring` is even, otherwise 0."""
    hamming_weight = sum(int(k) for k in list(bitstring))
    return (hamming_weight+1) % 2

def label_probability(results):
    """Converts a dict of bitstrings and their counts,
    to parities and their counts"""
    shots = sum(results.values())
    probabilities = {0: 0, 1: 0}
    for bitstring, counts in results.items():
        label = parity(bitstring)
        probabilities[label] += counts / shots
    return probabilities

from qiskit import BasicAer, execute

def classification_probability(data, variational):
    """Classify data points using given parameters.
    Args:
        data (list): Set of data points to classify
        variational (list): Parameters for `VAR_FORM`
    Returns:
        list[dict]: Probability of circuit classifying
                    each data point as 0 or 1.
    """
    circuits = [circuit_instance(d, variational) for d in data]
    backend = BasicAer.get_backend('qasm_simulator')
    results = execute(circuits, backend).result()
    classification = [
        label_probability(results.get_counts(c)) for c in circuits]
    return classification

def cross_entropy_loss(classification, expected):
    """Calculate accuracy of predictions using cross entropy loss.
    Args:
        classification (dict): Dict where keys are possible classes,
                               and values are the probability our
                               circuit chooses that class.
        expected (int): Correct classification of the data point.

    Returns:
        float: Cross entropy loss
    """
    p = classification.get(expected)  # Prob. of correct classification
    return -np.log(p + 1e-10)

def cost_function(data, labels, variational):
    """Evaluates performance of our circuit with `variational`
    parameters on `data`.

    Args:
        data (list): List of data points to classify
        labels (list): List of correct labels for each data point
        variational (list): Parameters to use in circuit

    Returns:
        float: Cost (metric of performance)
    """
    classifications = classification_probability(data, variational)
    cost = 0
    for i, classification in enumerate(classifications):
        cost += cross_entropy_loss(classification, labels[i])
    cost /= len(data)
    return cost

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

# Set up the optimization
from qiskit.algorithms.optimizers import GradientDescent
from qiskit.algorithms.optimizers.steppable_optimizer import AskData, TellData, OptimizerState, SteppableOptimizer

log = OptimizerLog()


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


learning_rate = _('lr', 0.1)
optimizer_type = _('optim','gd')

print(learning_rate, optimizer_type)

if optimizer_type == 'gd':
    optimizer = GradientDescent(maxiter=2000, learning_rate=learning_rate, callback=log.update)
else: 
    optimizer = RandomCoordinateDescent(maxiter=2000, learning_rate=learning_rate, callback=log.update)


#initial_point = np.random.random(VAR_FORM.num_parameters)
initial_point = np.array([3.28559355, 5.48514978, 5.13099949,
                          0.88372228, 4.08885928, 2.45568528,
                          4.92364593, 5.59032015, 3.66837805,
                          4.84632313, 3.60713748, 2.43546])

def objective_function(variational):
    """Cost function of circuit parameters on training data.
    The optimizer will attempt to minimize this."""
    return cost_function(TRAIN_DATA, TRAIN_LABELS, variational)

# Run the optimization
result = optimizer.minimize(objective_function, initial_point)

opt_var = result.x
opt_value = result.fun

make_dir(f'results/quantum_classifier/lr_{learning_rate}')
data_dict = {'result': result, 'log': log}
dump(data_dict, f'results/quantum_classifier/lr_{learning_rate}/{optimizer_type}.pkl')

