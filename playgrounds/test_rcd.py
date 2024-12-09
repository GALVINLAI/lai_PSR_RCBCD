# test the random coordinate descent algorithm

import random
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def random_coordinate_descent(f, grad_f, initial_point, alpha, num_iterations):
    """
    Random Coordinate Descent algorithm for optimizing a function f with gradient grad_f.

    Parameters:
        - f: function to optimize
        - grad_f: gradient function of f
        - initial_point: starting point for the algorithm
        - alpha: step size for the algorithm
        - num_iterations: number of iterations to run the algorithm for

    Returns:
        - best_point: the point that achieves the minimum value of f found during the algorithm
        - best_value: the minimum value of f found during the algorithm
    """
    current_point = initial_point
    best_point = current_point
    best_value = f(current_point)

    coord_list = []
    value_list = []

    for i in range(num_iterations):
        j = random.randint(0, len(current_point)-1)  # choose a random coordinate
        coord_list.append(j)
        # move in the negative direction of the chosen coordinate
        next_point = list(current_point)
        next_point[j] -= alpha * grad_f[j](current_point)

        # update the best point and value if the new point is better
        next_value = f(next_point)
        value_list.append(next_value)
        
        if next_value < best_value:
            best_point = next_point
            best_value = next_value

        current_point = next_point
    # print(coord_list)
    # print(value_list)
    
    # make a plot for value list 
    plt.plot(value_list)
    plt.xlabel('Iteration')
    plt.ylabel('Value')
    plt.xlim(0, 500)
    plt.title('Value vs Iteration')
    plt.savefig('value_vs_iteration.png')
    
    return best_point, best_value


if __name__ == "__main__":
    # test the random coordinate descent algorithm on a simple function
    def f(x):
        return x[0]**2 + x[1]**2

    def grad_f0(x):
        return 2 * x[0]

    def grad_f1(x):
        return 2 * x[1]

    grad_f = [grad_f0, grad_f1]

    initial_point = [random.random(), random.random()]
    print("Initial point:", initial_point)
    alpha = 0.01
    num_iterations = 10000

    best_point, best_value = random_coordinate_descent(f, grad_f, initial_point, alpha, num_iterations)
    print("Best point found:", best_point)
    print("Best value found:", best_value)

    # test the random coordinate descent algorithm on a more complex function
    
    