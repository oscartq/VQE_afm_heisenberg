import csv 
import multiprocessing as mp
import numpy as np
from scipy.optimize import minimize
Pi = np.pi

def optimize_by_lbfgsb(function, initial_gamma, initial_beta, initial_phi=None, initial_theta=None, bounds=None, parameters=2, print_results=True, filepath=""):
    """
    Optimize a given function using the L-BFGS-B algorithm.

    Parameters:
    function (callable): The function to be optimized.
    initial_gamma (array-like): Initial values for gamma parameters.
    initial_beta (array-like): Initial values for beta parameters.
    initial_phi (array-like, optional): Initial values for phi parameters.
    initial_theta (array-like, optional): Initial values for theta parameters.
    bounds (list of tuple, optional): Bounds for the parameters.
    parameters (int): Number of parameter sets (2, 3, or 4).
    figure (bool): Whether to print the optimization process.
    filepath (str): Path to the CSV file for logging.

    Returns:
    tuple: Optimized parameter values.
    """
    if parameters == 2:
        gamma, beta = initial_gamma.copy(), initial_beta.copy()
        initial_params = np.concatenate([initial_gamma, initial_beta])
        split_count = 2
    elif parameters == 3:
        gamma, beta, phi = initial_gamma.copy(), initial_beta.copy(), initial_phi.copy()
        initial_params = np.concatenate([initial_gamma, initial_beta, initial_phi])
        split_count = 3
    elif parameters == 4:
        gamma, beta, phi, theta = initial_gamma.copy(), initial_beta.copy(), initial_phi.copy(), initial_theta.copy()
        initial_params = np.concatenate([initial_gamma, initial_beta, initial_phi, initial_theta])
        split_count = 4    
    else:
        raise ValueError("Unsupported number of parameters. Only 2, 3 or 4 parameters are supported.")
    
    def energy_function(params):
        if parameters == 2:
            gamma, beta = np.split(params, split_count)
            energy = function(gamma=gamma, beta=beta)
        elif parameters == 3:
            gamma, beta, phi = np.split(params, split_count)
            energy = function(gamma=gamma, beta=beta, phi=phi)
        elif parameters == 4:
            gamma, beta, phi, theta = np.split(params, split_count)
            energy = function(gamma=gamma, beta=beta, phi=phi, theta=theta)    
        return energy
    
    history_params = []
    history_energy = []

    def callback(params):
        history_params.append(params)
        history_energy.append(energy_function(params))
        if parameters == 2:
            gamma, beta = np.split(params, split_count)
            record = [len(history_energy), energy_function(params)] + list(gamma) + list(beta)
        elif parameters == 3:
            gamma, beta, phi = np.split(params, split_count)
            record = [len(history_energy), energy_function(params)] + list(gamma) + list(beta) + list(phi)
        elif parameters == 4:
            gamma, beta, phi, theta = np.split(params, split_count)
            record = [len(history_energy), energy_function(params)] + list(gamma) + list(beta) + list(phi) + list(theta)
        
        # Open the file in append mode and write the record
        with open(filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(record)
            if print_results:
                print(record)
            f.flush()
            
    # Write the header to the CSV file
    with open(filepath, mode='w', newline='') as f:
        writer = csv.writer(f)
        headline = ["iter", "energy"]
        for p in range(int(len(initial_gamma))):
            headline.append(f"gamma[{p}]")
            headline.append(f"beta[{p}]")
            if parameters >= 3:
                headline.append(f"phi[{p}]")
            if parameters == 4: 
                headline.append(f"theta[{p}]")   
        writer.writerow(headline)
            
    # Perform the optimization
    if bounds is None:
        bounds = [(0, None)] * len(initial_params)

    result = minimize(
        fun=energy_function,
        x0=initial_params,
        jac="3-point",
        method='L-BFGS-B',
        options={'gtol': 1e-8},
        bounds=bounds,
        tol=1e-10,
        callback=callback
    )
    
    if print_results:
        print(result)
    
    if parameters == 2:
        gamma, beta = np.split(result.x, split_count)
        return gamma, beta
    elif parameters == 3:
        gamma, beta, phi = np.split(result.x, split_count)
        return gamma, beta, phi
    elif parameters == 4:
        gamma, beta, phi, theta = np.split(result.x, split_count)
        return gamma, beta, phi, theta

def get_gradient(function, gamma, beta, delta_gamma, delta_beta, iter):
    """
    Compute the gradient of a function with respect to gamma and beta parameters.

    Parameters:
    function (callable): The function for which the gradient is computed.
    gamma (array-like): Current values of gamma parameters.
    beta (array-like): Current values of beta parameters.
    delta_gamma (float): Perturbation for gamma.
    delta_beta (float): Perturbation for beta.
    iter (int): Current iteration number.

    Returns:
    tuple: Gradients with respect to gamma and beta.
    """
    grad_gamma = np.zeros_like(gamma)
    grad_beta = np.zeros_like(beta)
    gamma_edge = gamma.copy()
    beta_edge = beta.copy()
    
    if not (gamma.size == beta.size):
        return 1

    for index in range(gamma.size):
        center = gamma[index]
        gamma_edge[index] = gamma[index] - delta_gamma
        e1 = function(gamma=gamma_edge, beta=beta)
        gamma_edge[index] = gamma[index] + delta_gamma
        e2 = function(gamma=gamma_edge, beta=beta)
        grad_gamma[index] = (e2.real - e1.real) / (2 * delta_gamma)
        gamma[index] = center

        center = beta[index]
        beta_edge[index] = beta[index] - delta_beta
        e1 = function(gamma=gamma, beta=beta_edge)
        beta_edge[index] = beta[index] + delta_beta
        e2 = function(gamma=gamma, beta=beta_edge)
        grad_beta[index] = (e2.real - e1.real) / (2 * delta_beta)
        beta[index] = center
    
    return grad_gamma, grad_beta

def optimize_by_gradient_descent(function, initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration, figure=True, filepath=""):
    """
    Optimize a function using gradient descent.

    Parameters:
    function (callable): The function to be optimized.
    initial_gamma (array-like): Initial values for gamma parameters.
    initial_beta (array-like): Initial values for beta parameters.
    alpha (float): Learning rate.
    delta_gamma (float): Perturbation for gamma.
    delta_beta (float): Perturbation for beta.
    iteration (int): Number of iterations.
    figure (bool): Whether to print the optimization process.
    filepath (str): Path to the CSV file for logging.

    Returns:
    tuple: Optimized gamma and beta parameters.
    """
    gamma, beta = initial_gamma, initial_beta

    textlines = []
    headline = ["iter", "energy"]
    for p in range(int(len(initial_gamma))):
        headline.append(f"gamma[{p}]")
        headline.append(f"beta[{p}]")
    print(headline)
    textlines.append(headline)

    for iter in range(int(iteration)):
        grad_gamma, grad_beta = get_gradient(function, gamma, beta, delta_gamma, delta_beta, iter)
        gamma -= alpha * grad_gamma
        beta -= alpha * grad_beta
        energy = function(gamma=gamma, beta=beta)

        record = [iter, energy] + list(gamma) + list(beta)
        textlines.append(record)
        print(record)
    
    if filepath:
        with open(filepath, mode='a') as f:
            writer = csv.writer(f)
            for textline in textlines:
                writer.writerow(textline)

    return gamma, beta

def get_gradient_gpu(function, gamma, beta, delta_gamma, delta_beta, iter):
    """
    Compute the gradient of a function with respect to gamma and beta parameters using GPU.

    Parameters:
    function (callable): The function for which the gradient is computed.
    gamma (array-like): Current values of gamma parameters.
    beta (array-like): Current values of beta parameters.
    delta_gamma (float): Perturbation for gamma.
    delta_beta (float): Perturbation for beta.
    iter (int): Current iteration number.

    Returns:
    tuple: Gradients with respect to gamma and beta.
    """
    grad_gamma = np.zeros_like(gamma)
    grad_beta = np.zeros_like(beta)
    gamma_edge = gamma.copy()
    beta_edge = beta.copy()
    
    if not (gamma.size == beta.size):
        return 1

    for index in range(gamma.size):
        center = gamma[index]
        gamma_edge[index] = gamma[index] - delta_gamma
        e1 = function(gamma=gamma_edge, beta=beta)
        gamma_edge[index] = gamma[index] + delta_gamma
        e2 = function(gamma=gamma_edge, beta=beta)
        grad_gamma[index] = (e2.real - e1.real) / (2 * delta_gamma)
        gamma[index] = center

        center = beta[index]
        beta_edge[index] = beta[index] - delta_beta
        e1 = function(gamma=gamma, beta=beta_edge)
        beta_edge[index] = beta[index] + delta_beta
        e2 = function(gamma=gamma, beta=beta_edge)
        grad_beta[index] = (e2.real - e1.real) / (2 * delta_beta)
        beta[index] = center
    
    return grad_gamma, grad_beta

def optimize_by_gradient_descent_gpu(function, initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration, figure=True, filepath=""):
    """
    Optimize a function using gradient descent on GPU.

    Parameters:
    function (callable): The function to be optimized.
    initial_gamma (array-like): Initial values for gamma parameters.
    initial_beta (array-like): Initial values for beta parameters.
    alpha (float): Learning rate.
    delta_gamma (float): Perturbation for gamma.
    delta_beta (float): Perturbation for beta.
    iteration (int): Number of iterations.
    figure (bool): Whether to print the optimization process.
    filepath (str): Path to the CSV file for logging.

    Returns:
    tuple: Optimized gamma and beta parameters.
    """
    gamma, beta = np.asarray(initial_gamma), np.asarray(initial_beta)

    textlines = []
    headline = ["iter", "energy"]
    for p in range(int(len(initial_gamma))):
        headline.append(f"gamma[{p}]")
        headline.append(f"beta[{p}]")
    print(headline)
    textlines.append(headline)

    for iter in range(int(iteration)):
        grad_gamma, grad_beta = get_gradient_gpu(function, gamma, beta, delta_gamma, delta_beta, iter)
        gamma -= alpha * grad_gamma
        beta -= alpha * grad_beta
        energy = function(gamma=gamma, beta=beta)

        record = [iter, energy] + list(gamma) + list(beta)
        textlines.append(record)
        print(record)
    
    if filepath:
        with open(filepath, mode='a') as f:
            writer = csv.writer(f)
            for textline in textlines:
                writer.writerow(textline)

    return gamma, beta

def gradient_parallel(pool, f, gamma, beta, h=1e-5):
    """
    Compute the gradient in parallel using multiprocessing.

    Parameters:
    pool (multiprocessing.Pool): Pool object for multiprocessing.
    f (callable): The function for which the gradient is computed.
    gamma (array-like): Current values of gamma parameters.
    beta (array-like): Current values of beta parameters.
    h (float): Perturbation for numerical differentiation.

    Returns:
    tuple: Gradients with respect to gamma and beta.
    """
    if not (gamma.size == beta.size):
        return None  # Return None if the vector sizes do not match

    n = gamma.size
    args_gamma = [(f, gamma, beta, i, h, 'gamma') for i in range(n)]
    args_beta = [(f, gamma, beta, i, h, 'beta') for i in range(n)]
    grad_gamma = pool.starmap(partial_derivative_gamma, args_gamma)
    grad_beta = pool.starmap(partial_derivative_beta, args_beta)
    return np.array(grad_gamma), np.array(grad_beta)

def partial_derivative_gamma(f, gamma, beta, i, h=1e-5, variable='gamma'):
    """
    Compute the partial derivative with respect to gamma.

    Parameters:
    f (callable): The function for which the derivative is computed.
    gamma (array-like): Current values of gamma parameters.
    beta (array-like): Current values of beta parameters.
    i (int): Index of the parameter.
    h (float): Perturbation for numerical differentiation.
    variable (str): Name of the variable (default is 'gamma').

    Returns:
    float: Partial derivative with respect to gamma[i].
    """
    var_plus = np.array(gamma, dtype=float)
    var_minus = np.array(var_plus, dtype=float)
    var_plus[i] += h
    var_minus[i] -= h
    derivative = (f(gamma=var_plus, beta=beta).real - f(gamma=var_minus, beta=beta).real) / (2 * h)
    return derivative

def partial_derivative_beta(f, gamma, beta, i, h=1e-5, variable='beta'):
    """
    Compute the partial derivative with respect to beta.

    Parameters:
    f (callable): The function for which the derivative is computed.
    gamma (array-like): Current values of gamma parameters.
    beta (array-like): Current values of beta parameters.
    i (int): Index of the parameter.
    h (float): Perturbation for numerical differentiation.
    variable (str): Name of the variable (default is 'beta').

    Returns:
    float: Partial derivative with respect to beta[i].
    """
    var_plus = np.array(beta, dtype=float)
    var_minus = np.array(var_plus, dtype=float)
    var_plus[i] += h
    var_minus[i] -= h
    derivative = (f(gamma=gamma, beta=var_plus).real - f(gamma=gamma, beta=var_minus).real) / (2 * h)
    return derivative

def optimize_by_gradient_descent_multiprocess(function, initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration, tol, figure=True, filepath="", pool=mp.Pool(2)):
    """
    Optimize a function using gradient descent with multiprocessing.

    Parameters:
    function (callable): The function to be optimized.
    initial_gamma (array-like): Initial values for gamma parameters.
    initial_beta (array-like): Initial values for beta parameters.
    alpha (float): Learning rate.
    delta_gamma (float): Perturbation for gamma.
    delta_beta (float): Perturbation for beta.
    iteration (int): Number of iterations.
    tol (float): Tolerance for convergence.
    figure (bool): Whether to print the optimization process.
    filepath (str): Path to the CSV file for logging.
    pool (multiprocessing.Pool): Pool object for multiprocessing.

    Returns:
    tuple: Optimized gamma and beta parameters.
    """
    gamma, beta = initial_gamma.copy(), initial_beta.copy()
    min_iterations = max(1, int(0.1 * iteration)) if iteration != -1 else 1  # Ensure at least 10% of the total iterations, minimum of 1

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        headline = ["iter", "energy"]
        for p in range(int(len(initial_gamma))):
            headline.append(f"gamma[{p}]")
            headline.append(f"beta[{p}]")
        print(headline)
        writer.writerow(headline)

        iter = 0
        while True:
            if iter == 0:
                energy = 0
            prev_gamma = gamma.copy()
            prev_beta = beta.copy()
            prev_energy = energy
            
            grad_gamma, grad_beta = gradient_parallel(pool, function, gamma, beta, delta_gamma)
            
            gamma -= alpha * grad_gamma
            beta -= alpha * grad_beta
            
            energy = function(gamma=gamma, beta=beta)

            gamma_change = max(abs((gamma - prev_gamma) / (prev_gamma + 1e-10)))
            beta_change = max(abs((beta - prev_beta) / (prev_beta + 1e-10)))        
            energy_change = abs((energy - prev_energy) / (prev_energy + 1e-10))
            
            if iter >= min_iterations and energy_change < tol:
                print(f"Converged at iteration {iter}")
                break
            
            record = [iter, energy] + [val for pair in zip(gamma, beta) for val in pair]
            writer.writerow(record)
            if figure:
                print(record)

            if iteration != -1 and iter >= iteration - 1:
                break

            iter += 1

    return gamma, beta
