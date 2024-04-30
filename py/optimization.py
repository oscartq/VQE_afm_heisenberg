import os 
import sys
import datetime
import csv 
import multiprocessing as mp

import cirq
import openfermion
import numpy as np
import cupy as cp
Pi=3.1415

from anzats import Anzats
# from expectation import get_expectation_ZiZj, get_expectation_ghz_l4, get_expectation_ghz_l8
# from optimization import get_gradient, optimize_by_gradient_descent

from multiprocessing import Process, Pipe, Pool

def get_gradient(function, gamma: np.array, beta: np.array, delta_gamma, delta_beta, iter):
    grad_gamma = np.zeros_like(gamma)
    grad_beta  = np.zeros_like(beta)
    gamma_edge = gamma
    beta_edge  = beta
    # initial gamma, beta?
    
    if not (gamma.size == beta.size):
        return 1

    for index in range(gamma.size):
        center = gamma[index]
        gamma_edge[index] = gamma[index] - delta_gamma
        e1 = function(gamma=gamma_edge, beta=beta)
        gamma_edge[index] = gamma[index] + delta_gamma
        e2 = function(gamma=gamma_edge, beta=beta)
        grad_gamma[index] = (e2.real-e1.real)/(2*delta_gamma)
        gamma[index] = center

        center = beta[index]
        beta_edge[index] = beta[index] - delta_beta
        e1 = function(gamma=gamma, beta=beta_edge)
        beta_edge[index] = beta[index] + delta_beta
        e2 = function(gamma=gamma, beta=beta_edge)
        grad_beta[index] = (e2.real-e1.real)/(2*delta_beta)
        beta[index] = center
    
    return grad_gamma, grad_beta


def optimize_by_gradient_descent(function, initial_gamma: np.array, initial_beta: np.array, alpha, delta_gamma, delta_beta, iteration, Figure=True,filepath=""):
    gamma, beta = initial_gamma, initial_beta

    textlines = []
    headline = ["iter", "energy"]
    for p in range(int(len(initial_gamma))):
        headline.append("gamma[{}]".format(p))
        headline.append("bata[{}]".format(p))
    print(headline)
    textlines.append(headline)

    for iter in range(int(iteration)):
        # it is complex for me to set get_gradient for two optical parameter_vector
        grad_gamma, grad_beta = get_gradient(function, gamma, beta, delta_gamma, delta_beta, iter)
        gamma -= alpha * grad_gamma
        beta  -= alpha * grad_beta
        energy = function(gamma=gamma, beta=beta)

        record = [iter, energy]
        for index in range(gamma.size):
            record.append(gamma[index])
            record.append(beta[index])
        textlines.append(record)
        print(record)
    
    if len(filepath)>0:
        with open(filepath, mode='a') as f:
            writer = csv.writer(f)
            for i, textline in enumerate(textlines):
                writer.writerow(textline)
                # f.write("{}\n".format(textline))

    return gamma, beta

def get_gradient_gpu(function, gamma: np.array, beta: np.array, delta_gamma, delta_beta, iter):
    grad_gamma = np.zeros_like(gamma)
    grad_beta  = np.zeros_like(beta)
    gamma_edge = gamma
    beta_edge  = beta
    # initial gamma, beta?
    
    if not (gamma.size == beta.size):
        return 1

    for index in range(gamma.size):
        center = gamma[index]
        gamma_edge[index] = gamma[index] - delta_gamma
        e1 = function(gamma=gamma_edge, beta=beta)
        gamma_edge[index] = gamma[index] + delta_gamma
        e2 = function(gamma=gamma_edge, beta=beta)
        grad_gamma[index] = (e2.real-e1.real)/(2*delta_gamma)
        gamma[index] = center

        center = beta[index]
        beta_edge[index] = beta[index] - delta_beta
        e1 = function(gamma=gamma, beta=beta_edge)
        beta_edge[index] = beta[index] + delta_beta
        e2 = function(gamma=gamma, beta=beta_edge)
        grad_beta[index] = (e2.real-e1.real)/(2*delta_beta)
        beta[index] = center
    
    return grad_gamma, grad_beta


def optimize_by_gradient_descent_gpu(function, initial_gamma: np.array, initial_beta: np.array, alpha, delta_gamma, delta_beta, iteration, Figure=True,filepath=""):
    gamma, beta = np.asarray(initial_gamma), np.asarray(initial_beta)

    textlines = []
    headline = ["iter", "energy"]
    for p in range(int(len(initial_gamma))):
        headline.append("gamma[{}]".format(p))
        headline.append("bata[{}]".format(p))
    print(headline)
    textlines.append(headline)

    for iter in range(int(iteration)):
        # it is complex for me to set get_gradient for two optical parameter_vector
        grad_gamma, grad_beta = get_gradient_gpu(function, gamma, beta, delta_gamma, delta_beta, iter)
        gamma -= alpha * grad_gamma
        beta  -= alpha * grad_beta
        energy = function(gamma=gamma, beta=beta)

        record = [iter, energy]
        for index in range(gamma.size):
            record.append(gamma[index])
            record.append(beta[index])
        textlines.append(record)
        print(record)
    
    if len(filepath)>0:
        with open(filepath, mode='a') as f:
            writer = csv.writer(f)
            for i, textline in enumerate(textlines):
                writer.writerow(textline)
                # f.write("{}\n".format(textline))

    return gamma, beta


def gradient_parallel(pool, f, gamma, beta, h=1e-5):
    if not (gamma.size == beta.size):
        return None  # ベクトルサイズが一致しなければNoneを返す

    n = gamma.size
    args_gamma = [(f, gamma, beta, i, h, 'gamma') for i in range(n)]
    args_beta = [(f, gamma, beta, i, h, 'beta') for i in range(n)]
    grad_gamma = pool.starmap(partial_derivative_gamma, args_gamma)
    grad_beta = pool.starmap(partial_derivative_beta, args_beta)
    # pool.join()
    return np.array(grad_gamma), np.array(grad_beta)

def partial_derivative_gamma(f, gamma, beta, i, h=1e-5, variable='gamma'):
    var_plus = np.array(gamma, dtype=float)
    var_minus = np.array(var_plus, dtype=float)
    var_plus[i] += h
    var_minus[i] -= h
    derivative = (f(gamma=var_plus, beta=beta).real - f(gamma=var_minus, beta=beta).real) / (2 * h)
    return derivative

def partial_derivative_beta(f, gamma, beta, i, h=1e-5, variable='gamma'):
    var_plus = np.array(beta, dtype=float)
    var_minus = np.array(var_plus, dtype=float)
    var_plus[i] += h
    var_minus[i] -= h
    derivative = (f(gamma=gamma, beta=var_plus).real - f(gamma=gamma, beta=var_minus).real) / (2 * h)
    return derivative

def optimize_by_gradient_descent_multiprocess(function, initial_gamma, initial_beta, alpha, delta_gamma, delta_beta, iteration, figure=True, filepath="", pool=mp.Pool(2)):
    gamma, beta = initial_gamma.copy(), initial_beta.copy()

    with open(filepath, mode='a', newline='') as f:
        writer = csv.writer(f)
        headline = ["iter", "energy"]
        for p in range(int(len(initial_gamma))):
            headline.append("gamma[{}]".format(p))
            headline.append("bata[{}]".format(p))
        print(headline)
        writer.writerow(headline)

        for iter in range(iteration):
            grad_gamma, grad_beta = gradient_parallel(pool, function, gamma, beta, delta_gamma)
            # print(grad_beta, grad_gamma)
            gamma -= alpha * grad_gamma
            beta -= alpha * grad_beta
            energy = function(gamma=gamma, beta=beta)

            record = [iter, energy] + [val for pair in zip(gamma, beta) for val in pair]
            writer.writerow(record)
            if figure:
                print(record)

    return gamma, beta
