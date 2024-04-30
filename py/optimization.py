import os 
import sys
import datetime
import csv 

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


def optimize_by_gradient_descent_multiprocess(function, initial_gamma: np.array, initial_beta: np.array, alpha, delta_gamma, delta_beta, iteration, Figure=True,filepath=""):
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
        grad_gamma, grad_beta = get_gradient_multiprocess(function, gamma, beta, delta_gamma, delta_beta, iter)
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

def get_gradient_multiprocess(function, gamma: np.array, beta: np.array, delta_gamma, delta_beta, iter):
    grad_gamma = np.zeros_like(gamma)
    grad_beta  = np.zeros_like(beta)
    gamma_edge = gamma
    beta_edge  = beta
    # initial gamma, beta?
    
    if not (gamma.size == beta.size):
        return 1



    tasks = []
    job_list = []
    pipe_list = []
    p = Pool(4) # プロセス数を4に設定

    for index in range(gamma.size):
        get_rev,send_rev  = Pipe(False)
        center = gamma[index]
        gamma_edge[index] = gamma[index] - delta_gamma
        p1 = p.apply_async(function, args=[gamma_edge, beta])
        gamma_edge[index] = gamma[index] + delta_gamma
        p2 = p.apply_async(function, args=[gamma_edge, beta])
        gamma[index] = center


        center = beta[index]
        beta_edge[index] = beta[index] - delta_beta
        p3 = p.apply_async(function, args=[gamma, beta_edge])
        beta_edge[index] = beta[index] + delta_beta
        p4 = p.apply_async(function, args=[gamma, beta_edge])

        grad_gamma[index] = (p2.get().real-p1.get().real)/(2*delta_gamma)
        grad_beta[index] = (p4.get().real-p3.get().real)/(2*delta_beta)
        beta[index] = center
    
    p.close()

    return grad_gamma, grad_beta

