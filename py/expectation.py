import cirq
import openfermion
import numpy as np
import datetime
from anzats import Anzats
import qsimcirq
import cupy as cp
# from expectation import get_expectation_ZiZj, get_expectation_ghz_l4, get_expectation_ghz_l8
# from optimization import get_gradient, optimize_by_gradient_descent

class TFIMStateArgs():
    def __init__(self, length, g):
        self.length = length
        self.g = g


def get_expectation_critical_state(function_args, gamma, beta):
    anzats = Anzats(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    vector = cirq.final_state_vector(circuit)

    value = 0 + 0j
    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(cirq.Z(qubits[(i+1)%function_args.length]))
        vector2 = cirq.final_state_vector(circuit)
        value -= np.dot(vector2.conj(), vector)
    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        # circuit.append(function_args.g * cirq.X(qubits[i]))
        circuit.append(cirq.XPowGate(exponent=function_args.g).on(qubits[i]))
        vector2 = cirq.final_state_vector(circuit)
        value -= np.dot(vector2.conj(), vector)
    return value

def get_expectation_ghz(length, gamma, beta):
    anzats = Anzats(length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    vector = cirq.final_state_vector(circuit)

    value = 0 + 0j
    for i in range(length):
        circuit = anzats.circuit.copy()
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(cirq.Z(qubits[(i+1)%length]))
        vector2 = cirq.final_state_vector(circuit)
        value -= np.dot(vector2.conj(), vector)
    return value

def get_expectation_ghz_l4(gamma: np.array, beta: np.array):
    return get_expectation_ghz(4, gamma, beta)

def get_expectation_ghz_l8(gamma: np.array, beta: np.array):
    return get_expectation_ghz(8, gamma, beta)

def get_expectation_ghz_l10(gamma: np.array, beta: np.array):
    return get_expectation_ghz(10, gamma, beta)

def get_expectation_ghz_l12(gamma: np.array, beta: np.array):
    return get_expectation_ghz(12, gamma, beta)

def get_expectation_ghz_l14(gamma: np.array, beta: np.array):
    return get_expectation_ghz(14, gamma, beta)

def get_expectation_ghz_l16(gamma: np.array, beta: np.array):
    return get_expectation_ghz(16, gamma, beta)


def get_expectation_critical_state_gpu(function_args, gamma, beta):
    anzats = Anzats(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator()
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j
    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(cirq.Z(qubits[(i+1)%function_args.length]))
        # simulator = qsimcirq.QSimSimulator()
        vector2 = simulator.simulate(circuit).state_vector()
        value -= cp.dot(vector2.conj(), vector)
    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        # circuit.append(function_args.g * cirq.X(qubits[i]))
        circuit.append(cirq.XPowGate(exponent=function_args.g).on(qubits[i]))
        # simulator = qsimcirq.QSimSimulator()
        vector2 = simulator.simulate(circuit).state_vector()
        value -= cp.dot(vector2.conj(), vector)
    return value