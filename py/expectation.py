import cirq
import openfermion
import numpy as np
import datetime
from anzats import Anzats, AnzatsAFMHeisenberg, AnzatsToricCode
import qsimcirq
import cupy as cp
# from expectation import get_expectation_ZiZj, get_expectation_ghz_l4, get_expectation_ghz_l8
# from optimization import get_gradient, optimize_by_gradient_descent

class TFIMStateArgs():
    def __init__(self, length, g, qsim_option=None):
        self.length = length
        self.g = g
        self.qsim_option = qsim_option


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
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
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

def get_expectation_critical_state_multicore(function_args, gamma, beta):
    anzats = Anzats(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j
    threads = []


    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(cirq.Z(qubits[(i+1)%function_args.length]))
        # simulator = qsimcirq.QSimSimulator()
        vector2 = simulator.simulate(circuit).state_vector()
        return -cp.dot(vector2.conj(), vector)
    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        # circuit.append(function_args.g * cirq.X(qubits[i]))
        circuit.append(cirq.XPowGate(exponent=function_args.g).on(qubits[i]))
        # simulator = qsimcirq.QSimSimulator()
        vector2 = simulator.simulate(circuit).state_vector()
        value -= cp.dot(vector2.conj(), vector)
    return value

class AFMHeisenbergArgs():
    def __init__(self, length, qsim_option):
        self.length = length
        self.qsim_option = qsim_option


def get_expectation_afm_heisenberg(function_args, gamma, beta):
    # open boundary
    anzats = AnzatsAFMHeisenberg(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j
    for i in range(function_args.length-1):
        circuitX = anzats.circuit.copy()
        circuitY = anzats.circuit.copy()
        circuitZ = anzats.circuit.copy()

        circuitX.append(cirq.X(qubits[i]))
        circuitX.append(cirq.X(qubits[(i+1)]))
        vector2 = simulator.simulate(circuitX).state_vector()
        value += np.dot(vector2.conj(), vector)

        circuitY.append(cirq.Y(qubits[i]))
        circuitY.append(cirq.Y(qubits[(i+1)]))
        vector2 = simulator.simulate(circuitY).state_vector()
        value += np.dot(vector2.conj(), vector)

        circuitZ.append(cirq.Z(qubits[i]))
        circuitZ.append(cirq.Z(qubits[(i+1)]))
        vector2 = simulator.simulate(circuitZ).state_vector()
        value += np.dot(vector2.conj(), vector)
    return value

class ToricCodeArgs():
    def __init__(self, length, qsim_option):
        self.length = length
        self.qsim_option = qsim_option


def get_expectation_toric_code(function_args, gamma, beta):
    # open boundary
    anzats = AnzatsToricCode(function_args.length, gamma, beta)
    length = function_args.length

    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()
    value = 0 + 0j
    
    for i in range(function_args.length-1):
        for j in range(function_args.length-1):
            circuit = anzats.circuit.copy()
            circuit.append(cirq.X(qubits[i][(j+1)%length]))
            circuit.append(cirq.Y(qubits[(i+1)%length][(j+1)%length]))
            circuit.append(cirq.X(qubits[(i+1)%length][j]))
            circuit.append(cirq.Y(qubits[i][j]))
            vector2 = simulator.simulate(circuit).state_vector()
            value -= np.dot(vector2.conj(), vector)

    return value

