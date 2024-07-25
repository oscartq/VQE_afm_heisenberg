import cirq
import openfermion as of
import numpy as np
import datetime
from anzats import Anzats, AnzatsAFMHeisenberg, AnzatsAFMHeisenbergLattice_grid
import qsimcirq

class AFMHeisenbergLatticeArgs():
    def __init__(self, rows, cols, periodic, qsim_option):
        self.rows = rows
        self.cols  = cols
        self.qsim_option = qsim_option
        self.periodic = periodic

def get_expectation_afm_heisenberg_lattice(function_args, gamma, beta, phi):
    # Variables from function_args
    rows = function_args.rows
    cols = function_args.cols
    periodic = function_args.periodic
    
    # Create an instance of the AnzatsAFMHeisenbergLattice_3p class
    anzats = AnzatsAFMHeisenbergLattice_grid(function_args.rows, function_args.cols, gamma, beta, phi, periodic)
    
    # Extract the circuit and qubits from the anzats object
    circuit = anzats.circuit
    qubits = anzats.qubits
    
    # Initialize the simulator with the provided options
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    
    # Simulate the circuit and get the state vector
    vector = simulator.simulate(circuit).state_vector()
    
    edge = 1-1 if periodic else 1-0
    value = 0 + 0j

    # Calculate the expectation value for row interactions
    for i in range(rows - edge):
        for j in range(cols):
            current_index = j * rows + i
            right_neighbor = j * rows + (i + 1) % rows

            # Create copies of the circuit for X, Y, Z operations
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()

            # Append X operations and simulate
            circuitX.append(cirq.X(qubits[current_index]))
            circuitX.append(cirq.X(qubits[right_neighbor]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Y operations and simulate
            circuitY.append(cirq.Y(qubits[current_index]))
            circuitY.append(cirq.Y(qubits[right_neighbor]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Z operations and simulate
            circuitZ.append(cirq.Z(qubits[current_index]))
            circuitZ.append(cirq.Z(qubits[right_neighbor]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)

    # Calculate the expectation value for column interactions
    for i in range(rows):
        for j in range(cols - edge):
            current_index = j * rows + i
            down_neighbor = ((j + 1) % cols) * rows + i

            # Create copies of the circuit for X, Y, Z operations
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()

            # Append X operations and simulate
            circuitX.append(cirq.X(qubits[current_index]))
            circuitX.append(cirq.X(qubits[down_neighbor]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Y operations and simulate
            circuitY.append(cirq.Y(qubits[current_index]))
            circuitY.append(cirq.Y(qubits[down_neighbor]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Z operations and simulate
            circuitZ.append(cirq.Z(qubits[current_index]))
            circuitZ.append(cirq.Z(qubits[down_neighbor]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)

    # Return the real part of the calculated value
    return np.real(value)

class AFMHeisenbergArgs():
    def __init__(self, length, qsim_option):
        self.length = length
        self.qsim_option = qsim_option

def get_expectation_afm_heisenberg(function_args, gamma, beta):
    # This function calculates the expectation value for the AFM Heisenberg model using a quantum circuit ansatz.

    # Initialize the ansatz for the AFM Heisenberg model with given parameters
    anzats = AnzatsAFMHeisenberg(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits

    # Initialize the quantum simulator
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)

    # Simulate the circuit to get the initial state vector
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j  # Initialize the expectation value as a complex number

    # Loop through each pair of neighboring qubits
    for i in range(function_args.length - 1):
        # Create copies of the original circuit for each Pauli operator (X, Y, Z)
        circuitX = anzats.circuit.copy()
        circuitY = anzats.circuit.copy()
        circuitZ = anzats.circuit.copy()

        # Apply Pauli-X operators to the i-th and (i+1)-th qubits and simulate the circuit
        circuitX.append(cirq.X(qubits[i]))
        circuitX.append(cirq.X(qubits[(i + 1)]))
        vector2 = simulator.simulate(circuitX).state_vector()
        value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value

        # Apply Pauli-Y operators to the i-th and (i+1)-th qubits and simulate the circuit
        circuitY.append(cirq.Y(qubits[i]))
        circuitY.append(cirq.Y(qubits[(i + 1)]))
        vector2 = simulator.simulate(circuitY).state_vector()
        value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
        
        # Apply Pauli-Z operators to the i-th and (i+1)-th qubits and simulate the circuit
        circuitZ.append(cirq.Z(qubits[i]))
        circuitZ.append(cirq.Z(qubits[(i + 1)]))
        vector2 = simulator.simulate(circuitZ).state_vector()
        value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
        
    # Return the real part of the expectation value
    return np.real(value)