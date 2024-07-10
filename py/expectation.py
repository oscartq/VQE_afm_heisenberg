import cirq
import openfermion as of
import numpy as np
import datetime
from anzats import Anzats, AnzatsAFMHeisenberg, AnzatsAFMHeisenberg_periodic, AnzatsAFMHeisenbergLattice
import qsimcirq

class AFMHeisenbergLatticeArgs():
    def __init__(self, rows, cols, qsim_option):
        self.rows = rows
        self.cols  = cols
        self.qsim_option = qsim_option

def get_expectation_afm_heisenberg_lattice(function_args, gamma, beta):
    # periodic boundary
    anzats = AnzatsAFMHeisenbergLattice(function_args.rows, function_args.cols, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()

    rows = function_args.rows
    cols  = function_args.cols
    value = 0 + 0j
    for i in range(rows):
        for j in range(cols):
            current_index = j*rows + i

            right_neighbor = j * rows + (i + 1) % rows
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()

            circuitX.append(cirq.X(qubits[current_index]))
            circuitX.append(cirq.X(qubits[right_neighbor]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            circuitY.append(cirq.Y(qubits[current_index]))
            circuitY.append(cirq.Y(qubits[right_neighbor]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            circuitZ.append(cirq.Z(qubits[current_index]))
            circuitZ.append(cirq.Z(qubits[right_neighbor]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)


            down_neighbor = ((j + 1) % cols) * rows + i 
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()

            circuitX.append(cirq.X(qubits[current_index]))
            circuitX.append(cirq.X(qubits[down_neighbor]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            circuitY.append(cirq.Y(qubits[current_index]))
            circuitY.append(cirq.Y(qubits[down_neighbor]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            circuitZ.append(cirq.Z(qubits[current_index]))
            circuitZ.append(cirq.Z(qubits[down_neighbor]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)
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

def get_expectation_afm_heisenberg_periodic(function_args, gamma, beta):
    # periodic
    anzats = AnzatsAFMHeisenberg_periodic(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    length = function_args.length
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j
    for i in range(function_args.length-1):
        circuitX = anzats.circuit.copy()
        circuitY = anzats.circuit.copy()
        circuitZ = anzats.circuit.copy()

        circuitX.append(cirq.X(qubits[i%length]))
        circuitX.append(cirq.X(qubits[(i+1)%length]))
        vector2 = simulator.simulate(circuitX).state_vector()
        value += np.dot(vector2.conj(), vector)

        circuitY.append(cirq.Y(qubits[i%length]))
        circuitY.append(cirq.Y(qubits[(i+1)%length]))
        vector2 = simulator.simulate(circuitY).state_vector()
        value += np.dot(vector2.conj(), vector)

        circuitZ.append(cirq.Z(qubits[i%length]))
        circuitZ.append(cirq.Z(qubits[(i+1)%length]))
        vector2 = simulator.simulate(circuitZ).state_vector()
        value += np.dot(vector2.conj(), vector)
    return np.real(value)

def get_expectation_afm_heisenberg_new_symmetry(function_args, gamma, beta):
    # This function calculates the expectation value for the AFM Heisenberg model with a new symmetry
    # using a 2D quantum circuit ansatz.

    # Initialize the ansatz for the 2D AFM Heisenberg model with given parameters
    anzats = AnzatsAFMHeisenberg_2d(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits

    # Initialize the quantum simulator
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)

    # Simulate the circuit to get the initial state vector
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j  # Initialize the expectation value as a complex number

    rows = 2  # Number of rows in the 2D grid of qubits
    cols = int(function_args.length / 2)  # Number of columns in the 2D grid of qubits

    # Iterate over the 2D grid of qubits
    for i in range(rows):
        for j in range(cols - 1):
            # Horizontal neighbors

            # Create copies of the original circuit for each Pauli operator (X, Y, Z)
            circuitX = circuit.copy()
            circuitY = circuit.copy()
            circuitZ = circuit.copy()

            # Apply Pauli-X operators to the horizontal neighbors and simulate the circuit
            circuitX.append(cirq.X(qubits[i * cols + j]))
            circuitX.append(cirq.X(qubits[i * cols + j + 1]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            
            # Apply Pauli-Y operators to the horizontal neighbors and simulate the circuit
            circuitY.append(cirq.Y(qubits[i * cols + j]))
            circuitY.append(cirq.Y(qubits[i * cols + j + 1]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            
            # Apply Pauli-Z operators to the horizontal neighbors and simulate the circuit
            circuitZ.append(cirq.Z(qubits[i * cols + j]))
            circuitZ.append(cirq.Z(qubits[i * cols + j + 1]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            
    for i in range(rows - 1):
        for j in range(cols):
            # Vertical neighbors

            # Create copies of the original circuit for each Pauli operator (X, Y, Z)
            circuitX = circuit.copy()
            circuitY = circuit.copy()
            circuitZ = circuit.copy()

            # Apply Pauli-X operators to the vertical neighbors and simulate the circuit
            circuitX.append(cirq.X(qubits[i * cols + j]))
            circuitX.append(cirq.X(qubits[(i + 1) * cols + j]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            
            # Apply Pauli-Y operators to the vertical neighbors and simulate the circuit
            circuitY.append(cirq.Y(qubits[i * cols + j]))
            circuitY.append(cirq.Y(qubits[(i + 1) * cols + j]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            
            # Apply Pauli-Z operators to the vertical neighbors and simulate the circuit
            circuitZ.append(cirq.Z(qubits[i * cols + j]))
            circuitZ.append(cirq.Z(qubits[(i + 1) * cols + j]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            
    # Return the real part of the expectation value
    return np.real(value)