import cirq
import openfermion as of
import numpy as np
from anzats import AnzatsAFMHeisenberg, AnzatsAFMHeisenbergLattice, AnzatsAFMHeisenbergMatrix
import qsimcirq

class AFMHeisenbergArgs:
    """
    Arguments for the AFM Heisenberg model.
    
    Attributes:
        length (int): Length of the 1D lattice.
        qsim_option (dict): Options for the qsim simulator.
    """
    
    def __init__(self, length, qsim_option):
        self.length = length
        self.qsim_option = qsim_option

def get_expectation_afm_heisenberg(function_args, gamma, beta):
    """
    Calculate the expectation value for the AFM Heisenberg model using a quantum circuit ansatz.
    
    Args:
        function_args (AFMHeisenbergArgs): Arguments for the AFM Heisenberg model.
        gamma (np.ndarray): Array of gamma parameters.
        beta (np.ndarray): Array of beta parameters.
        
    Returns:
        float: Real part of the calculated expectation value.
    """
    
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

class AFMHeisenbergLatticeArgs:
    """
    Arguments for the AFM Heisenberg model on a lattice.
    
    Attributes:
        rows (int): Number of rows in the lattice.
        cols (int): Number of columns in the lattice.
        periodic (bool): If True, periodic boundary conditions are used.
        qsim_option (dict): Options for the qsim simulator.
    """
    
    def __init__(self, rows, cols, periodic, qsim_option):
        self.rows = rows
        self.cols = cols
        self.periodic = periodic
        self.qsim_option = qsim_option

def get_expectation_afm_heisenberg_lattice(function_args, gamma, beta, phi):
    """
    Calculate the expectation value for the AFM Heisenberg model on a lattice using a quantum circuit ansatz.
    
    Args:
        function_args (AFMHeisenbergLatticeArgs): Arguments for the AFM Heisenberg lattice model.
        gamma (np.ndarray): Array of gamma parameters.
        beta (np.ndarray): Array of beta parameters.
        phi (np.ndarray): Array of phi parameters.
        
    Returns:
        float: Real part of the calculated expectation value.
    """
    
    # Variables from function_args
    rows = function_args.rows
    cols = function_args.cols
    periodic = function_args.periodic
    
    # Create an instance of the AnzatsAFMHeisenbergLattice class
    anzats = AnzatsAFMHeisenbergLattice(rows, cols, gamma, beta, phi, periodic)
    
    # Extract the circuit and qubits from the anzats object
    circuit = anzats.circuit
    qubits = anzats.qubits
    
    # Initialize the simulator with the provided options
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    
    # Simulate the circuit and get the state vector
    vector = simulator.simulate(circuit).state_vector()
    
    edge = 0 if periodic else 1
    value = 0 + 0j

    # Calculate the expectation value for row interactions
    for i in range(rows):
        for j in range(cols - edge):
            right_neighbor = (j + 1) % cols

            # Create copies of the circuit for X, Y, Z operations
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()
            
            # Append X operations and simulate
            circuitX.append(cirq.X(cirq.GridQubit(i, j)))
            circuitX.append(cirq.X(cirq.GridQubit(i, right_neighbor)))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Y operations and simulate
            circuitY.append(cirq.Y(cirq.GridQubit(i, j)))
            circuitY.append(cirq.Y(cirq.GridQubit(i, right_neighbor)))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Z operations and simulate
            circuitZ.append(cirq.Z(cirq.GridQubit(i, j)))
            circuitZ.append(cirq.Z(cirq.GridQubit(i, right_neighbor)))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)

    # Calculate the expectation value for column interactions
    for i in range(rows - edge):
        for j in range(cols):
            bottom_neighbor = (i + 1) % rows

            # Create copies of the circuit for X, Y, Z operations
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()

            # Append X operations and simulate
            circuitX.append(cirq.X(cirq.GridQubit(i, j)))
            circuitX.append(cirq.X(cirq.GridQubit(bottom_neighbor, j)))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Y operations and simulate
            circuitY.append(cirq.Y(cirq.GridQubit(i, j)))
            circuitY.append(cirq.Y(cirq.GridQubit(bottom_neighbor, j)))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Z operations and simulate
            circuitZ.append(cirq.Z(cirq.GridQubit(i, j)))
            circuitZ.append(cirq.Z(cirq.GridQubit(bottom_neighbor, j)))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)

    # Return the real part of the calculated value
    return np.real(value)

class AFMHeisenbergMatrixArgs:
    """
    Arguments for the AFM Heisenberg model on a matrix.
    
    Attributes:
        rows (int): Number of rows in the matrix.
        cols (int): Number of columns in the matrix.
        periodic (bool): If True, periodic boundary conditions are used.
        qsim_option (dict): Options for the qsim simulator.
    """
    
    def __init__(self, rows, cols, periodic, qsim_option):
        self.rows = rows
        self.cols = cols
        self.periodic = periodic
        self.qsim_option = qsim_option

def get_expectation_afm_heisenberg_matrix(function_args, gamma, beta, phi, theta):
    """
    Calculate the expectation value for the AFM Heisenberg model on a matrix using a quantum circuit ansatz.
    
    Args:
        function_args (AFMHeisenbergMatrixArgs): Arguments for the AFM Heisenberg matrix model.
        gamma (np.ndarray): Array of gamma parameters.
        beta (np.ndarray): Array of beta parameters.
        phi (np.ndarray): Array of phi parameters.
        theta (np.ndarray): Array of theta parameters.
        
    Returns:
        float: Real part of the calculated expectation value.
    """
    
    # Variables from function_args
    rows = function_args.rows
    cols = function_args.cols
    periodic = function_args.periodic
    
    anzats = AnzatsAFMHeisenbergMatrix(rows, cols, gamma, beta, phi, theta, periodic)
    
    # Extract the circuit and qubits from the anzats object
    circuit = anzats.circuit
    qubits = anzats.qubits
    
    # Initialize the simulator with the provided options
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    
    # Simulate the circuit and get the state vector
    vector = simulator.simulate(circuit).state_vector()
    
    edge = 0 if periodic else 1
    value = 0 + 0j
    
    # Calculate the expectation value for row interactions
    for i in range(rows):
        for j in range(cols - edge):
            right_neighbor = (j + 1) % cols

            # Create copies of the circuit for X, Y, Z operations
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()
            
            # Append X operations and simulate
            circuitX.append(cirq.X(cirq.GridQubit(i, j)))
            circuitX.append(cirq.X(cirq.GridQubit(i, right_neighbor)))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Y operations and simulate
            circuitY.append(cirq.Y(cirq.GridQubit(i, j)))
            circuitY.append(cirq.Y(cirq.GridQubit(i, right_neighbor)))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Z operations and simulate
            circuitZ.append(cirq.Z(cirq.GridQubit(i, j)))
            circuitZ.append(cirq.Z(cirq.GridQubit(i, right_neighbor)))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)

    # Calculate the expectation value for column interactions
    for i in range(rows - edge):
        for j in range(cols):
            bottom_neighbor = (i + 1) % rows

            # Create copies of the circuit for X, Y, Z operations
            circuitX = anzats.circuit.copy()
            circuitY = anzats.circuit.copy()
            circuitZ = anzats.circuit.copy()

            # Append X operations and simulate
            circuitX.append(cirq.X(cirq.GridQubit(i, j)))
            circuitX.append(cirq.X(cirq.GridQubit(bottom_neighbor, j)))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Y operations and simulate
            circuitY.append(cirq.Y(cirq.GridQubit(i, j)))
            circuitY.append(cirq.Y(cirq.GridQubit(bottom_neighbor, j)))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)

            # Append Z operations and simulate
            circuitZ.append(cirq.Z(cirq.GridQubit(i, j)))
            circuitZ.append(cirq.Z(cirq.GridQubit(bottom_neighbor, j)))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)

    # Return the real part of the calculated value
    return np.real(value)