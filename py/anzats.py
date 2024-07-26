import cirq
import openfermion as of
import numpy as np
import datetime
import matplotlib.pyplot as plt
from numpy import pi as Pi

class AnzatsAFMHeisenberg():
    """
    Class to construct and represent an ansatz for the AFM Heisenberg model on a 1D chain.
    
    Attributes:
        circuit (cirq.Circuit): The quantum circuit for the ansatz.
        qubits (List[cirq.LineQubit]): List of qubits used in the circuit.
        gamma (np.ndarray): Array of gamma parameters for the circuit.
        beta (np.ndarray): Array of beta parameters for the circuit.
    """
    def __init__(self, length, gamma, beta, periodic=True):
        """
        Initialize the AFM Heisenberg ansatz circuit for a 1D chain.
        
        Args:
            length (int): Number of qubits in the chain.
            gamma (np.ndarray): Array of gamma parameters.
            beta (np.ndarray): Array of beta parameters.
            periodic (bool): If True, periodic boundary conditions (PBC) are used; otherwise, open boundary conditions (OBC).
        """
        
        edge = 0 if periodic else 1  # Edge = 0 for PBC and edge = 1 for OBC
        
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length)

        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        # Even qubits get H + Y-gates and odd qubits get X-gates, CNOT gates between for correlation
        for i in range(0, length, 2):
            circuit.append(cirq.H(qubits[i]))
            circuit.append(cirq.Y(qubits[i]))
            circuit.append(cirq.X(qubits[i+1]))
            circuit.append(cirq.CNOT(qubits[i], qubits[i+1]))

        # Add correlation gates XX, YY, ZZ
        for index in range(len(gamma)):
            # Add gamma circuit, first AFM Hamiltonian
            for i in range(0, length, 2):
                circuit.append(cirq.XX(qubits[i], qubits[(i+1)]) ** (-gamma[index]*2/Pi))
                circuit.append(cirq.YY(qubits[i], qubits[(i+1)]) ** (-gamma[index]*2/Pi))
                circuit.append(cirq.ZZ(qubits[i], qubits[(i+1)]) ** (-gamma[index]*2/Pi))
                
            # Add beta circuit, second AFM Hamiltonian
            for i in range(1, length-edge, 2):
                right_neighbor = (i + 1) % length  # Modulus operation sets (i+1) = 0 if i+1 > length, only applicable for PBC
                circuit.append(cirq.XX(qubits[i], qubits[right_neighbor]) ** (-beta[index]*2/Pi))
                circuit.append(cirq.YY(qubits[i], qubits[right_neighbor]) ** (-beta[index]*2/Pi))
                circuit.append(cirq.ZZ(qubits[i], qubits[right_neighbor]) ** (-beta[index]*2/Pi))

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta

    def circuit_to_latex_using_qcircuit(self):
        """
        Convert the circuit to LaTeX format using QCircuit.
        
        Returns:
            str: LaTeX representation of the circuit.
        """
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )

class AnzatsAFMHeisenbergLattice():
    """
    Class to construct and represent an ansatz for the AFM Heisenberg model on a 2D lattice.
    
    Attributes:
        circuit (cirq.Circuit): The quantum circuit for the ansatz.
        qubits (List[cirq.GridQubit]): List of qubits used in the circuit.
        gamma (np.ndarray): Array of gamma parameters for the circuit.
        beta (np.ndarray): Array of beta parameters for the circuit.
        phi (np.ndarray): Array of phi parameters for the circuit.
    """
    def __init__(self, rows, cols, gamma, beta, phi, periodic=True):
        """
        Initialize the AFM Heisenberg ansatz circuit for a 2D lattice.
        
        Args:
            rows (int): Number of rows in the lattice.
            cols (int): Number of columns in the lattice.
            gamma (np.ndarray): Array of gamma parameters.
            beta (np.ndarray): Array of beta parameters.
            phi (np.ndarray): Array of phi parameters.
            periodic (bool): If True, periodic boundary conditions (PBC) are used; otherwise, open boundary conditions (OBC).
        """
        
        edge = 0 if periodic else 1  # Edge = 0 for PBC and edge = 1 for OBC
        
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = [cirq.GridQubit(row, col) for row in range(rows) for col in range(cols)]  # Create grid of qubits

        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        # Even qubits get H + Y-gates and odd qubits get X-gates, CNOT gates between for correlation
        for i in range(rows):        
            for j in range(0, cols, 2):
                circuit.append(cirq.H(cirq.GridQubit(i, j)))
                circuit.append(cirq.Y(cirq.GridQubit(i, j)))
                circuit.append(cirq.X(cirq.GridQubit(i, (j + 1) % cols)))
                circuit.append(cirq.CNOT(cirq.GridQubit(i, j), cirq.GridQubit(i, (j + 1) % cols)))
        
        # Add correlation gates XX, YY, ZZ
        for index in range(gamma.size):
            # Add gamma circuit
            for i in range(rows):
                for j in range(0, cols, 2):
                    right_neighbor = (j + 1) % cols # Modulus operation sets index = 0 if index > cols, only applicable for PBC
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
            
            # Add beta circuit
            for i in range(rows):
                for j in range(1, cols - edge, 2):
                    right_neighbor = (j + 1) % cols
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))

            # Add phi circuit
            for i in range(0, rows, 2):
                for j in range(cols):
                    bottom_neighbor = (i + 1) % rows
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))

        # Store circuit, qubits, gamma, beta, and phi parameters
        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta
        self.phi = phi

    def circuit_to_latex_using_qcircuit(self):
        """
        Convert the circuit to LaTeX format using QCircuit.
        
        Returns:
            str: LaTeX representation of the circuit.
        """
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )

class AnzatsAFMHeisenbergMatrix():
    """
    Class to construct and represent an ansatz for the AFM Heisenberg model on a matrix (2D lattice).
    
    Attributes:
        circuit (cirq.Circuit): The quantum circuit for the ansatz.
        qubits (List[cirq.LineQubit]): List of qubits used in the circuit.
        gamma (np.ndarray): Array of gamma parameters for the circuit.
        beta (np.ndarray): Array of beta parameters for the circuit.
        phi (np.ndarray): Array of phi parameters for the circuit.
        theta (np.ndarray): Array of theta parameters for the circuit.
    """
    def __init__(self, rows, cols, gamma, beta, phi, theta, periodic=True):
        """
        Initialize the AFM Heisenberg ansatz circuit for a matrix (2D lattice).
        
        Args:
            rows (int): Number of rows in the lattice.
            cols (int): Number of columns in the lattice.
            gamma (np.ndarray): Array of gamma parameters.
            beta (np.ndarray): Array of beta parameters.
            phi (np.ndarray): Array of phi parameters.
            theta (np.ndarray): Array of theta parameters.
            periodic (bool): If True, periodic boundary conditions (PBC) are used; otherwise, open boundary conditions (OBC).
        """
        
        edge = 0 if periodic else 1  # Edge = 0 for PBC and edge = 1 for OBC
        
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(rows * cols)
        
        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        # Even qubits get H + Y-gates and odd qubits get X-gates, CNOT gates between for correlation
        for i in range(rows):        
            for j in range(0, cols, 2):
                circuit.append(cirq.H(cirq.GridQubit(i, j)))
                circuit.append(cirq.Y(cirq.GridQubit(i, j)))
                circuit.append(cirq.X(cirq.GridQubit(i, (j + 1) % cols)))
                circuit.append(cirq.CNOT(cirq.GridQubit(i, j), cirq.GridQubit(i, (j + 1) % cols)))
        
         # Add correlation gates XX, YY, ZZ
        for index in range(gamma.size):
            # Add gamma circuit
            for i in range(rows):
                for j in range(0, cols, 2):
                    right_neighbor = (j + 1) % cols  # Modulus operation sets index = 0 if index > cols, only applicable for PBC
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
            
            # Add beta circuit
            for i in range(rows):
                for j in range(1, cols - edge, 2):
                    right_neighbor = (j + 1) % cols
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))

            # Add phi circuit
            for i in range(0, rows, 2):
                for j in range(cols):
                    bottom_neighbor = (i + 1) % rows
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))

            # Add theta circuit
            for i in range(1, rows - edge, 2):
                for j in range(cols):
                    bottom_neighbor = (i + 1) % rows
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-theta[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-theta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-theta[index] * 2 / Pi))
                
        # Store circuit, qubits, gamma, beta, phi and theta parameters
        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.theta = theta
        
    def circuit_to_latex_using_qcircuit(self):
        """
        Convert the circuit to LaTeX format using QCircuit.
        
        Returns:
            str: LaTeX representation of the circuit.
        """
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )
