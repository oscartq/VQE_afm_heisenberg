import cirq
import openfermion as of
import numpy as np
import datetime
import matplotlib.pyplot as plt
from numpy import pi as Pi

class Anzats():
    def __init__(self, length, gamma, beta):

        # initialize circuit
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length)

        for i in range(length):
            circuit.append(cirq.H(qubits[i]))

        # add gamma circuit
        for index in range(len(gamma)):
            # add gates on 2n and 2n+1
            for i in range(1, length, 2):
                circuit.append(
                    cirq.ZZ(qubits[i], qubits[(i+1)%length]) ** (gamma[index]*2/Pi)
                    )
                
            for i in range(0, length, 2):
                circuit.append(
                    cirq.ZZ(qubits[i], qubits[(i+1)%length]) ** (gamma[index]*2/Pi)
                    )
                
            for i in range(length):
                circuit.append(
                    cirq.XPowGate(
                        exponent=beta[index]*2/Pi, global_shift=0.0).on(
                        qubits[i])
                )

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta
        
class AnzatsAFMHeisenberg():
    def __init__(self, length, gamma, beta):

        # initialize circuit
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length)

        for i in range(int(length/2)):
            circuit.append(cirq.H(qubits[int(2*i)]))
            circuit.append(cirq.Y(qubits[int(2*i)]))
            circuit.append(cirq.X(qubits[int(2*i+1)]))
            circuit.append(cirq.CNOT(qubits[int(2*i)], qubits[int(2*i+1)]))

        # add gamma circuit
        for index in range(len(gamma)):
            # add gates on 2n and 2n+1
            for i in range(1, length-1, 2):
                circuit.append(
                    cirq.XX(qubits[i], qubits[(i+1)]) ** (-gamma[index]*2/Pi)
                    )
                circuit.append(
                    cirq.YY(qubits[i], qubits[(i+1)]) ** (-gamma[index]*2/Pi)
                    )
                circuit.append(
                    cirq.ZZ(qubits[i], qubits[(i+1)]) ** (-gamma[index]*2/Pi)
                    )

            # add beta circuit
            for i in range(0, length-1, 2):
                circuit.append(
                    cirq.XX(qubits[i], qubits[(i+1)]) ** (-beta[index]*2/Pi)
                    )
                circuit.append(
                    cirq.YY(qubits[i], qubits[(i+1)]) ** (-beta[index]*2/Pi)
                    )
                circuit.append(
                    cirq.ZZ(qubits[i], qubits[(i+1)]) ** (-beta[index]*2/Pi)
                    )

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta

    def circuit_to_latex_using_qcircuit(self):
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )   

class AnzatsAFMHeisenbergLattice():
    def __init__(self, rows, cols, gamma, beta, phi, periodic=True):
        
        edge = 0 if periodic else 1
        
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = [cirq.GridQubit(row, col) for row in range(rows) for col in range(cols)]

        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        for i in range(rows):        
            for j in range(0, cols, 2):
                circuit.append(cirq.H(cirq.GridQubit(i, j)))
                circuit.append(cirq.Y(cirq.GridQubit(i, j)))
                circuit.append(cirq.X(cirq.GridQubit(i, (j + 1) % cols)))
                circuit.append(cirq.CNOT(cirq.GridQubit(i, j), cirq.GridQubit(i, (j + 1) % cols)))
        
        # Add gamma, beta, and phi operations to the circuit
        for index in range(gamma.size):
            # Add gamma circuit
            for i in range(rows):
                for j in range(0, cols, 2):
                    right_neighbor = (j + 1) % cols
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
        # Convert the circuit to LaTeX format using QCircuit
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )

class AnzatsAFMHeisenbergMatrix():
    def __init__(self, rows, cols, gamma, beta, phi, theta, periodic = True):
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(rows * cols)
        
        edge = 1-1 if periodic else 1-0
        
        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        for i in range(rows):        
            for j in range(0, cols, 2):
                circuit.append(cirq.H(cirq.GridQubit(i, j)))
                circuit.append(cirq.Y(cirq.GridQubit(i, j)))
                circuit.append(cirq.X(cirq.GridQubit(i, (j + 1) % cols)))
                circuit.append(cirq.CNOT(cirq.GridQubit(i, j), cirq.GridQubit(i, (j + 1) % cols)))
        
        # Add gamma, beta, and phi operations to the circuit
        for index in range(gamma.size):
            # Add gamma circuit (row-wise interactions)
            for i in range(rows):
                for j in range(0, cols, 2):
                    right_neighbor = (j + 1) % cols
                    #print(f'Correlation between qubit ({i},{j}) and qubit ({i},{right_neighbor})')
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-gamma[index] * 2 / Pi))
            
            # Add beta circuit (row-wise interactions)
            for i in range(rows):
                for j in range(1, cols - edge, 2):
                    right_neighbor = (j + 1) % cols
                    #print(f'{right_neighbor}')
                    #print(f'Correlation between qubit ({i},{j}) and qubit ({i},{right_neighbor})')
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(i, right_neighbor)) ** (-beta[index] * 2 / Pi))

            # Add phi circuit (column-wise interactions)
            for i in range(0, rows, 2):
                for j in range(cols):
                    bottom_neighbor = (i + 1) % rows
                    #print(f'Correlation between qubit ({i},{j}) and qubit ({bottom_neighbor},{j})')
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(bottom_neighbor, j)) ** (-phi[index] * 2 / Pi))

            # Add theta circuit (column-wise interactions)
            for i in range(1, rows - edge, 2):
                for j in range(cols):
                    bottom_neighbor = (i + 1) % rows
                    #print(f'{right_neighbor}')
                    #print(f'Correlation between qubit ({i},{j}) and qubit ({i},{right_neighbor})')
                    circuit.append(cirq.XX(cirq.GridQubit(i, j), cirq.GridQubit(i, bottom_neighbor)) ** (-theta[index] * 2 / Pi))
                    circuit.append(cirq.YY(cirq.GridQubit(i, j), cirq.GridQubit(i, bottom_neighbor)) ** (-theta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(cirq.GridQubit(i, j), cirq.GridQubit(i, bottom_neighbor)) ** (-theta[index] * 2 / Pi))   
                    
        # Store circuit, qubits, gamma, beta, phi and theta parameters
        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta
        self.phi = phi
        self.theta = theta
        
    def circuit_to_latex_using_qcircuit(self):
        # Convert the circuit to LaTeX format using QCircuit
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )