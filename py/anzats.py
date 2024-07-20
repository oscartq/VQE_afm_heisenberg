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
                # circuit.append(
                #     cirq.ZZPowGate(
                #         exponent=gamma[index]/Pi, global_shift=0.0).on(
                #             qubits[i], qubits[(i+1)%length]
                #     )
                # )
            # add gates on 2n-1 and 2n
            for i in range(0, length, 2):
                circuit.append(
                    cirq.ZZ(qubits[i], qubits[(i+1)%length]) ** (gamma[index]*2/Pi)
                    )
                # circuit.append(
                #     cirq.ZZPowGate(
                #         exponent=gamma[index]/Pi, global_shift=0.0).on(
                #             qubits[i], qubits[(i+1)%length]
                #         )
                # )

            # add beta circuit
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
    def __init__(self, rows, cols, gamma, beta):
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(rows * cols)
        edge = 0
        
        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        for i in range(0, rows - edge, 2):
            for j in range(0, cols):
                current_index = j * rows + i
                right_neighbor = j * rows+ (i + 1) % rows
                circuit.append(cirq.H(qubits[current_index]))
                circuit.append(cirq.Y(qubits[current_index]))
                circuit.append(cirq.X(qubits[right_neighbor]))
                circuit.append(cirq.CNOT(qubits[current_index], qubits[right_neighbor]))

        # Add gamma and beta operations to the circuit
        for index in range(gamma.size):
            # Add gamma circuit (row-wise interactions)
            for i in range(0, rows - edge, 2):
                for j in range(0, cols):
                    current_index = j * rows + i
                    right_neighbor = j * rows + (i + 1) % rows
                    circuit.append(cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
            
            # Add beta circuit (row-wise interactions)
            for i in range(1, rows - edge, 2):
                for j in range(0, cols):
                    current_index = j * rows + i
                    right_neighbor = j * rows + (i + 1) % rows
                    circuit.append(cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))

            # Add beta circuit (column-wise interactions)
            for i in range(0, rows):
                for j in range(1, cols - edge, 2):
                    current_index = j * rows + i
                    down_neighbor = ((j + 1) % cols) * rows + i
                    circuit.append(cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-beta[index] * 2 / Pi))

        # Store circuit, qubits, gamma, and beta parameters
        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta

    def circuit_to_latex_using_qcircuit(self):
        # Convert the circuit to LaTeX format using QCircuit
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )

class AnzatsAFMHeisenbergLattice_3p():
    def __init__(self, rows, cols, gamma, beta, alpha):
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(rows * cols)
        edge = 0
        
        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        for i in range(0, rows - edge, 2):
            for j in range(0, cols):
                current_index = j * rows + i
                right_neighbor = j * rows+ (i + 1) % rows
                circuit.append(cirq.H(qubits[current_index]))
                circuit.append(cirq.Y(qubits[current_index]))
                circuit.append(cirq.X(qubits[right_neighbor]))
                circuit.append(cirq.CNOT(qubits[current_index], qubits[right_neighbor]))
    
        # Add gamma and beta operations to the circuit
        for index in range(gamma.size):
            # Add gamma circuit (row-wise interactions)
            for i in range(0, rows - edge, 2):
                for j in range(0, cols):
                    current_index = j * rows + i
                    right_neighbor = j * rows + (i + 1) % rows
                    circuit.append(cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
            
            # Add beta circuit (row-wise interactions)
            for i in range(1, rows - edge, 2):
                for j in range(0, cols):
                    current_index = j * rows + i
                    right_neighbor = j * rows + (i + 1) % rows
                    circuit.append(cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))

            # Add beta circuit (column-wise interactions)
            for i in range(0, rows):
                for j in range(1, cols - edge, 2):
                    current_index = j * rows + i
                    down_neighbor = ((j + 1) % cols) * rows + i
                    circuit.append(cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-alpha[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-alpha[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-alpha[index] * 2 / Pi))

        # Store circuit, qubits, gamma, and beta parameters
        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta
        self.alpha = alpha

    def circuit_to_latex_using_qcircuit(self):
        # Convert the circuit to LaTeX format using QCircuit
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )