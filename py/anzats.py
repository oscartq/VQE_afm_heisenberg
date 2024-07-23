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
    def __init__(self, rows, cols, gamma, beta, periodic = True):
             
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(rows * cols)
        
        edge = 1-1 if periodic else 1-0
        
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
    def __init__(self, rows, cols, gamma, beta, phi, periodic = True):
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(rows * cols)
        edge = 1-1 if periodic else 1-0
        
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

            # Add phi circuit (column-wise interactions)
            for i in range(0, rows):
                for j in range(1, cols - edge, 2):
                    current_index = j * rows + i
                    down_neighbor = ((j + 1) % cols) * rows + i
                    circuit.append(cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-phi[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-phi[index] * 2 / Pi))

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
        
# class AnzatsAFMHeisenbergLattice():
#     def __init__(self, function_args, gamma, beta):
#         length = function_args.length
#         width = function_args.width
#         periodic= function_args.periodic

#         # initialize circuit
#         circuit = cirq.Circuit()
#         qubits = cirq.LineQubit.range(length*width)
#         edge   = 0 if periodic else 1
#         for i in range(0, length, 2):
#             for j in range(0, width):
#                 current_index = j * length + i
#                 right_neighbor = j * length + (i + 1) % length
#                 circuit.append(cirq.H(qubits[current_index]))
#                 circuit.append(cirq.Y(qubits[current_index]))
#                 circuit.append(cirq.X(qubits[right_neighbor]))
#                 circuit.append(cirq.CNOT(qubits[current_index], qubits[right_neighbor]))

#         for index in range(gamma.size):
#             # gamma
#             for i in range(0, length, 2):
#                 for j in range(0, width):
#                     current_index = j * length + i
#                     right_neighbor = j * length + (i + 1) % length

#                     # add gamma circuit
#                     circuit.append(
#                         cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
#                         )
#             # beta (row)
#             for i in range(1, length-edge, 2):
#                 for j in range(0, width):
#                     current_index = j * length + i
#                     right_neighbor = j * length + (i + 1) % length

#                     circuit.append(
#                         cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
#                         )

#             # beta (column)
#             for i in range(0, length):
#                 for j in range(0, width, 2):
#                     current_index = j * length + i
#                     down_neighbor = ((j + 1) % width) * length + i 

#                     circuit.append(
#                         cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
#                         )
#                 for j in range(1, width-edge, 2):
#                     current_index = j * length + i
#                     down_neighbor = ((j + 1) % width) * length + i 

#                     circuit.append(
#                         cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
#                         )
#                     circuit.append(
#                         cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
#                         )

#         self.circuit = circuit
#         self.qubits = qubits
#         self.gamma = gamma
#         self.beta = beta

#     def circuit_to_latex_using_qcircuit(self):
#         return cirq.contrib.circuit_to_latex_using_qcircuit(
#             self.circuit, self.qubits
#         )   