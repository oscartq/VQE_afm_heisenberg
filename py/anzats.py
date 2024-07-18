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
    def __init__(self, length, width, gamma, beta):
        # Initialize circuit and qubits
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length * width)
        edge = 0
        
        # Create the initial circuit with Hadamard, Y, X, and CNOT gates
        for i in range(0, length - edge, 2):
            for j in range(0, width):
                current_index = j * length + i
                right_neighbor = j * length + (i + 1) % length
                circuit.append(cirq.H(qubits[current_index]))
                circuit.append(cirq.Y(qubits[current_index]))
                circuit.append(cirq.X(qubits[right_neighbor]))
                circuit.append(cirq.CNOT(qubits[current_index], qubits[right_neighbor]))

        # Add gamma and beta operations to the circuit
        for index in range(gamma.size):
            # Add gamma circuit (row-wise interactions)
            for i in range(0, length - edge, 2):
                for j in range(0, width):
                    current_index = j * length + i
                    right_neighbor = j * length + (i + 1) % length
                    circuit.append(cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index] * 2 / Pi))
            
            # Add beta circuit (row-wise interactions)
            for i in range(1, length - edge, 2):
                for j in range(0, width):
                    current_index = j * length + i
                    right_neighbor = j * length + (i + 1) % length
                    circuit.append(cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))
                    circuit.append(cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-beta[index] * 2 / Pi))

            # Add beta circuit (column-wise interactions)
            for i in range(0, length):
                for j in range(1, width - edge, 2):
                    current_index = j * length + i
                    down_neighbor = ((j + 1) % width) * length + i
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
        
# class AnzatsAFMHeisenbergLattice():
#     def __init__(self, rows, cols, gamma, beta):
#         # Initialize circuit
#         circuit = cirq.Circuit()
#         qubits = [cirq.GridQubit(i, j) for i in range(rows) for j in range(cols)]
        
#         # Symmetry cols=4, rows=2
#         # q(0,0) q(0,1) q(0,2) q(0,3)
#         # q(1,0) q(1,1) q(1,2) q(1,3)
       
#         for i in range(rows):
#             for j in range(cols):
#                 if (i * cols + j) % 2 == 0:  # Even qubits
#                     circuit.append(cirq.H(qubits[i * cols + j]))
#                     circuit.append(cirq.Y(qubits[i * cols + j]))
#                 else:  # Odd qubits
#                     circuit.append(cirq.X(qubits[i * cols + j]))
                    
#         for i in range(cols // 2):             
#             #Correlation
#             circuit.append(cirq.CNOT(qubits[2*i], qubits[2*i + 1]))
#             circuit.append(cirq.CNOT(qubits[2*i], qubits[2*i + cols]))
#             circuit.append(cirq.CNOT(qubits[2*i + cols], qubits[2*i + 1 + cols]))
#             circuit.append(cirq.CNOT(qubits[2*i + 1], qubits[2*i + 1 + cols]))
                    
#         def add_correlation_gates(circuit, qubit1, qubit2, gamma, beta):
#             circuit.append(cirq.XX(qubit1, qubit2) ** (-gamma * 2 / Pi))
#             circuit.append(cirq.YY(qubit1, qubit2) ** (-gamma * 2 / Pi))
#             circuit.append(cirq.ZZ(qubit1, qubit2) ** (-gamma * 2 / Pi))
#             circuit.append(cirq.XX(qubit1, qubit2) ** (-beta * 2 / Pi))
#             circuit.append(cirq.YY(qubit1, qubit2) ** (-beta * 2 / Pi))
#             circuit.append(cirq.ZZ(qubit1, qubit2) ** (-beta * 2 / Pi))
        
#         for index in range(gamma.size):
#             #print('Start')
#             # Add gates between qubits q(1) and q(2)
#             #print('1--------------------------------------------------------------------------')
#             for i in range(1, cols, 2):  # Odd qubits
#                 for j in range(rows - 1):  # First row
#                     current_index = j * cols + i                    
#                     right_neighbor = j * cols + (i + 1) % cols
#                     add_correlation_gates(circuit, qubits[current_index], qubits[right_neighbor], gamma[index], beta[index])
#                     #print(f'Adding correlation between qubits {current_index} and {right_neighbor}...')

#             # Add gates between qubits q(0) and q(1) + q(2) and q(3)
#             #print('2--------------------------------------------------------------------------')
#             for i in range(0, cols, 2):
#                 for j in range(rows - 1):
#                     current_index = j * cols + i
#                     right_neighbor = j * cols + (i + 1) % cols
#                     add_correlation_gates(circuit, qubits[current_index], qubits[right_neighbor], gamma[index], beta[index])
#                     #print(f'Adding correlation between qubits {current_index} and {right_neighbor}...')
                    
#             # Add gates between qubits q(5) and q(6)
#             #print('3--------------------------------------------------------------------------')
#             for i in range(1, cols, 2):  # Odd qubits
#                 for j in range(rows - 1):  # Number of rows -1
#                     current_index = j * cols + i + cols                   
#                     right_neighbor = j * cols + (i + 1) % cols + cols
#                     add_correlation_gates(circuit, qubits[current_index], qubits[right_neighbor], gamma[index], beta[index])
#                     #print(f'Adding correlation between qubits {current_index} and {right_neighbor}...')
                    
#             # Add gates between qubits q(4) and q(5) + q(6) and q(7)
#             #print('4--------------------------------------------------------------------------')
#             for i in range(0, cols, 2):
#                 for j in range(rows - 1):
#                     current_index = j * cols + i + cols
#                     right_neighbor = j * cols + (i + 1) % cols + cols
#                     add_correlation_gates(circuit, qubits[current_index], qubits[right_neighbor], gamma[index], beta[index])
#                     #print(f'Adding correlation between qubits {current_index} and {right_neighbor}...')
                    
#             # Add gates between qubits q(0) and q(4) + q(1) and q(5) + q(2) and q(6) + q(3) and q(7)
#             #print('5--------------------------------------------------------------------------')
#             for i in range(cols):
#                 for j in range(0, rows, 2):
#                     current_index = j * cols + i
#                     down_neighbor = ((j + 1) % rows) * cols + i 
#                     add_correlation_gates(circuit, qubits[current_index], qubits[down_neighbor], gamma[index], beta[index])
#                     #print(f'Adding correlation between qubits {current_index} and {down_neighbor}...')
#             #print('End')
#         self.circuit = circuit
#         self.qubits = qubits
#         self.gamma = gamma
#         self.beta = beta

#     def circuit_to_latex_using_qcircuit(self):
#         return cirq.contrib.circuit_to_latex_using_qcircuit(
#             self.circuit, self.qubits
#         )
        

class AnzatsAFMHeisenberg_periodic():
    def __init__(self, length, gamma, beta):

        # initialize circuit
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length)


        for i in range(int(length/2)):
            circuit.append(cirq.H(qubits[int(2*i)%length]))
            circuit.append(cirq.Y(qubits[int(2*i)%length]))
            circuit.append(cirq.X(qubits[int(2*i+1)%length]))
            circuit.append(cirq.CNOT(qubits[int(2*i)%length], qubits[int(2*i+1)%length]))

        # add gamma circuit
        for index in range(len(gamma)):
            # add gates on 2n and 2n+1
            for i in range(1, length-1, 2):
                circuit.append(
                    cirq.XX(qubits[i], qubits[(i+1)%length]) ** (-gamma[index]*2/Pi)
                    )
                circuit.append(
                    cirq.YY(qubits[i], qubits[(i+1)%length]) ** (-gamma[index]*2/Pi)
                    )
                circuit.append(
                    cirq.ZZ(qubits[i], qubits[(i+1)%length]) ** (-gamma[index]*2/Pi)
                    )
            # add gates on 2n-1 and 2n

            # add beta circuit
            for i in range(0, length-1, 2):
                circuit.append(
                    cirq.XX(qubits[i], qubits[(i+1)%length]) ** (-beta[index]*2/Pi)
                    )
                circuit.append(
                    cirq.YY(qubits[i], qubits[(i+1)%length]) ** (-beta[index]*2/Pi)
                    )
                circuit.append(
                    cirq.ZZ(qubits[i], qubits[(i+1)%length]) ** (-beta[index]*2/Pi)
                    )

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta

    def circuit_to_latex_using_qcircuit(self):
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        ) 