import cirq
import openfermion as of
import numpy as np
import datetime
from inner_functions import _get_one_body_term_on_hubbard, _get_two_body_term_on_hubbard, _exponentiate_quad_ham
import matplotlib.pyplot as plt
# from anzats import Anzats
# from expectation import get_expectation_ZiZj, get_expectation_ghz_l4, get_expectation_ghz_l8
# from optimization import get_gradient, optimize_by_gradient_descent

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
        
        # initialize circuit
        circuit = cirq.Circuit()
        qubits = [cirq.GridQubit(i, j) for i in range(rows) for j in range(cols)]
        #Symmetry cols=4, rows=2
        #q(0,0) q(0,1) q(0,2) q(0,3)
        #q(1,0) q(1,1) q(1,2) q(1,3)
        
        L=int(cols)
        for i in range(int(cols/2)):
            #First row
            circuit.append(cirq.H(qubits[int(2*i)]))
            circuit.append(cirq.Y(qubits[int(2*i)]))
            circuit.append(cirq.X(qubits[int(2*i+1)]))
            
            #Second row
            circuit.append(cirq.H(qubits[int(2*i)+L]))
            circuit.append(cirq.Y(qubits[int(2*i)+L]))
            circuit.append(cirq.X(qubits[int(2*i+1)+L]))
            
            #Correlation                                                        #i=0 , i=1
            circuit.append(cirq.CNOT(qubits[int(2*i)], qubits[int(2*i+1)]))     #0-1 , 2-3
            circuit.append(cirq.CNOT(qubits[int(2*i)], qubits[int(2*i)+L]))     #0-4 , 2-6
            circuit.append(cirq.CNOT(qubits[int(2*i)+L], qubits[int(2*i+1)+L])) #5-4 , 6-7
            circuit.append(cirq.CNOT(qubits[int(2*i+1)], qubits[int(2*i+1)+L])) #1-5 , 3-7
        
        for index in range(gamma.size):
            # add gates on columns: i=2n+1, 2n,...
            for i in range(1, cols-1, 2):
                for j in range(0, rows-1):
                    current_index = j*cols + i
                    right_neighbor = j * cols + (i + 1) % cols

                    # add gamma circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
                        )

                    # add beta circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    
            # add gates on columuns: i=2n, 2n+1
            for i in range(0, cols-1, 2):
                for j in range(0, rows-1):
                    current_index = j*cols + i
                    right_neighbor = j * cols + (i + 1) % cols

                    # add gamma circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-gamma[index]*2/Pi)
                        )

                    # add beta circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[right_neighbor]) ** (-beta[index]*2/Pi)
                        )
            # add gates on columuns: j=2n+1, 2n,...
            for i in range(0, cols):
                for j in range(1, rows-1, 2):
                    current_index = j*cols + i
                    down_neighbor = ((j + 1) % cols) * cols + i 

                    # add gamma circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-gamma[index]*2/Pi)
                        )

                    # add beta circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    
            # add gates on rows: j=2n, 2n+1
            for i in range(0, cols):
                for j in range(0, rows-1, 2):
                    current_index = j*cols + i
                    down_neighbor = ((j + 1) % rows) * cols + i 

                    # add gamma circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-gamma[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-gamma[index]*2/Pi)
                        )

                    # add beta circuit
                    circuit.append(
                        cirq.XX(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.YY(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
                        )
                    circuit.append(
                        cirq.ZZ(qubits[current_index], qubits[down_neighbor]) ** (-beta[index]*2/Pi)
                        )

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta

    def circuit_to_latex_using_qcircuit(self):
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )   
        
class AnzatsAFMHeisenberg_line_2d():
    def __init__(self, length, gamma, beta):

        # initialize circuit
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length)

        L=int(length/2)
        for i in range(int(length/4)):
            #First row
            circuit.append(cirq.H(qubits[int(2*i)]))
            circuit.append(cirq.Y(qubits[int(2*i)]))
            circuit.append(cirq.X(qubits[int(2*i+1)]))
            
            #Second row
            circuit.append(cirq.H(qubits[int(2*i)+L]))
            circuit.append(cirq.Y(qubits[int(2*i)+L]))
            circuit.append(cirq.X(qubits[int(2*i+1)+L]))
            
            #Correlation                                                        #i=0 , i=1
            circuit.append(cirq.CNOT(qubits[int(2*i)], qubits[int(2*i+1)]))     #0-1 , 2-3
            circuit.append(cirq.CNOT(qubits[int(2*i)], qubits[int(2*i)+L]))     #0-4 , 2-6
            circuit.append(cirq.CNOT(qubits[int(2*i)+L], qubits[int(2*i+1)+L])) #5-4 , 6-7
            circuit.append(cirq.CNOT(qubits[int(2*i+1)], qubits[int(2*i+1)+L])) #1-5 , 3-7

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
            # add gates on 2n-1 and 2n

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
        
class AnzatsAFMHeisenberg_2d:
    def __init__(self, length, gamma, beta):
        rows = 2
        cols = int(length / 2)
        if rows * cols % 2 != 0:
            raise ValueError("The number of qubits (rows * cols) must be even.")

        # Initialize circuit
        circuit = cirq.Circuit()
        qubits = [cirq.GridQubit(i, j) for i in range(rows) for j in range(cols)]

        # Apply initial gates and CNOT gates for correlations
        for i in range(rows):
            for j in range(cols):
                index = i * cols + j
                if index % 2 == 0:  # Even qubits
                    circuit.append(cirq.H(qubits[index]))
                    circuit.append(cirq.Y(qubits[index]))
                else:  # Odd qubits
                    circuit.append(cirq.X(qubits[index]))

                # Apply CNOT gates for correlations with neighbors
                if j + 1 < cols:  # Right neighbor
                    circuit.append(cirq.CNOT(qubits[index], qubits[index + 1]))
                if i + 1 < rows:  # Down neighbor
                    circuit.append(cirq.CNOT(qubits[index], qubits[index + cols]))
                if j - 1 >= 0:  # Left neighbor
                    circuit.append(cirq.CNOT(qubits[index], qubits[index - 1]))
                if i - 1 >= 0:  # Up neighbor
                    circuit.append(cirq.CNOT(qubits[index], qubits[index - cols]))

        # Add gamma circuit
        for index in range(len(gamma)):
            for i in range(rows):
                for j in range(cols - 1):
                    # Apply XX, YY, ZZ gates to horizontal neighbors
                    circuit.append(
                        cirq.XX(qubits[i * cols + j], qubits[i * cols + j + 1]) ** (-gamma[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.YY(qubits[i * cols + j], qubits[i * cols + j + 1]) ** (-gamma[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.ZZ(qubits[i * cols + j], qubits[i * cols + j + 1]) ** (-gamma[index] * 2 / Pi)
                    )

            for j in range(cols):
                for i in range(rows - 1):
                    # Apply XX, YY, ZZ gates to vertical neighbors
                    circuit.append(
                        cirq.XX(qubits[i * cols + j], qubits[(i + 1) * cols + j]) ** (-gamma[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.YY(qubits[i * cols + j], qubits[(i + 1) * cols + j]) ** (-gamma[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.ZZ(qubits[i * cols + j], qubits[(i + 1) * cols + j]) ** (-gamma[index] * 2 / Pi)
                    )

            # Add beta circuit
            for i in range(rows):
                for j in range(cols - 1):
                    # Apply XX, YY, ZZ gates to horizontal neighbors
                    circuit.append(
                        cirq.XX(qubits[i * cols + j], qubits[i * cols + j + 1]) ** (-beta[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.YY(qubits[i * cols + j], qubits[i * cols + j + 1]) ** (-beta[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.ZZ(qubits[i * cols + j], qubits[i * cols + j + 1]) ** (-beta[index] * 2 / Pi)
                    )

            for j in range(cols):
                for i in range(rows - 1):
                    # Apply XX, YY, ZZ gates to vertical neighbors
                    circuit.append(
                        cirq.XX(qubits[i * cols + j], qubits[(i + 1) * cols + j]) ** (-beta[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.YY(qubits[i * cols + j], qubits[(i + 1) * cols + j]) ** (-beta[index] * 2 / Pi)
                    )
                    circuit.append(
                        cirq.ZZ(qubits[i * cols + j], qubits[(i + 1) * cols + j]) ** (-beta[index] * 2 / Pi)
                    )

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta
        
        # Function to visualize the circuit using matplotlib
        # def visualize_circuit(circuit):
        #     diagram = cirq.CircuitDiagramInfo(circuit)
        #     fig, ax = plt.subplots(figsize=(12, 6))
        #     cirq.circuits.circuit_diagram.draw_circuit_diagram(ax, diagram)
        #     plt.show()

        # # Visualize the circuit
        # visualize_circuit(ansatz.circuit) 
           
    def circuit_to_latex_using_qcircuit(self):
        return cirq.contrib.circuit_to_latex_using_qcircuit(
            self.circuit, self.qubits
        )
    
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
          
class AnzatsAFMHeisenberg_new_symmetry():
    def __init__(self, length, gamma, beta):

        # initialize circuit
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length)

        for i in range(int(length/2)):
            circuit.append(cirq.H(qubits[int(2*i)]))
            circuit.append(cirq.Y(qubits[int(2*i)]))
            circuit.append(cirq.X(qubits[int(2*i+2)]))
            circuit.append(cirq.CNOT(qubits[int(2*i)], qubits[int(2*i+2)]))

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
            # add gates on 2n-1 and 2n

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
        
# class AnzatsAFMHeisenberg_new_symmetry():
#     def __init__(self, length, gamma, beta):

#         # initialize circuit
#         circuit = cirq.Circuit()
#         qubits = cirq.LineQubit.range(length)
        
#         for i in range(int(length/2)):
#             circuit.append(cirq.H(qubits[int(i)]))
#             circuit.append(cirq.Y(qubits[int(i)]))
#             circuit.append(cirq.X(qubits[int(i+2)]))
#             circuit.append(cirq.CNOT(qubits[int(i)], qubits[int(i+2)])) #Correlation between qubits 2*i and 2*i+2 instead of 2*i+1

#         # add gamma circuit
#         for index in range(len(gamma)):
#             # add gates on n+1 and n+2
#             for i in range(0, int(length/2), 1):
#                 circuit.append(
#                     cirq.XX(qubits[i+1], qubits[(i+2)]) ** (-gamma[index]*2/Pi)
#                     )
#                 circuit.append(
#                     cirq.YY(qubits[i+1], qubits[(i+2)]) ** (-gamma[index]*2/Pi)
#                     )
#                 circuit.append(
#                     cirq.ZZ(qubits[i+1], qubits[(i+2)]) ** (-gamma[index]*2/Pi)
#                     )
                
#             # add gates on n and n+2
#             # add beta circuit
#             for i in range(0, int(length/2), 1):
#                 circuit.append(
#                     cirq.XX(qubits[i], qubits[(i+2)]) ** (-beta[index]*2/Pi)
#                     )
#                 circuit.append(
#                     cirq.YY(qubits[i], qubits[(i+2)]) ** (-beta[index]*2/Pi)
#                     )
#                 circuit.append(
#                     cirq.ZZ(qubits[i], qubits[(i+2)]) ** (-beta[index]*2/Pi)
#                     )

#         self.circuit = circuit
#         self.qubits = qubits
#         self.gamma = gamma
#         self.beta = beta

#     def circuit_to_latex_using_qcircuit(self):
#         return cirq.contrib.circuit_to_latex_using_qcircuit(
#             self.circuit, self.qubits
#         )   

class AnzatsToricCode():
    def __init__(self, length, gamma, beta):

        # initialize circuit
        circuit = cirq.Circuit()
        qubits = [[cirq.GridQubit(i, j) for i in range(length)] for j in range(length)]

        for i in range(length):
            for j in range(length):
                circuit.append(cirq.H(qubits[i][j]))

        # add gamma circuit
        for index in range(len(gamma)):
            # add gates on 2n and 2n+1
            for i in range(length):
                for j in range(length):
                    circuit.append(cirq.X(qubits[i][(j+1)%length]) ** (gamma[index]*2/Pi))
                    circuit.append(cirq.Y(qubits[(i+1)%length][(j+1)%length]) ** (gamma[index]*2/Pi))
                    circuit.append(cirq.X(qubits[(i+1)%length][j]) ** (gamma[index]*2/Pi))
                    circuit.append(cirq.Y(qubits[i][j]) ** (gamma[index]*2/Pi))

            # add beta circuit
            for i in range(length):
                for j in range(length):
                    circuit.append(
                        cirq.XPowGate(
                            exponent=beta[index]*2/Pi, global_shift=0.0).on(
                            qubits[i][j])
                    )

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta


class AnzatsBCSHubbard():
    def __init__(self, function_args, gamma, beta):
        # initialize circuit
        x = function_args.x_dimension
        y = function_args.y_dimension
        n_sites = function_args.n_sites
        tunneling=function_args.tunneling
        hopping_matrix=function_args.hopping_matrix
        coulomb=function_args.coulomb
        periodic=function_args.periodic
        sc_gap=function_args.sc_gap
        n_qubits = function_args.n_qubits
        qubits = cirq.LineQubit.range(n_qubits)
        
        # |\psi 1> circuit
        mean_field_model = of.mean_field_dwave(
            x, y, tunneling, sc_gap, periodic)
        quad_ham = of.get_quadratic_hamiltonian(mean_field_model)
        
        circuit = cirq.Circuit(
            of.circuits.prepare_gaussian_state(qubits, quad_ham))

        for index in range(len(gamma)):
            # H2 (gamma) hubbard
            # circuit tunneling
            for term in _get_one_body_term_on_hubbard(hopping_matrix, n_sites):
                term_qubit = of.qubit_operator_to_pauli_sum(of.jordan_wigner(term))
                circuit.append(cirq.PauliSumExponential(
                    term_qubit,
                    exponent=gamma[index],
                    atol = 1e-08
                ))

            # circuit coulomb
            for term in _get_two_body_term_on_hubbard(coulomb, n_sites, n_qubits):
                term_qubit = of.qubit_operator_to_pauli_sum(of.jordan_wigner(term))
                circuit.append(cirq.PauliSumExponential(
                    term_qubit,
                    exponent=gamma[index],
                    atol = 1e-08
                ))

            # H1 (beta) bcs 
            circuit.append(
                cirq.Circuit(
                    _exponentiate_quad_ham(qubits, quad_ham, beta[index]
            )))

        self.circuit = circuit
        self.qubits = qubits
        self.gamma = gamma
        self.beta = beta

