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
            print(index)
            #print(f'gamma: {gamma.size}')
            # add gates on columns: i=2n+1, 2n,...
            # print(f'i: {range(1, cols-1, 2)}')
            # print(f'j: {range(0, rows-1)}')
            for i in range(1, cols-1, 2):
                print(f'i: {i}')
                for j in range(0, rows-1):
                    print(f'j: {i}')
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