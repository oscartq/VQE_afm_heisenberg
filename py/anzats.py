import cirq
import openfermion
import numpy as np
import datetime
# from anzats import Anzats
# from expectation import get_expectation_ZiZj, get_expectation_ghz_l4, get_expectation_ghz_l8
# from optimization import get_gradient, optimize_by_gradient_descent

Pi=3.1415
class Anzats():
    def __init__(self, length, gamma, beta):

        # initialize circuit
        circuit = cirq.Circuit()
        qubits = cirq.LineQubit.range(length)

        for i in range(length):
            circuit.append(cirq.H(qubits[i]))

        # add gamma circuit
        for index in range(gamma.size):
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


