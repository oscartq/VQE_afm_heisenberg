import cirq
import qsimcirq

# Define a simple circuit.
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))
print("Circuit:")
print(circuit)
print()

# Simulate the circuit with qsim and return the amplitudes for |00) and |01).
print('Cirq results:')
cirq_simulator = cirq.Simulator()
cirq_results = cirq_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01])
print(cirq_results)
print()

# Simulate the circuit with qsim and return the amplitudes for |00) and |01).
print('qsim results:')
qsim_simulator = qsimcirq.QSimSimulator()
qsim_results = qsim_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01])
print(qsim_results)

