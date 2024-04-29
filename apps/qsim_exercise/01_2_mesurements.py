import cirq
import qsimcirq

# Define a circuit with measurements.
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(
    cirq.H(q0), cirq.X(q1), cirq.CX(q0, q1),
    cirq.measure(q0, key='qubit_0'),
    cirq.measure(q1, key='qubit_1'),
)
print("Circuit:")
print(circuit)
print()

# Simulate the circuit with Cirq and return just the measurement values.
print('Cirq results:')
cirq_simulator = cirq.Simulator()
cirq_results = cirq_simulator.run(circuit, repetitions=5)
print(cirq_results)
print()

# Simulate the circuit with qsim and return just the measurement values.
print('qsim results:')
qsim_simulator = qsimcirq.QSimSimulator()
qsim_results = qsim_simulator.run(circuit, repetitions=5)
print(qsim_results)

# Define a circuit with intermediate measurements.
q0 = cirq.LineQubit(0)
circuit = cirq.Circuit(
    cirq.X(q0)**0.5, cirq.measure(q0, key='m0'),
    cirq.X(q0)**0.5, cirq.measure(q0, key='m1'),
    cirq.X(q0)**0.5, cirq.measure(q0, key='m2'),
)
print("Circuit:")
print(circuit)
print()

# Simulate the circuit with qsim and return just the measurement values.
print('qsim results:')
qsim_simulator = qsimcirq.QSimSimulator()
qsim_results = qsim_simulator.run(circuit, repetitions=5)
print(qsim_results)