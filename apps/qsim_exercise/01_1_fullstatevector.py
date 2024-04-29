import time
import cirq
import qsimcirq

# Define qubits and a short circuit.
q0, q1 = cirq.LineQubit.range(2)
circuit = cirq.Circuit(cirq.H(q0), cirq.CX(q0, q1))
print("Circuit:")
print(circuit)
print()

# Simulate the circuit with Cirq and return the full state vector.
print('Cirq results:')
start_time = time.time()
cirq_simulator = cirq.Simulator()
cirq_results = cirq_simulator.simulate(circuit)
end_time = time.time()
print(cirq_results)
print(f"Cirq Time: {end_time - start_time} seconds")
print()

# Simulate the circuit with qsim and return the full state vector.
print('qsim results:')
start_time = time.time()
qsim_simulator = qsimcirq.QSimSimulator()
qsim_results = qsim_simulator.simulate(circuit)
end_time = time.time()
print(qsim_results)
print(f"qsim Time: {end_time - start_time} seconds")

samples = cirq.sample_state_vector(
    qsim_results.state_vector(), indices=[0, 1], repetitions=10)
print(samples)
