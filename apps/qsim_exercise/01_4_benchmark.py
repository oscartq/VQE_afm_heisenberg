import cirq
import qsimcirq

import time

# Get a rectangular grid of qubits.
qubits = cirq.GridQubit.rect(4, 5)

# Generates a random circuit on the provided qubits.
circuit = cirq.experiments.random_rotations_between_grid_interaction_layers_circuit(
    qubits=qubits, depth=20)

# Simulate the circuit with Cirq and print the runtime.
cirq_simulator = cirq.Simulator()
cirq_start = time.time()
cirq_results = cirq_simulator.simulate(circuit)
cirq_elapsed = time.time() - cirq_start
print(f'Cirq runtime: {cirq_elapsed} seconds.')
print()

# Simulate the circuit with qsim and print the runtime.
qsim_simulator = qsimcirq.QSimSimulator()
qsim_start = time.time()
qsim_results = qsim_simulator.simulate(circuit)
qsim_elapsed = time.time() - qsim_start
print(f'qsim runtime: {qsim_elapsed} seconds.')

# Use eight threads to parallelize simulation.
options = {'t': 15} # more or less, it takes longer time

qsim_simulator = qsimcirq.QSimSimulator(options)
qsim_start = time.time()
qsim_results = qsim_simulator.simulate(circuit)
qsim_elapsed = time.time() - qsim_start
print(f'qsim runtime ({options}): {qsim_elapsed} seconds.')

# Increase maximum fused gate size to three qubits.
options = {'f': 3}

qsim_simulator = qsimcirq.QSimSimulator(options)
qsim_start = time.time()
qsim_results = qsim_simulator.simulate(circuit)
qsim_elapsed = time.time() - qsim_start
print(f'qsim runtime ({options}): {qsim_elapsed} seconds.')


