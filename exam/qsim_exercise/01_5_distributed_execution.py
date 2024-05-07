import cirq
import qsimcirq

# Pick a pair of qubits.
q0 = cirq.GridQubit(0, 0)
q1 = cirq.GridQubit(0, 1)

# Create a circuit that entangles the pair.
circuit = cirq.Circuit(
    cirq.H(q0), cirq.CX(q0, q1), cirq.X(q1)
)
print("Circuit:")
print(circuit)

options = {}

# 'k' indicates the qubits on one side of the cut.
# We'll use qubit 0 for this.
options['k'] = [0]

# 'p' and 'r' control when values are assigned to cut indices.
# There are some intricacies in choosing values for these options,
# but for now we'll set p=1 and r=0.
# This allows us to pre-assign the value of the CX indices
# and distribute its execution to multiple jobs.
options['p'] = 1
options['r'] = 0

# 'w' indicates the value pre-assigned to the cut.
# This should change for each execution.
options['w'] = 0

# Create the qsimh simulator with those options.
qsimh_simulator = qsimcirq.QSimhSimulator(options)
results_0 = qsimh_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01, 0b10, 0b11])
print(results_0)

options['w'] = 1

qsimh_simulator = qsimcirq.QSimhSimulator(options)
results_1 = qsimh_simulator.compute_amplitudes(
    circuit, bitstrings=[0b00, 0b01, 0b10, 0b11])
print(results_1)

results = [r0 + r1 for r0, r1 in zip(results_0, results_1)]
print("qsimh results:")
print(results)

qsim_simulator = qsimcirq.QSimSimulator()
qsim_simulator.compute_amplitudes(circuit, bitstrings=[0b00, 0b01, 0b10, 0b11])
print("qsim results:")
print(results)
