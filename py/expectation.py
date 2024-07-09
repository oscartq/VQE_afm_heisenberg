import cirq
import openfermion as of
import numpy as np
import datetime
from anzats import Anzats, AnzatsAFMHeisenberg, AnzatsAFMHeisenberg_periodic, AnzatsAFMHeisenberg_2d, AnzatsAFMHeisenberg_new_symmetry, AnzatsToricCode, AnzatsBCSHubbard
import qsimcirq
# import cupy as cp
# from expectation import get_expectation_ZiZj, get_expectation_ghz_l4, get_expectation_ghz_l8
# from optimization import get_gradient, optimize_by_gradient_descent
from inner_functions import _get_one_body_term_on_hubbard, _get_two_body_term_on_hubbard, _exponentiate_quad_ham

class GHZStateArgs():
    def __init__(self, length, qsim_option=None):
        self.length = length
        self.qsim_option = qsim_option

def get_expectation_ghz(function_args, gamma, beta):
    length = function_args.length
    anzats = Anzats(length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j
    for i in range(length):
        circuit = anzats.circuit.copy()
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(cirq.Z(qubits[(i+1)%length]))
        vector2 = simulator.simulate(circuit).state_vector()
        value -= np.dot(vector2.conj(), vector)
    return value

def get_expectation_ghz_l4(gamma: np.array, beta: np.array):
    function_args = GHZStateArgs(4)
    return get_expectation_ghz(function_args, gamma, beta)

def get_expectation_ghz_l8(gamma: np.array, beta: np.array):
    function_args = GHZStateArgs(8)
    return get_expectation_ghz(function_args, gamma, beta)

def get_expectation_ghz_l10(gamma: np.array, beta: np.array):
    function_args = GHZStateArgs(10)
    return get_expectation_ghz(function_args, gamma, beta)

def get_expectation_ghz_l12(gamma: np.array, beta: np.array):
    function_args = GHZStateArgs(12)
    return get_expectation_ghz(function_args, gamma, beta)

def get_expectation_ghz_l14(gamma: np.array, beta: np.array):
    function_args = GHZStateArgs(14)
    return get_expectation_ghz(function_args, gamma, beta)

def get_expectation_ghz_l16(gamma: np.array, beta: np.array):
    function_args = GHZStateArgs(16)
    return get_expectation_ghz(function_args, gamma, beta)

class TFIMStateArgs():
    def __init__(self, length, g, qsim_option=None):
        self.length = length
        self.g = g
        self.qsim_option = qsim_option

def get_expectation_critical_state(function_args, gamma, beta):
    anzats = Anzats(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j
    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        circuit.append(cirq.Z(qubits[i]))
        circuit.append(cirq.Z(qubits[(i+1)%function_args.length]))
        vector2 = simulator.simulate(circuit).state_vector()
        value -= np.dot(vector2.conj(), vector)
    for i in range(function_args.length):
        circuit = anzats.circuit.copy()
        # circuit.append(function_args.g * cirq.X(qubits[i]))
        circuit.append(cirq.XPowGate(exponent=function_args.g).on(qubits[i]))
        vector2 = simulator.simulate(circuit).state_vector()
        value -= np.dot(vector2.conj(), vector)
    return value

class AFMHeisenbergArgs():
    def __init__(self, length, qsim_option):
        self.length = length
        self.qsim_option = qsim_option


def get_expectation_afm_heisenberg(function_args, gamma, beta):
    # This function calculates the expectation value for the AFM Heisenberg model using a quantum circuit ansatz.

    # Initialize the ansatz for the AFM Heisenberg model with given parameters
    anzats = AnzatsAFMHeisenberg(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits

    # Initialize the quantum simulator
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)

    # Simulate the circuit to get the initial state vector
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j  # Initialize the expectation value as a complex number

    # Loop through each pair of neighboring qubits
    for i in range(function_args.length - 1):
        print(f'Iteration {i}')
        # Create copies of the original circuit for each Pauli operator (X, Y, Z)
        circuitX = anzats.circuit.copy()
        circuitY = anzats.circuit.copy()
        circuitZ = anzats.circuit.copy()

        # Apply Pauli-X operators to the i-th and (i+1)-th qubits and simulate the circuit
        circuitX.append(cirq.X(qubits[i]))
        circuitX.append(cirq.X(qubits[(i + 1)]))
        vector2 = simulator.simulate(circuitX).state_vector()
        value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
        print(f'Applying Pauli-X operators to {qubits[i]} and {qubits[(i + 1)]}')

        # Apply Pauli-Y operators to the i-th and (i+1)-th qubits and simulate the circuit
        circuitY.append(cirq.Y(qubits[i]))
        circuitY.append(cirq.Y(qubits[(i + 1)]))
        vector2 = simulator.simulate(circuitY).state_vector()
        value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
        print(f'Applying Pauli-Y operators to {qubits[i]} and {qubits[(i + 1)]}')
        
        # Apply Pauli-Z operators to the i-th and (i+1)-th qubits and simulate the circuit
        circuitZ.append(cirq.Z(qubits[i]))
        circuitZ.append(cirq.Z(qubits[(i + 1)]))
        vector2 = simulator.simulate(circuitZ).state_vector()
        value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
        print(f'Applying Pauli-Z operators to {qubits[i]} and {qubits[(i + 1)]}')
        
    # Return the real part of the expectation value
    return np.real(value)

def get_expectation_afm_heisenberg_periodic(function_args, gamma, beta):
    # periodic
    anzats = AnzatsAFMHeisenberg_periodic(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits
    length = function_args.length
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j
    for i in range(function_args.length-1):
        circuitX = anzats.circuit.copy()
        circuitY = anzats.circuit.copy()
        circuitZ = anzats.circuit.copy()

        circuitX.append(cirq.X(qubits[i%length]))
        circuitX.append(cirq.X(qubits[(i+1)%length]))
        vector2 = simulator.simulate(circuitX).state_vector()
        value += np.dot(vector2.conj(), vector)

        circuitY.append(cirq.Y(qubits[i%length]))
        circuitY.append(cirq.Y(qubits[(i+1)%length]))
        vector2 = simulator.simulate(circuitY).state_vector()
        value += np.dot(vector2.conj(), vector)

        circuitZ.append(cirq.Z(qubits[i%length]))
        circuitZ.append(cirq.Z(qubits[(i+1)%length]))
        vector2 = simulator.simulate(circuitZ).state_vector()
        value += np.dot(vector2.conj(), vector)
    return np.real(value)

def get_expectation_afm_heisenberg_new_symmetry(function_args, gamma, beta):
    # This function calculates the expectation value for the AFM Heisenberg model with a new symmetry
    # using a 2D quantum circuit ansatz.

    # Initialize the ansatz for the 2D AFM Heisenberg model with given parameters
    anzats = AnzatsAFMHeisenberg_2d(function_args.length, gamma, beta)
    circuit = anzats.circuit
    qubits = anzats.qubits

    # Initialize the quantum simulator
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)

    # Simulate the circuit to get the initial state vector
    vector = simulator.simulate(circuit).state_vector()

    value = 0 + 0j  # Initialize the expectation value as a complex number

    rows = 2  # Number of rows in the 2D grid of qubits
    cols = int(function_args.length / 2)  # Number of columns in the 2D grid of qubits

    # Iterate over the 2D grid of qubits
    for i in range(rows):
        for j in range(cols - 1):
            # Horizontal neighbors

            # Create copies of the original circuit for each Pauli operator (X, Y, Z)
            circuitX = circuit.copy()
            circuitY = circuit.copy()
            circuitZ = circuit.copy()

            # Apply Pauli-X operators to the horizontal neighbors and simulate the circuit
            circuitX.append(cirq.X(qubits[i * cols + j]))
            circuitX.append(cirq.X(qubits[i * cols + j + 1]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            print(f'Applying Pauli-X operators to {qubits[i * cols + j]} and {qubits[i * cols + j + 1]}')
            
            # Apply Pauli-Y operators to the horizontal neighbors and simulate the circuit
            circuitY.append(cirq.Y(qubits[i * cols + j]))
            circuitY.append(cirq.Y(qubits[i * cols + j + 1]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            print(f'Applying Pauli-Y operators to {qubits[i * cols + j]} and {qubits[i * cols + j + 1]}')
            
            # Apply Pauli-Z operators to the horizontal neighbors and simulate the circuit
            circuitZ.append(cirq.Z(qubits[i * cols + j]))
            circuitZ.append(cirq.Z(qubits[i * cols + j + 1]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            print(f'Applying Pauli-Z operators to {qubits[i * cols + j]} and {qubits[i * cols + j + 1]}')
            
    for i in range(rows - 1):
        for j in range(cols):
            # Vertical neighbors

            # Create copies of the original circuit for each Pauli operator (X, Y, Z)
            circuitX = circuit.copy()
            circuitY = circuit.copy()
            circuitZ = circuit.copy()

            # Apply Pauli-X operators to the vertical neighbors and simulate the circuit
            circuitX.append(cirq.X(qubits[i * cols + j]))
            circuitX.append(cirq.X(qubits[(i + 1) * cols + j]))
            vector2 = simulator.simulate(circuitX).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            print(f'Applying Pauli-X operators to {qubits[i * cols + j]} and {qubits[i * cols + j + 1]}')
            
            # Apply Pauli-Y operators to the vertical neighbors and simulate the circuit
            circuitY.append(cirq.Y(qubits[i * cols + j]))
            circuitY.append(cirq.Y(qubits[(i + 1) * cols + j]))
            vector2 = simulator.simulate(circuitY).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            print(f'Applying Pauli-Y operators to {qubits[i * cols + j]} and {qubits[i * cols + j + 1]}')
            
            # Apply Pauli-Z operators to the vertical neighbors and simulate the circuit
            circuitZ.append(cirq.Z(qubits[i * cols + j]))
            circuitZ.append(cirq.Z(qubits[(i + 1) * cols + j]))
            vector2 = simulator.simulate(circuitZ).state_vector()
            value += np.dot(vector2.conj(), vector)  # Add the overlap to the expectation value
            print(f'Applying Pauli-Z operators to {qubits[i * cols + j]} and {qubits[i * cols + j + 1]}')
            
    # Return the real part of the expectation value
    return np.real(value)

class ToricCodeArgs():
    def __init__(self, length, qsim_option):
        self.length = length
        self.qsim_option = qsim_option


def get_expectation_toric_code(function_args, gamma, beta):
    # open boundary
    anzats = AnzatsToricCode(function_args.length, gamma, beta)
    length = function_args.length

    circuit = anzats.circuit
    qubits = anzats.qubits
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    vector = simulator.simulate(circuit).state_vector()
    value = 0 + 0j
    
    for i in range(function_args.length-1):
        for j in range(function_args.length-1):
            circuit = anzats.circuit.copy()
            circuit.append(cirq.X(qubits[i][(j+1)%length]))
            circuit.append(cirq.Y(qubits[(i+1)%length][(j+1)%length]))
            circuit.append(cirq.X(qubits[(i+1)%length][j]))
            circuit.append(cirq.Y(qubits[i][j]))
            vector2 = simulator.simulate(circuit).state_vector()
            value -= np.dot(vector2.conj(), vector)

    return value

class HubbardArgs():
    def __init__(self, 
                length, 
                x_dimension,
                y_dimension,
                tunneling,
                coulomb,
                chemical_potential=0.0,
                magnetic_field=0.0,
                periodic=True,
                spinless=False,
                particle_hole_symmetry=False,
                qsim_option=None):
        
        self.length = length
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        self.tunneling = tunneling
        self.coulomb = coulomb
        
        self.chemical_potential = chemical_potential
        self.magnetic_field = magnetic_field
        self.periodic = periodic
        self.spinless = spinless
        self.particle_hole_symmetry = particle_hole_symmetry

        if not (qsim_option):
            self.qsim_option = qsim_option
        else:
            qsim_option = {'t': 4, 'f': 1}

class HubbardArgs():
    def __init__(self, 
                x_dimension,
                y_dimension,
                tunneling,
                coulomb,
                hopping_matrix = -1.0 * np.array([ \
                    [ 0., 1., 1., 0.], \
                    [ 1., 0., 0., 1.], \
                    [ 1., 0., 0., 1.], \
                    [ 0., 1., 1., 0.], \
                ]),
                chemical_potential=0.0,
                magnetic_field=0.0,
                periodic=True,
                spinless=False,
                particle_hole_symmetry=False,
                sc_gap=1.0,
                qsim_option=None):
        
        self.x_dimension = x_dimension
        self.y_dimension = y_dimension
        n_sites = x_dimension * y_dimension
        self.n_sites = n_sites
        self.n_qubits = 2*n_sites
        self.tunneling = tunneling
        self.coulomb = coulomb
        
        self.chemical_potential = chemical_potential
        self.magnetic_field = magnetic_field
        self.periodic = periodic
        self.spinless = spinless
        self.particle_hole_symmetry = particle_hole_symmetry
        self.sc_gap = sc_gap

        if not (qsim_option):
            self.qsim_option = qsim_option
        else:
            self.qsim_option = {'t': 4, 'f': 1}

        self.hopping_matrix = hopping_matrix
        
def get_expectation_bcs_hubbard(function_args, gamma, beta):
    # Initialize the anzats circuit for the Hubbard model
    anzats = AnzatsBCSHubbard(function_args=function_args, gamma=gamma, beta=beta)
    circuit = anzats.circuit
    simulator = qsimcirq.QSimSimulator(function_args.qsim_option)
    # vector = simulator.simulate(circuit, qubit_order=anzats.qubits).state_vector()

    value = 0 + 0j

    # # Calculate contribution from one-body terms (hopping terms)
    term = sum(_get_one_body_term_on_hubbard(function_args.hopping_matrix, function_args.n_sites))
    term_qubit = of.qubit_operator_to_pauli_sum(of.jordan_wigner(term))
    value_one = simulator.simulate_expectation_values(circuit, observables=[term_qubit])
    value += sum(value_one)

    # Calculate contribution from two-body terms (Coulomb interaction)
    term = sum(_get_two_body_term_on_hubbard(function_args.coulomb, function_args.n_sites, function_args.n_qubits))
    term_qubit = of.qubit_operator_to_pauli_sum(of.jordan_wigner(term))
    value_two = simulator.simulate_expectation_values(circuit, observables=[term_qubit])
    value += sum(value_two)

    return value

