import cirq 
import openfermion as of
import numpy as np

def _get_one_body_term_on_hubbard(hopping_matrix, n_sites):

    # get fermionOperator terms list in Hubbard model
    one_body_terms = []
    # spin up
    for i in range(n_sites):
        for j in range(i):
            hopping = hopping_matrix[i][j]
            # hopping = tunneling
            site_i = i*2
            site_j = j*2
            term_conjugated = of.FermionOperator(((site_i, 1), (site_j, 0)), coefficient=hopping) \
                + of.FermionOperator(((site_j, 1), (site_i, 0)), coefficient=hopping)
            one_body_terms.append(term_conjugated)
    # spin down
    for i in range(n_sites):
        for j in range(i):
            hopping = hopping_matrix[i][j]
            # hopping = tunneling
            site_i = i*2 + 1
            site_j = j*2 + 1
            term_conjugated = of.FermionOperator(((site_i, 1), (site_j, 0)), coefficient=hopping) \
                + of.FermionOperator(((site_j, 1), (site_i, 0)), coefficient=hopping)
            one_body_terms.append(term_conjugated)
            
    return one_body_terms

def _get_two_body_term_on_hubbard(coulomb, n_sites, n_qubits):
    # charge_matrix = coulomb * (np.diag([1] * n_sites)) # diagonal matrix
    two_body_terms = [
            of.FermionOperator(((i, 1), (i, 0), (i + 1, 1), (i + 1, 0)), coefficient=coulomb)
            for i in range(0, n_qubits, 2)
        ]
    return two_body_terms
    

def _exponentiate_quad_ham(qubits, quad_ham, time):
    _, basis_change_matrix, _ = quad_ham.diagonalizing_bogoliubov_transform()
    orbital_energies, _ = quad_ham.orbital_energies()

    yield cirq.inverse(of.bogoliubov_transform(qubits, basis_change_matrix))
    for i in range(len(qubits)):
        yield cirq.rz(rads=-orbital_energies[i]*time).on(qubits[i])
    yield of.bogoliubov_transform(qubits, basis_change_matrix)
