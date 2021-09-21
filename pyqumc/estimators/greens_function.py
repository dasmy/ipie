import numpy
import scipy.linalg

def greens_function(walker, trial):
    """Compute walker's green's function.

    Parameters
    ----------
    walker : object
        Walker wavefunction object.
    trial : object
        Trial wavefunction object.
    Returns
    -------
    det : float64 / complex128
        Determinant of overlap matrix.
    """
    nup = walker.nup
    ndown = walker.ndown

    if (walker.name == "SingleDetWalker"):
        ovlp = numpy.dot(walker.phi[:,:nup].T, trial.psi[:,:nup].conj())
        walker.Ghalf[0] = numpy.dot(scipy.linalg.inv(ovlp), walker.phi[:,:nup].T)
        walker.G[0] = numpy.dot(trial.psi[:,:nup].conj(), walker.Ghalf[0])
        sign_a, log_ovlp_a = numpy.linalg.slogdet(ovlp)
        sign_b, log_ovlp_b = 1.0, 0.0
        if ndown > 0:
            ovlp = numpy.dot(walker.phi[:,nup:].T, trial.psi[:,nup:].conj())
            sign_b, log_ovlp_b = numpy.linalg.slogdet(ovlp)
            walker.Ghalf[1] = numpy.dot(scipy.linalg.inv(ovlp), walker.phi[:,nup:].T)
            walker.G[1] = numpy.dot(trial.psi[:,nup:].conj(), walker.Ghalf[1])
        det = sign_a*sign_b*numpy.exp(log_ovlp_a+log_ovlp_b-walker.log_shift)
        return det
    elif (walker.name == "MultiDetWalker"):
        tot_ovlp = 0.0
        for (ix, detix) in enumerate(trial.psi):
            # construct "local" green's functions for each component of psi_T
            Oup = numpy.dot(walker.phi[:,:nup].T, detix[:,:nup].conj())
            # det(A) = det(A^T)
            ovlp = scipy.linalg.det(Oup)
            if abs(ovlp) < 1e-16:
                continue
            inv_ovlp = scipy.linalg.inv(Oup)
            walker.Gi[ix,0,:,:] = numpy.dot(detix[:,:nup].conj(),
                                          numpy.dot(inv_ovlp,
                                                    walker.phi[:,:nup].T)
                                          )
            Odn = numpy.dot(walker.phi[:,nup:].T, detix[:,nup:].conj())
            ovlp *= scipy.linalg.det(Odn)
            if abs(ovlp) < 1e-16:
                continue
            inv_ovlp = scipy.linalg.inv(Odn)
            tot_ovlp += trial.coeffs[ix].conj()*ovlp
            walker.Gi[ix,1,:,:] = numpy.dot(detix[:,nup:].conj(),
                                          numpy.dot(inv_ovlp,
                                                    walker.phi[:,nup:].T)
                                          )
            walker.ovlps[ix] = ovlp
            walker.weights[ix] = trial.coeffs[ix].conj() * walker.ovlps[ix]

        if(walker.split_trial_local_energy):
            tot_ovlp_energy = 0.0
            for (ix, detix) in enumerate(trial.le_psi):
                # construct "local" green's functions for each component of psi_T
                Oup = numpy.dot(walker.phi[:,:nup].T, detix[:,:nup].conj())
                # det(A) = det(A^T)
                ovlp = scipy.linalg.det(Oup)
                if abs(ovlp) < 1e-16:
                    continue
                inv_ovlp = scipy.linalg.inv(Oup)
                walker.le_Gi[ix,0,:,:] = numpy.dot(detix[:,:nup].conj(),
                                              numpy.dot(inv_ovlp,
                                                        walker.phi[:,:nup].T)
                                              )
                Odn = numpy.dot(walker.phi[:,nup:].T, detix[:,nup:].conj())
                ovlp *= scipy.linalg.det(Odn)
                if abs(ovlp) < 1e-16:
                    continue
                inv_ovlp = scipy.linalg.inv(Odn)
                tot_ovlp_energy += trial.le_coeffs[ix].conj()*ovlp
                walker.le_Gi[ix,1,:,:] = numpy.dot(detix[:,nup:].conj(),
                                              numpy.dot(inv_ovlp,
                                                        walker.phi[:,nup:].T)
                                              )
                walker.le_weights[ix] = trial.le_coeffs[ix].conj() * walker.ovlps[ix]

            # walker.le_weights *= (tot_ovlp_energy / tot_ovlp)
            walker.le_oratio = tot_ovlp_energy / tot_ovlp
        return tot_ovlp
    elif (walker.name == "ThermalWalker"):
        return walker.greens_function(trial)
        
# Green's functions
def gab(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    inv_O = scipy.linalg.inv((A.conj().T).dot(B))
    GAB = B.dot(inv_O.dot(A.conj().T))
    return GAB


def gab_mod(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    O = numpy.dot(B.T, A.conj())
    GHalf = numpy.dot(scipy.linalg.inv(O), B.T)
    G = numpy.dot(A.conj(), GHalf)
    return (G, GHalf)

def gab_spin(A, B, na, nb):
    GA, GAH = gab_mod(A[:,:na],B[:,:na])
    if nb > 0:
        GB, GBH = gab_mod(A[:,na:],B[:,na:])
    return numpy.array([GA, GB]), [GAH, GBH]


def gab_mod_ovlp(A, B):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle =
        [B(A^{\dagger}B)^{-1}A^{\dagger}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent (an element of) the trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Matrix representation of the bra used to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    inv_O = scipy.linalg.inv(numpy.dot(B.T, A.conj()))
    GHalf = numpy.dot(inv_O, B.T)
    G = numpy.dot(A.conj(), GHalf)
    return (G, GHalf, inv_O)


def gab_multi_det(A, B, coeffs):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    For example, usually A would represent a multi-determinant trial wavefunction.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    coeffs: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients. Assumed to be complex
        conjugated.

    Returns
    -------
    GAB : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    # Todo: check energy evaluation at later point, i.e., if this needs to be
    # transposed. Shouldn't matter for Hubbard model.
    Gi = numpy.zeros(A.shape)
    overlaps = numpy.zeros(A.shape[1])
    for (ix, Aix) in enumerate(A):
        # construct "local" green's functions for each component of A
        # Todo: list comprehension here.
        inv_O = scipy.linalg.inv((Aix.conj().T).dot(B))
        Gi[ix] = (B.dot(inv_O.dot(Aix.conj().T))).T
        overlaps[ix] = 1.0 / scipy.linalg.det(inv_O)
    denom = numpy.dot(coeffs, overlaps)
    return numpy.einsum('i,ijk,i->jk', coeffs, Gi, overlaps) / denom


def gab_multi_ghf_full(A, B, coeffs, bp_weights):
    """Green's function for back propagation.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    coeffs: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients. Assumed to be complex
        conjugated.
    bp_weights : :class:`numpy.ndarray`
        Factors arising from GS orthogonalisation.

    Returns
    -------
    G : :class:`numpy.ndarray`
        (One minus) the green's function.
    """
    M = A.shape[1] // 2
    Gi, overlaps = construct_multi_ghf_gab(A, B, coeffs)
    scale = max(max(bp_weights), max(overlaps))
    full_weights = bp_weights * coeffs * overlaps / scale
    denom = sum(full_weights)
    G = numpy.einsum('i,ijk->jk', full_weights, Gi) / denom

    return G


def gab_multi_ghf(A, B, coeffs, Gi=None, overlaps=None):
    """Construct components of multi-ghf trial wavefunction.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Matrix representation of the ket used to construct G.
    Gi : :class:`numpy.ndarray`
        Array to store components of G. Default: None.
    overlaps : :class:`numpy.ndarray`
        Array to overlaps. Default: None.

    Returns
    -------
    Gi : :class:`numpy.ndarray`
        Array to store components of G. Default: None.
    overlaps : :class:`numpy.ndarray`
        Array to overlaps. Default: None.
    """
    M = B.shape[0] // 2
    if Gi is None:
        Gi = numpy.zeros(shape=(A.shape[0],A.shape[1],A.shape[1]), dtype=A.dtype)
    if overlaps is None:
        overlaps = numpy.zeros(A.shape[0], dtype=A.dtype)
    for (ix, Aix) in enumerate(A):
        # construct "local" green's functions for each component of A
        # Todo: list comprehension here.
        inv_O = scipy.linalg.inv((Aix.conj().T).dot(B))
        Gi[ix] = (B.dot(inv_O.dot(Aix.conj().T)))
        overlaps[ix] = 1.0 / scipy.linalg.det(inv_O)
    return (Gi, overlaps)


def gab_multi_det_full(A, B, coeffsA, coeffsB, GAB, weights):
    r"""One-particle Green's function.

    This actually returns 1-G since it's more useful, i.e.,

    .. math::
        \langle \phi_A|c_i^{\dagger}c_j|\phi_B\rangle = [B(A^{*T}B)^{-1}A^{*T}]_{ji}

    where :math:`A,B` are the matrices representing the Slater determinants
    :math:`|\psi_{A,B}\rangle`.

    .. todo: Fix docstring

    Here we assume both A and B are multi-determinant expansions.

    .. warning::
        Assumes A and B are not orthogonal.

    Parameters
    ----------
    A : :class:`numpy.ndarray`
        Numpy array of the Matrix representation of the elements of the bra used
        to construct G.
    B : :class:`numpy.ndarray`
        Array containing elements of multi-determinant matrix representation of
        the ket used to construct G.
    coeffsA: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients for wavefunction A. Assumed to
        be complex conjugated.
    coeffsB: :class:`numpy.ndarray`
        Trial wavefunction expansion coefficients for wavefunction A. Assumed to
        be complex conjugated.
    GAB : :class:`numpy.ndarray`
        Matrix of Green's functions.
    weights : :class:`numpy.ndarray`
        Matrix of weights needed to construct G

    Returns
    -------
    G : :class:`numpy.ndarray`
        Full Green's function.
    """
    for ix, (Aix, cix) in enumerate(zip(A, coeffsA)):
        for iy, (Biy, ciy) in enumerate(zip(B, coeffsB)):
            # construct "local" green's functions for each component of A
            inv_O = scipy.linalg.inv((Aix.conj().T).dot(Biy))
            GAB[ix,iy] = (Biy.dot(inv_O)).dot(Aix.conj().T)
            GAB[ix,iy] = (Biy.dot(inv_O)).dot(Aix.conj().T)
            weights[ix,iy] =  cix*(ciy.conj()) / scipy.linalg.det(inv_O)
    denom = numpy.sum(weights)
    G = numpy.einsum('ij,ijkl->kl', weights, GAB) / denom
    return G
