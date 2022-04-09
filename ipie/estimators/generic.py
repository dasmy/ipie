import sys
import numpy
from ipie.utils.backend import numlib as nl

def local_energy_generic_opt(system, G, Ghalf=None, eri=None):

    na = system.nup
    nb = system.ndown
    M = system.nbasis

    vipjq_aa = eri[0,:na**2*M**2].reshape((na,M,na,M))
    vipjq_bb = eri[0,na**2*M**2:na**2*M**2+nb**2*M**2].reshape((nb,M,nb,M))
    vipjq_ab = eri[0,na**2*M**2+nb**2*M**2:].reshape((na,M,nb,M))

    Ga, Gb = Ghalf[0], Ghalf[1]
    # Element wise multiplication.
    e1b = numpy.sum(system.H1[0]*G[0]) + numpy.sum(system.H1[1]*G[1])
    # Coulomb
    eJaa = 0.5 * numpy.einsum("irjs,ir,js", vipjq_aa, Ga, Ga)
    eJbb = 0.5 * numpy.einsum("irjs,ir,js", vipjq_bb, Gb, Gb)
    eJab = numpy.einsum("irjs,ir,js", vipjq_ab, Ga, Gb)

    eKaa = -0.5 * numpy.einsum("irjs,is,jr", vipjq_aa, Ga, Ga)
    eKbb = -0.5 * numpy.einsum("irjs,is,jr", vipjq_bb, Gb, Gb)


    e2b = eJaa + eJbb + eJab + eKaa + eKbb

    return (e1b + e2b + system.ecore, e1b + system.ecore, e2b)


def local_energy_generic_cholesky_opt(system, ham, Ga, Gb, Ghalfa, Ghalfb, rchola, rcholb):
    r"""Calculate local for generic two-body hamiltonian.

    This uses the cholesky decomposed two-electron integrals.

    Parameters
    ----------
    system : :class:`Generic`
        System information for Generic.
    ham : :class:`Abinitio`
        Contains necessary hamiltonian information
    G : :class:`numpy.ndarray`
        Walker's "green's function"
    Ghalf : :class:`numpy.ndarray`
        Walker's half-rotated "green's function"
    rchol : :class:`numpy.ndarray`
        trial's half-rotated choleksy vectors

    Returns
    -------
    (E, T, V): tuple
        Local, kinetic and potential energies.
    """
    # Element wise multiplication.

    e1b = nl.sum(ham.H1[0]*Ga) + nl.sum(ham.H1[1]*Gb)
    nalpha, nbeta = system.nup, system.ndown
    nbasis = ham.nbasis
    if rchola is not None:
        naux = rchola.shape[0]

    if (nl.isrealobj(rchola) and nl.isrealobj(rcholb)):
        Xa = rchola.dot(Ghalfa.real.ravel()) + 1.j * rchola.dot(Ghalfa.imag.ravel())
        Xb = rcholb.dot(Ghalfb.real.ravel()) + 1.j * rcholb.dot(Ghalfb.imag.ravel())
    else:
        Xa = rchola.dot(Ghalfa.ravel())
        Xb = rcholb.dot(Ghalfb.ravel())

    ecoul = nl.dot(Xa,Xa)
    ecoul += nl.dot(Xb,Xb)
    ecoul += 2*nl.dot(Xa,Xb)

    GhalfaT = Ghalfa.T.copy()
    GhalfbT = Ghalfb.T.copy()

    Ta = nl.zeros((nalpha,nalpha), dtype=numpy.complex128)
    Tb = nl.zeros((nbeta,nbeta), dtype=numpy.complex128)

    exx  = 0.j  # we will iterate over cholesky index to update Ex energy for alpha and beta
    if (nl.isrealobj(rchola) and nl.isrealobj(rcholb)):
        for x in range(naux):  # write a cython function that calls blas for this.
            rmi_a = rchola[x].reshape((nalpha,nbasis))
            rmi_b = rcholb[x].reshape((nbeta,nbasis))
            Ta[:,:].real = rmi_a.dot(GhalfaT.real)
            Ta[:,:].imag = rmi_a.dot(GhalfaT.imag)  # this is a (nalpha, nalpha)
            Tb[:,:].real = rmi_b.dot(GhalfbT.real)
            Tb[:,:].imag = rmi_b.dot(GhalfbT.imag) # this is (nbeta, nbeta)
            exx += nl.trace(Ta.dot(Ta)) + nl.trace(Tb.dot(Tb))
    else:
        for x in range(naux):  # write a cython function that calls blas for this.
            rmi_a = rchola[x].reshape((nalpha,nbasis))
            rmi_b = rcholb[x].reshape((nbeta,nbasis))
            Ta[:,:] = rmi_a.dot(GhalfaT)  # this is a (nalpha, nalpha)
            Tb[:,:] = rmi_b.dot(GhalfbT)  # this is (nbeta, nbeta)
            exx += nl.trace(Ta.dot(Ta)) + nl.trace(Tb.dot(Tb))

    e2b = 0.5 * (ecoul - exx)

    return (e1b + e2b + ham.ecore, e1b + ham.ecore, e2b)

def core_contribution(system, Gcore):
    hc_a = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[0]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[0]))
    hc_b = (numpy.einsum('pqrs,pq->rs', system.h2e, Gcore[1]) -
            0.5*numpy.einsum('prsq,pq->rs', system.h2e, Gcore[1]))
    return (hc_a, hc_b)

def core_contribution_cholesky(chol, G):
    nb = G[0].shape[-1]
    cmat = chol.reshape((-1,nb*nb))
    X = numpy.dot(cmat, G[0].ravel())
    Ja = numpy.dot(cmat.T, X).reshape(nb,nb)
    T = numpy.tensordot(chol, G[0], axes=((1),(0)))
    Ka = numpy.tensordot(T, chol, axes=((0,2),(0,2)))
    hca = Ja - 0.5 * Ka
    X = numpy.dot(cmat, G[1].ravel())
    Jb = numpy.dot(cmat.T, X).reshape(nb,nb)
    T = numpy.tensordot(chol, G[1], axes=((1),(0)))
    Kb = numpy.tensordot(T, chol, axes=((0,2),(0,2)))
    hcb = Jb - 0.5 * Kb
    return (hca, hcb)

def fock_generic(system, P):
    if system.sparse:
        mf_shift = 1j*P[0].ravel()*system.hs_pot
        mf_shift += 1j*P[1].ravel()*system.hs_pot
        VMF = 1j*system.hs_pot.dot(mf_shift).reshape(system.nbasis,system.nbasis)
    else:
        mf_shift = 1j*numpy.einsum('lpq,spq->l', system.hs_pot, P)
        VMF = 1j*numpy.einsum('lpq,l->pq', system.hs_pot, mf_shift)
    return system.h1e_mod - VMF
