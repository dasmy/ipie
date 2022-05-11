import numpy
import sys
import time

from ipie.hamiltonians.generic import Generic, read_integrals, construct_h1e_mod
from ipie.utils.mpi import get_shared_array, have_shared_mem
from ipie.utils.pack import pack_cholesky

def get_hamiltonian(system, ham_opts=None, verbose=0, comm=None):
    """Wrapper to select hamiltonian class

    Parameters
    ----------
    ham_opts : dict
        Hamiltonian input options.
    verbose : bool
        Output verbosity.

    Returns
    -------
    ham : object
        Hamiltonian class.
    """
    if ham_opts['name'] == 'Generic':
        filename = ham_opts.get('integrals', None)
        use_shmem = ham_opts.get('use_shared_memory', True)
        if filename is None:
            if comm.rank == 0:
                print("# Error: integrals not specfied.")
                sys.exit()
        start = time.time()
        hcore, chol, h1e_mod, enuc = get_generic_integrals(filename,
                                                           comm=comm,
                                                           verbose=verbose)
        if verbose:
            print("# Time to read integrals: {:.6f}".format(time.time()-start))


        start = time.time()

        nbsf = hcore.shape[-1]
        nchol = chol.shape[-1]
        idx = numpy.triu_indices(nbsf)

        chol = chol.reshape((nbsf,nbsf,nchol))

        shmem = have_shared_mem(comm) and use_shmem
        if shmem:
            if comm.rank == 0:
                cp_shape = (nbsf*(nbsf+1)//2, nchol)
                dtype = chol.dtype
            else:
                cp_shape = None
                dtype = None

            shape = comm.bcast(cp_shape, root=0)
            dtype = comm.bcast(dtype, root=0)
            chol_packed = get_shared_array(
                    comm,
                    shape,
                    dtype,
                    use_shmem=use_shmem)
            if comm.rank == 0:
                pack_cholesky(idx[0],idx[1], chol_packed, chol)
            comm.Barrier()
        else:
            dtype = chol.dtype
            cp_shape = (nbsf*(nbsf+1)//2, nchol)
            chol_packed = numpy.zeros(cp_shape, dtype=dtype)
            pack_cholesky(idx[0],idx[1], chol_packed, chol)

        chol = chol.reshape((nbsf*nbsf,nchol))

        if verbose:
            print("# Time to pack Cholesky vectors: {:.6f}".format(time.time()-start))

        ham = Generic(h1e = hcore, chol=chol, chol_packed=chol_packed, ecore=enuc, h1e_mod = h1e_mod, options=ham_opts, verbose = verbose)
        mem = ham.chol_packed.nbytes / (1024.0**3)
        if verbose:
            print("# Approximate memory required by packed Cholesky vectors %f GB"%mem)
    else:
        if comm.rank == 0:
            print("# Error: unrecognized hamiltonian name {}.".format(ham_opts['name']))
            sys.exit()

    return ham

def get_generic_integrals(filename, comm=None, verbose=False, use_shem=True):
    """Read generic integrals, potentially into shared memory.

    Parameters
    ----------
    filename : string
        File containing 1e- and 2e-integrals.
    comm : MPI communicator
        split communicator. Optional. Default: None.
    verbose : bool
        Write information.

    Returns
    -------
    hcore : :class:`numpy.ndarray`
        One-body hamiltonian.
    chol : :class:`numpy.ndarray`
        Cholesky tensor L[ik,n].
    h1e_mod : :class:`numpy.ndarray`
        Modified one-body Hamiltonian following subtraction of normal ordered
        contributions.
    enuc : float
        Core energy.
    """
    shmem = have_shared_mem(comm) and use_shmem
    if verbose:
        print("# Have shared memory: {}".format(shmem))
    if shmem:
        if comm.rank == 0:
            hcore, chol, enuc = read_integrals(filename)
            hc_shape = hcore.shape
            ch_shape = chol.shape
            dtype = chol.dtype
        else:
            hc_shape = None
            ch_shape = None
            dtype = None
            enuc = None
        shape = comm.bcast(hc_shape, root=0)
        dtype = comm.bcast(dtype, root=0)
        enuc = comm.bcast(enuc, root=0)
        hcore_shmem = get_shared_array(comm, (2,)+shape, dtype)
        if comm.rank == 0:
            hcore_shmem[0] = hcore[:]
            hcore_shmem[1] = hcore[:]
        comm.Barrier()
        shape = comm.bcast(ch_shape, root=0)
        chol_shmem = get_shared_array(comm, shape, dtype)
        if comm.rank == 0:
            chol_shmem[:] = chol[:]
        comm.Barrier()
        h1e_mod_shmem = get_shared_array(comm, hcore_shmem.shape, dtype)
        if comm.rank == 0:
            construct_h1e_mod(chol_shmem, hcore_shmem, h1e_mod_shmem)
        comm.Barrier()
        return hcore_shmem, chol_shmem, h1e_mod_shmem, enuc
    else:
        hcore, chol, enuc = read_integrals(filename)
        h1 = numpy.array([hcore, hcore])
        h1e_mod = numpy.zeros(h1.shape, dtype=h1.dtype)
        construct_h1e_mod(chol, h1, h1e_mod)
        return h1, chol, h1e_mod, enuc
