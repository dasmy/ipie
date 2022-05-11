import numpy
import sys
from ipie.trial_wavefunction.multi_slater import MultiSlater
from ipie.utils.io import read_qmcpack_wfn_hdf, get_input_value
from ipie.legacy.estimators.greens_function import gab_spin

def get_trial_wavefunction(system, hamiltonian, options={}, mf=None,
                           comm=None, scomm=None, verbose=0):
    """Wrapper to select trial wavefunction class.

    Parameters
    ----------
    options : dict
        Trial wavefunction input options.
    system : class
        System class.
    cplx : bool
        If true then trial wavefunction will be complex.
    parallel : bool
        If true then running in parallel.

    Returns
    -------
    trial : class or None
        Trial wavfunction class.
    """
    if comm is not None and comm.rank == 0:
        if verbose:
            print("# Building trial wavefunction object.")
    wfn_file = get_input_value(options, 'filename', default=None,
                               alias=['wavefunction_file'], verbose=verbose)
    wfn_type = options.get('name', 'MultiSlater')
    shmem = options.get('used_shared_memory', True)
    if wfn_type == 'MultiSlater':
        psi0 = None
        if wfn_file is not None:
            if verbose:
                print("# Reading wavefunction from {}.".format(wfn_file))
            read, psi0 = read_qmcpack_wfn_hdf(wfn_file)
            thresh = options.get('threshold', None)
            if thresh is not None:
                coeff = read[0]
                ndets = len(coeff[abs(coeff)>thresh])
                if verbose:
                    print("# Discarding determinants with weight "
                          "  below {}.".format(thresh))
            else:
                ndets = options.get('ndets', None)
                if ndets is None:
                    ndets = len(read[0])
            if verbose:
                print("# Number of determinants in trial wavefunction: {}"
                      .format(ndets))
            if ndets is not None:
                wfn = []
                # Wavefunction is a tuple, immutable so have to iterate through
                for x in read:
                    wfn.append(x[:ndets])
        else:
            if verbose:
                print("# Guessing RHF trial wavefunction.")
            na = system.nup
            nb = system.ndown
            wfn = numpy.zeros((1,hamiltonian.nbasis,system.nup+system.ndown),
                              dtype=numpy.complex128)
            coeffs = numpy.array([1.0+0j])
            I = numpy.identity(hamiltonian.nbasis, dtype=numpy.complex128)
            wfn[0,:,:na] = I[:,:na]
            wfn[0,:,na:] = I[:,:nb]
            wfn = (coeffs, wfn)
        trial = MultiSlater(system, hamiltonian, wfn, init=psi0, options=options, verbose=verbose)
        if system.name == 'Generic':
            if (trial.ndets == 1 or trial.ortho_expansion):
                trial.half_rotate(system, hamiltonian, scomm,
                        use_shmem=use_shmem)
        rediag = get_input_value(
                options,
                'recompute_ci',
                default=False,
                alias=['rediag'],
                verbose=verbose)
        if rediag:
            if comm.rank == 0:
                if verbose:
                    print("# Recomputing trial wavefunction ci coeffs.")
                coeffs = trial.recompute_ci_coeffs(
                                system.nup,
                                system.ndown,
                                hamiltonian)
            else:
                coeffs = None
            coeffs = comm.bcast(coeffs, root=0)
            trial.coeffs = coeffs
    else:
        print("Unknown trial wavefunction type.")
        sys.exit()

    spin_proj = get_input_value(options, 'spin_proj', default=None,
                                alias=['spin_project'], verbose=verbose)
    init_walker = get_input_value(options, 'init_walker', default=None,
                              alias=['initial_walker'], verbose=verbose)
    if spin_proj:
        na, nb = system.nelec
        if verbose:
            print("# Performing spin projection for walker's initial wavefunction.")
        if comm.rank == 0:
            if init_walker == 'free_electron':
                eigs, eigv = numpy.linalg.eigh(system.H1[0])
            else:
                rdm, rdmh = gab_spin(trial.psi[0], trial.psi[0], na, nb)
                eigs, eigv = numpy.linalg.eigh(rdm[0]+rdm[1])
                ix = numpy.argsort(eigs)[::-1]
                trial.noons = eigs[ix]
                eigv = eigv[:,ix]
        else:
            eigv = None
        eigv = comm.bcast(eigv, root=0)
        trial.init[:,:na] = eigv[:,:na].copy()
        trial.init[:,na:] = eigv[:,:nb].copy()


    return trial
