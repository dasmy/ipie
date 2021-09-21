import copy
import numpy
import scipy.linalg
from pyqumc.estimators.local_energy import local_energy_multi_det
from pyqumc.estimators.greens_function import greens_function
from pyqumc.walkers.walker import Walker
from pyqumc.utils.misc import get_numeric_names

class MultiDetWalker(Walker):
    """Multi-Det style walker.

    Parameters
    ----------
    weight : int
        Walker weight.
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    index : int
        Element of trial wavefunction to initalise walker to.
    weights : string
        Initialise weights to zeros or ones.
    wfn0 : string
        Initial wavefunction.
    """

    def __init__(self, system, hamiltonian, trial, walker_opts={}, index=0,
                 weights='zeros', verbose=False, nprop_tot=None, nbp=None):
        if verbose:
            print("# Setting up MultiDetWalker object.")
        Walker.__init__(self, system, hamiltonian, trial, walker_opts, index, nprop_tot, nbp)
        self.name = "MultiDetWalker"
        self.ndets = trial.psi.shape[0]
        dtype = numpy.complex128
        # This stores an array of overlap matrices with the various elements of
        # the trial wavefunction.
        self.inv_ovlp = [numpy.zeros(shape=(self.ndets, system.nup, system.nup),
                                     dtype=dtype),
                         numpy.zeros(shape=(self.ndets, system.ndown, system.ndown),
                                    dtype=dtype)]
        # TODO: RENAME to something less like weight
        if weights == 'zeros':
            self.weights = numpy.zeros(self.ndets, dtype=dtype)
        else:
            self.weights = numpy.ones(self.ndets, dtype=dtype)
        self.ovlps = numpy.zeros(self.ndets, dtype=dtype)
        # Compute initial overlap. Avoids issues with singular matrices for
        # PHMSD.
        self.ot = self.overlap_direct(trial)
        # TODO: fix name.
        self.ovlp = self.ot
        self.le_oratio = 1.0
        if verbose:
            print("# Initial overlap of walker with trial wavefunction: {:13.8e}"
                  .format(self.ot.real))
        # Green's functions for various elements of the trial wavefunction.
        self.Gi = numpy.zeros(shape=(self.ndets, 2, hamiltonian.nbasis,
                                     hamiltonian.nbasis), dtype=dtype)
        
        self.split_trial_local_energy = trial.split_trial_local_energy

        if(trial.split_trial_local_energy):
            self.le_ndets = trial.le_psi.shape[0]
            self.le_Gi = numpy.zeros(shape=(self.le_ndets, 2, hamiltonian.nbasis,
                                         hamiltonian.nbasis), dtype=dtype)
            if weights == 'zeros':
                self.le_weights = numpy.zeros(self.le_ndets, dtype=dtype)
            else:
                self.le_weights = numpy.ones(self.le_ndets, dtype=dtype)

        # Actual green's function contracted over determinant index in Gi above.
        # i.e., <psi_T|c_i^d c_j|phi>
        self.G = numpy.zeros(shape=(2, hamiltonian.nbasis, hamiltonian.nbasis),
                             dtype=dtype)
        # Contains overlaps of the current walker with the trial wavefunction.
        greens_function(self, trial)
        self.nb = hamiltonian.nbasis
        self.buff_names, self.buff_size = get_numeric_names(self.__dict__)

        self.le_oratio = 1.0

        # self.noisy_overlap = walker_opts.get('noisy_overlap', False)
        # self.noise_level = walker_opts.get('noise_level', -5)

        # if (verbose):
        #     if (self.noisy_overlap):
        #         print("# Overlap measurement is noisy with a level {}".format(self.noise_level))

    def overlap_direct(self, trial):
        nup = self.nup
        for (i, det) in enumerate(trial.psi):
            Oup = numpy.dot(det[:,:nup].conj().T, self.phi[:,:nup])
            Odn = numpy.dot(det[:,nup:].conj().T, self.phi[:,nup:])
            self.ovlps[i] = scipy.linalg.det(Oup) * scipy.linalg.det(Odn)
            if abs(self.ovlps[i]) > 1e-16:
                self.inv_ovlp[0][i] = scipy.linalg.inv(Oup)
                self.inv_ovlp[1][i] = scipy.linalg.inv(Odn)
            self.weights[i] = trial.coeffs[i].conj() * self.ovlps[i]
        return sum(self.weights)

    def inverse_overlap(self, trial):
        """Compute inverse overlap matrix from scratch.

        Parameters
        ----------
        trial : :class:`numpy.ndarray`
            Trial wavefunction.
        """
        nup = self.nup
        for (indx, t) in enumerate(trial.psi):
            Oup = numpy.dot(t[:,:nup].conj().T, self.phi[:,:nup])
            self.inv_ovlp[0][indx,:,:] = scipy.linalg.inv(Oup)
            Odn = numpy.dot(t[:,nup:].conj().T, self.phi[:,nup:])
            self.inv_ovlp[1][indx,:,:] = scipy.linalg.inv(Odn)

    def calc_otrial(self, trial):
        """Caculate overlap with trial wavefunction.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ovlp : float / complex
            Overlap.
        """
        for ix in range(self.ndets):
            det_O_up = 1.0 / scipy.linalg.det(self.inv_ovlp[0][ix])
            det_O_dn = 1.0 / scipy.linalg.det(self.inv_ovlp[1][ix])
            self.ovlps[ix] = det_O_up * det_O_dn
            self.weights[ix] = trial.coeffs[ix].conj() * self.ovlps[ix]
        return sum(self.weights)

    def calc_overlap(self, trial):
        """Caculate overlap with trial wavefunction.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.

        Returns
        -------
        ovlp : float / complex
            Overlap.
        """
        nup = self.nup
        for ix in range(self.ndets):
            Oup = numpy.dot(trial.psi[ix,:,:nup].conj().T, self.phi[:,:nup])
            Odn = numpy.dot(trial.psi[ix,:,nup:].conj().T, self.phi[:,nup:])
            det_Oup = scipy.linalg.det(Oup)
            det_Odn = scipy.linalg.det(Odn)
            self.ovlps[ix] = det_Oup * det_Odn
            self.weights[ix] = trial.coeffs[ix].conj() * self.ovlps[ix]
        
        ovlp = sum(self.weights)

        # if(self.noisy_overlap):
        #     ovlp += numpy.random.normal(scale=10**(self.noise_level),size=1)

        return ovlp

    def reortho(self, trial):
        """reorthogonalise walker.

        parameters
        ----------
        trial : object
            trial wavefunction object. for interface consistency.
        """
        nup = self.nup
        ndown = self.ndown
        (self.phi[:,:nup], Rup) = scipy.linalg.qr(self.phi[:,:nup],
                                                  mode='economic')
        Rdown = numpy.zeros(Rup.shape)
        if ndown > 0:
            (self.phi[:,nup:], Rdown) = scipy.linalg.qr(self.phi[:,nup:],
                                                        mode='economic')
        signs_up = numpy.diag(numpy.sign(numpy.diag(Rup)))
        if (ndown > 0):
            signs_down = numpy.diag(numpy.sign(numpy.diag(Rdown)))
        self.phi[:,:nup] = self.phi[:,:nup].dot(signs_up)
        if (ndown > 0):
            self.phi[:,nup:] = self.phi[:,nup:].dot(signs_down)
        drup = scipy.linalg.det(signs_up.dot(Rup))
        drdn = 1.0
        if (ndown > 0):
            drdn = scipy.linalg.det(signs_down.dot(Rdown))
        detR = drup * drdn
        self.ot = self.ot / detR
        return detR

    def contract_one_body(self, ints, trial):
        numer = 0.0
        denom = 0.0
        for i, Gi in enumerate(self.Gi):
            ofac = trial.coeffs[i].conj()*self.ovlps[i]
            numer += ofac * numpy.dot((Gi[0]+Gi[1]).ravel(),ints.ravel())
            denom += ofac
        return numer / denom
