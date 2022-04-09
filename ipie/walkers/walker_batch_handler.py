import copy
import cmath
import h5py
import math
import numpy
import scipy.linalg
import sys
import time
from ipie.legacy.walkers.single_det_batch import SingleDetWalkerBatch
from ipie.legacy.walkers.multi_det_batch import MultiDetTrialWalkerBatch
from ipie.legacy.walkers.stack import FieldConfig
from ipie.utils.io import get_input_value
from ipie.utils.misc import update_stack
from mpi4py import MPI

from ipie.utils.backend import numlib as nl

class WalkerBatchHandler(object):
    """Container for groups of walkers which make up a wavefunction.

    Parameters
    ----------
    system : object
        System object.
    trial : object
        Trial wavefunction object.
    nwalkers : int
        Number of walkers to initialise.
    nprop_tot : int
        Total number of propagators to store for back propagation + itcf.
    nbp : int
        Number of back propagation steps.
    """

    def __init__(self, system, hamiltonian, trial, qmc, walker_opts={}, verbose=False,
                 comm=None, nprop_tot=None, nbp=None):
        self.nwalkers = qmc.nwalkers
        self.ntot_walkers = qmc.ntot_walkers
        self.write_freq = walker_opts.get('write_freq', 0)
        self.write_file = walker_opts.get('write_file', 'restart.h5')
        self.read_file = walker_opts.get('read_file', None)
        if comm is None:
            rank = 0
        else:
            rank = comm.rank
        if verbose:
            print("# Setting up walkers.handler_batch.Walkers.")
            print("# qmc.nwalkers = {}".format(self.nwalkers))
            print("# qmc.ntot_walkers = {}".format(self.ntot_walkers))

        assert(trial.name == 'MultiSlater')
        
        if (trial.ndets == 1):
            if verbose:
                print("# Using single det walker with a single det trial.")
            self.walker_type = 'SD'
            if (len(trial.psi.shape) == 3):
                trial.psi = trial.psi[0]
                trial.psia = trial.psia[0]
                trial.psib = trial.psib[0]
            self.walkers_batch = SingleDetWalkerBatch(system, hamiltonian, trial, 
                                nwalkers = self.nwalkers, walker_opts=walker_opts,
                                index=0, nprop_tot=nprop_tot,nbp=nbp)
        elif (trial.ndets > 1):
            if verbose:
                print("# Using single det walker with a multi det trial.")
            self.walker_type = 'SD'
            self.walkers_batch = MultiDetTrialWalkerBatch(system, hamiltonian, trial, 
                                nwalkers = self.nwalkers, walker_opts=walker_opts,
                                index=0, nprop_tot=nprop_tot,nbp=nbp)

        self.buff_size = self.walkers_batch.buff_size

        assert (nbp == None)

        self.walker_buffer = numpy.zeros(self.buff_size,
                                         dtype=numpy.complex128)

        self.pcont_method = get_input_value(walker_opts, 'population_control',
                                            default='pair_branch',
                                            alias=['pop_control'],
                                            verbose=verbose)
        self.min_weight = walker_opts.get('min_weight', 0.1)
        self.max_weight = walker_opts.get('max_weight', 4.0)
        
        if verbose:
            print("# Using {} population control "
                  "algorithm.".format(self.pcont_method))
            mem = float(self.walker_buffer.nbytes) / (1024.0**3)
            print("# Buffer size for communication: {:13.8e} GB".format(mem))
            if mem > 2.0:
                # TODO: FDM FIX THIS
                print(" # Warning: Walker buffer size > 2GB. May run into MPI"
                      "issues.")
        
        if not self.walker_type == "thermal":
            walker_batch_size = 3 * self.nwalkers + self.walkers_batch.phia.size + self.walkers_batch.phib.size
        if self.write_freq > 0:
            self.write_restart = True
            self.dsets = []
            with h5py.File(self.write_file,'w',driver='mpio',comm=comm) as fh5:
                fh5.create_dataset('walker_batch_%d'%mpi.rank, (walker_batch_size,),
                                   dtype=numpy.complex128)
        else:
            self.write_restart = False
        if self.read_file is not None:
            if verbose:
                print("# Reading walkers from %s file series."%self.read_file)
            self.read_walkers(comm)

        self.target_weight = qmc.ntot_walkers
        # self.nw = qmc.nwalkers
        self.set_total_weight(qmc.ntot_walkers)
        self.start_time_const = 0.0
        self.communication_time = 0.0
        self.non_communication_time = 0.0
        self.recv_time = 0.0
        self.send_time = 0.0

        if verbose:
            print("# Finish setting up walkers.handler.Walkers.")
        
    def orthogonalise(self, trial, free_projection):
        """Orthogonalise all walkers.

        Parameters
        ----------
        trial : object
            Trial wavefunction object.
        free_projection : bool
            True if doing free projection.
        """
        detR = self.walkers_batch.reortho()
        if free_projection:
            (magn, dtheta) = cmath.polar(self.walkers_batch.detR)
            self.walkers_batch.weight *= magn
            self.walkers_batch.phase *= cmath.exp(1j*dtheta)

    def add_field_config(self, nprop_tot, nbp, system, dtype):
        """Add FieldConfig object to walker object.

        Parameters
        ----------
        nprop_tot : int
            Total number of propagators to store for back propagation + itcf.
        nbp : int
            Number of back propagation steps.
        nfields : int
            Number of fields to store for each back propagation step.
        dtype : type
            Field configuration type.
        """
        for fc in self.walkers_batch.field_configs:
            fc = FieldConfig(system.nfields, nprop_tot, nbp, dtype)

    def copy_historic_wfn(self):
        """Copy current wavefunction to psi_n for next back propagation step."""
        for (i,w) in enumerate(self.walkers):
            numpy.copyto(self.walkers[i].phi_old, self.walkers[i].phi)

    def copy_bp_wfn(self, phi_bp):
        """Copy back propagated wavefunction.

        Parameters
        ----------
        phi_bp : object
            list of walker objects containing back propagated walkers.
        """
        for (i, (w,wbp)) in enumerate(zip(self.walkers, phi_bp)):
            numpy.copyto(self.walkers[i].phi_bp, wbp.phi)


    def start_time(self):
        self.start_time_const = time.time()
    def add_non_communication(self):
        self.non_communication_time += time.time() - self.start_time_const
    def add_communication(self):
        self.communication_time += time.time() - self.start_time_const
    def add_recv_time(self):
        self.recv_time += time.time() - self.start_time_const
    def add_send_time(self):
        self.send_time += time.time() - self.start_time_const

    def pop_control(self, comm):
        self.start_time()
        if self.ntot_walkers == 1:
            return
        weights = numpy.abs(nl.to_host(self.walkers_batch.weight))
        global_weights = numpy.empty(len(weights)*comm.size)
        self.add_non_communication()
        self.start_time()
        if self.pcont_method == "comb":
            comm.Allgather(weights, global_weights)
            total_weight = sum(global_weights)
        else:
            sum_weights = numpy.sum(weights)
            total_weight = numpy.empty(1, dtype=numpy.float64)
            comm.Reduce(sum_weights, total_weight,
                        op=MPI.SUM, root=0)
            comm.Bcast(total_weight, root=0)
            total_weight = total_weight[0]

        self.add_communication()
        self.start_time()

        # Rescale weights to combat exponential decay/growth.
        scale = total_weight / self.target_weight
        if total_weight < 1e-8:
            if comm.rank == 0:
                print("# Warning: Total weight is {:13.8e}: "
                      .format(total_weight))
                print("# Something is seriously wrong.")
            sys.exit()
        self.set_total_weight(total_weight)
        # Todo: Just standardise information we want to send between routines.
        self.walkers_batch.unscaled_weight = self.walkers_batch.weight
        self.walkers_batch.weight = self.walkers_batch.weight / scale
        if self.pcont_method == "comb":
            global_weights = global_weights / scale
            self.add_non_communication()
            self.comb(comm, global_weights)
        elif self.pcont_method == "pair_branch":
            # self.pair_branch(comm)
            self.pair_branch_fast(comm)
        else:
            if comm.rank == 0:
                print("Unknown population control method.")

    def comb(self, comm, weights):
        """Apply the comb method of population control / branching.

        See Booth & Gubernatis PRE 80, 046704 (2009).

        Parameters
        ----------
        comm : MPI communicator
        """
        # Need make a copy to since the elements in psi are only references to
        # walker objects in memory. We don't want future changes in a given
        # element of psi having unintended consequences.
        # todo : add phase to walker for free projection
        self.start_time()
        if comm.rank == 0:
            parent_ix = numpy.zeros(len(weights), dtype='i')
        else:
            parent_ix = numpy.empty(len(weights), dtype='i')
        if comm.rank == 0:
            total_weight = sum(weights)
            cprobs = numpy.cumsum(weights)
            r = numpy.random.random()
            comb = [(i+r) * (total_weight/self.target_weight) for i in
                    range(self.target_weight)]
            iw = 0
            ic = 0
            while ic < len(comb):
                if comb[ic] < cprobs[iw]:
                    parent_ix[iw] += 1
                    ic += 1
                else:
                    iw += 1
            data = {'ix': parent_ix}
        else:
            data = None

        self.add_non_communication()

        self.start_time()
        data = comm.bcast(data, root=0)
        self.add_communication()
        self.start_time()
        parent_ix = data['ix']
        # Keep total weight saved for capping purposes.
        # where returns a tuple (array,), selecting first element.
        kill = numpy.where(parent_ix == 0)[0]
        clone = numpy.where(parent_ix > 1)[0]
        reqs = []
        walker_buffers = []
        # First initiate non-blocking sends of walkers.
        self.add_non_communication()
        self.start_time()
        comm.barrier()
        self.add_communication()
        for i, (c, k) in enumerate(zip(clone, kill)):
            # Sending from current processor?
            if c // self.nwalkers == comm.rank:
                self.start_time()
                # Location of walker to clone in local list.
                clone_pos = c % self.nwalkers
                # copying walker data to intermediate buffer to avoid issues
                # with accessing walker data during send. Might not be
                # necessary.
                dest_proc = k // self.nwalkers
                # with h5py.File('before_{}.h5'.format(comm.rank), 'a') as fh5:
                    # fh5['walker_{}_{}_{}'.format(c,k,dest_proc)] = self.walkers[clone_pos].get_buffer()
                buff = self.walkers_batch.get_buffer(clone_pos)
                self.add_non_communication()
                self.start_time()
                reqs.append(comm.Isend(buff, dest=dest_proc, tag=i))
                self.add_send_time()
        # Now receive walkers on processors where walkers are to be killed.
        for i, (c, k) in enumerate(zip(clone, kill)):
            # Receiving to current processor?
            if k // self.nwalkers == comm.rank:
                self.start_time()
                # Processor we are receiving from.
                source_proc = c // self.nwalkers
                # Location of walker to kill in local list of walkers.
                kill_pos = k % self.nwalkers
                self.add_non_communication()
                self.start_time()
                comm.Recv(self.walker_buffer, source=source_proc, tag=i)
                # with h5py.File('walkers_recv.h5', 'w') as fh5:
                    # fh5['walk_{}'.format(k)] = self.walker_buffer.copy()
                self.add_recv_time()
                self.start_time()
                self.walkers_batch.set_buffer(kill_pos, self.walker_buffer)
                self.add_non_communication()
                # with h5py.File('after_{}.h5'.format(comm.rank), 'a') as fh5:
                    # fh5['walker_{}_{}_{}'.format(c,k,comm.rank)] = self.walkers[kill_pos].get_buffer()
        self.start_time()
        # Complete non-blocking send.
        for rs in reqs:
            rs.wait()
        # Necessary?
        # if len(kill) > 0 or len(clone) > 0:
            # sys.exit()
        comm.Barrier()
        self.add_communication()
        # Reset walker weight.
        # TODO: check this.
        # for w in self.walkers:
            # w.weight = 1.0
        self.start_time()
        self.walkers_batch.weight.fill(1.0)
        self.add_non_communication()

    def pair_branch_fast(self, comm):

        self.start_time()
        walker_info_0 = nl.to_host(nl.abs(self.walkers_batch.weight))
        self.add_non_communication()

        self.start_time()
        glob_inf = None
        glob_inf_0 = None
        glob_inf_1 = None
        glob_inf_2 = None
        glob_inf_3 = None
        if comm.rank == 0:
            glob_inf_0 = numpy.empty([comm.size, self.walkers_batch.nwalkers], dtype=numpy.float64)
            glob_inf_1 = numpy.empty([comm.size, self.walkers_batch.nwalkers], dtype=numpy.int64)
            glob_inf_1.fill(1)
            glob_inf_2 = numpy.array([[r for i in range (self.walkers_batch.nwalkers)] for r in range(comm.size)], dtype=numpy.int64)
            glob_inf_3 = numpy.array([[r for i in range (self.walkers_batch.nwalkers)] for r in range(comm.size)], dtype=numpy.int64)

        self.add_non_communication()

        self.start_time()
        comm.Gather(walker_info_0, glob_inf_0, root=0)
        self.add_communication()

        # Want same random number seed used on all processors
        self.start_time()
        if comm.rank == 0:
            # Rescale weights.
            glob_inf = numpy.zeros((self.walkers_batch.nwalkers*comm.size, 4), dtype=numpy.float64)
            glob_inf[:,0] = glob_inf_0.ravel()
            glob_inf[:,1] = glob_inf_1.ravel()
            glob_inf[:,2] = glob_inf_2.ravel()
            glob_inf[:,3] = glob_inf_3.ravel()
            total_weight = sum(w[0] for w in glob_inf)
            sort = numpy.argsort(glob_inf[:,0], kind='mergesort')
            isort = numpy.argsort(sort, kind='mergesort')
            glob_inf = glob_inf[sort]
            s = 0
            e = len(glob_inf) - 1
            tags = []
            isend = 0
            while s < e:
                if glob_inf[s][0] < self.min_weight or glob_inf[e][0] > self.max_weight:
                    # sum of paired walker weights
                    wab = glob_inf[s][0] + glob_inf[e][0]
                    r = numpy.random.rand()
                    if r < glob_inf[e][0] / wab:
                        # clone large weight walker
                        glob_inf[e][0] = 0.5 * wab
                        glob_inf[e][1] = 2
                        # Processor we will send duplicated walker to
                        glob_inf[e][3] = glob_inf[s][2]
                        send = glob_inf[s][2]
                        # Kill small weight walker
                        glob_inf[s][0] = 0.0
                        glob_inf[s][1] = 0
                        glob_inf[s][3] = glob_inf[e][2]
                    else:
                        # clone small weight walker
                        glob_inf[s][0] = 0.5 * wab
                        glob_inf[s][1] = 2
                        # Processor we will send duplicated walker to
                        glob_inf[s][3] = glob_inf[e][2]
                        send = glob_inf[e][2]
                        # Kill small weight walker
                        glob_inf[e][0] = 0.0
                        glob_inf[e][1] = 0
                        glob_inf[e][3] = glob_inf[s][2]
                    tags.append([send])
                    s += 1
                    e -= 1
                else:
                    break
            nw = self.nwalkers
            glob_inf = glob_inf[isort].reshape((comm.size,nw,4))
        else:
            data = None
            glob_inf = None
            total_weight = 0
        self.add_non_communication()
        self.start_time()
        
        data = numpy.empty([self.walkers_batch.nwalkers, 4], dtype=numpy.float64)
        comm.Scatter(glob_inf, data, root=0)

        self.add_communication()
        # Keep total weight saved for capping purposes.
        walker_buffers = []
        reqs = []
        for iw, walker in enumerate(data):
            if walker[1] > 1:
                self.start_time()
                tag = comm.rank*self.walkers_batch.nwalkers + walker[3]
                self.walkers_batch.weight[iw] = walker[0]
                buff = self.walkers_batch.get_buffer(iw)
                self.add_non_communication()
                self.start_time()
                reqs.append(comm.Isend(buff,
                                       dest=int(round(walker[3])),
                                       tag=tag))
                self.add_send_time()
        for iw, walker in enumerate(data):
            if walker[1] == 0:
                self.start_time()
                tag = walker[3]*self.walkers_batch.nwalkers + comm.rank
                self.add_non_communication()
                self.start_time()
                comm.Recv(self.walker_buffer,
                          source=int(round(walker[3])),
                          tag=tag)
                self.add_recv_time()
                self.start_time()
                self.walkers_batch.set_buffer(iw, self.walker_buffer)
                self.add_non_communication()
        self.start_time()
        for r in reqs:
            r.wait()
        self.add_communication()

    def pair_branch(self, comm):
        self.start_time()
        walker_info = [[abs(self.walkers_batch.weight[w]),1,comm.rank,comm.rank] for w in range(self.walkers_batch.nwalkers)]
        self.add_non_communication()
        self.start_time()
        glob_inf = comm.gather(walker_info, root=0)
        self.add_communication()

        # Want same random number seed used on all processors
        self.start_time()
        if comm.rank == 0:
            # Rescale weights.
            glob_inf = numpy.array([item for sub in glob_inf for item in sub])
            total_weight = sum(w[0] for w in glob_inf)
            sort = numpy.argsort(glob_inf[:,0], kind='mergesort')
            isort = numpy.argsort(sort, kind='mergesort')
            glob_inf = glob_inf[sort]
            s = 0
            e = len(glob_inf) - 1
            tags = []
            isend = 0
            while s < e:
                if glob_inf[s][0] < self.min_weight or glob_inf[e][0] > self.max_weight:
                    # sum of paired walker weights
                    wab = glob_inf[s][0] + glob_inf[e][0]
                    r = numpy.random.rand()
                    if r < glob_inf[e][0] / wab:
                        # clone large weight walker
                        glob_inf[e][0] = 0.5 * wab
                        glob_inf[e][1] = 2
                        # Processor we will send duplicated walker to
                        glob_inf[e][3] = glob_inf[s][2]
                        send = glob_inf[s][2]
                        # Kill small weight walker
                        glob_inf[s][0] = 0.0
                        glob_inf[s][1] = 0
                        glob_inf[s][3] = glob_inf[e][2]
                    else:
                        # clone small weight walker
                        glob_inf[s][0] = 0.5 * wab
                        glob_inf[s][1] = 2
                        # Processor we will send duplicated walker to
                        glob_inf[s][3] = glob_inf[e][2]
                        send = glob_inf[e][2]
                        # Kill small weight walker
                        glob_inf[e][0] = 0.0
                        glob_inf[e][1] = 0
                        glob_inf[e][3] = glob_inf[s][2]
                    tags.append([send])
                    s += 1
                    e -= 1
                else:
                    break
            nw = self.nwalkers
            glob_inf = glob_inf[isort].reshape((comm.size,nw,4))
        else:
            data = None
            total_weight = 0
        self.add_non_communication()
        self.start_time()
        data = comm.scatter(glob_inf, root=0)
        self.add_communication()
        # Keep total weight saved for capping purposes.
        walker_buffers = []
        reqs = []
        for iw, walker in enumerate(data):
            if walker[1] > 1:
                self.start_time()
                tag = comm.rank*len(walker_info) + walker[3]
                self.walkers_batch.weight[iw] = walker[0]
                buff = self.walkers_batch.get_buffer(iw)
                self.add_non_communication()
                self.start_time()
                reqs.append(comm.Isend(buff,
                                       dest=int(round(walker[3])),
                                       tag=tag))
                self.add_send_time()
        for iw, walker in enumerate(data):
            if walker[1] == 0:
                self.start_time()
                tag = walker[3]*len(walker_info) + comm.rank
                self.add_non_communication()
                self.start_time()
                comm.Recv(self.walker_buffer,
                          source=int(round(walker[3])),
                          tag=tag)
                self.add_recv_time()
                self.start_time()
                self.walkers_batch.set_buffer(iw, self.walker_buffer)
                self.add_non_communication()
        self.start_time()
        for r in reqs:
            r.wait()
        self.add_communication()

    def set_total_weight(self, total_weight):
        self.walkers_batch.total_weight = total_weight

    def get_write_buffer(self):
        buff = numpy.concatenate([[self.walkers_batch.weight], [self.walkers_batch.phase], [self.walkers_batch.ot], self.walkers_batch.phi.ravel()])
        return buff

    def set_walkers_batch_from_buffer(self, buff):
        self.walkers_batch.weight = buff[0:self.nwalkers]
        self.walkers_batch.phase = buff[self.nwalkers:self.nwalkers*2]
        self.walkers_batch.ot = buff[self.nwalkers*2:self.nwalkers*3]
        self.walkers_batch.phi = buff[self.nwalkers*3:].reshape(self.walkers_batch.phi.shape)

    def write_walkers_batch(self, comm):
        start = time.time()
        with h5py.File(self.write_file,'r+',driver='mpio',comm=comm) as fh5:
            # for (i,w) in enumerate(self.walkers):
                # ix = i + self.nwalkers*comm.rank
            buff = self.get_write_buffer()
            fh5['walker_%d'%comm.rank][:] = self.get_write_buffer()
        if comm.rank == 0:
            print(" # Writing walkers to file.")
            print(" # Time to write restart: {:13.8e} s"
                  .format(time.time()-start))

    def read_walkers_batch(self, comm):
        with h5py.File(self.read_file, 'r') as fh5:
            try:
                self.set_walkers_batch_from_buffer(fh5['walker_%d'%comm.rank][:])
            except KeyError:
                print(" # Could not read walker data from:"
                      " %s"%(self.read_file))
