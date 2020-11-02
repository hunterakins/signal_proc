import numpy as np
from matplotlib import pyplot as plt

'''
Description:
Implement the test cases given in Ramdas Kumaresan and Donald Tufts
"Estimating the Parameters of Exponentially Damped Sinusoids and Pole-Zero
Modeling in Noise"
1982 IEEE paper
Author: Hunter Akins
'''

"""
Routine to build model
"""
def build_backward_model(data, filter_order):
    """
    Input:
    data - numpy 1-d array
    data representing measurements
    filter_order - int
    number of filter coefficients for the backwards ar model
    Output:
    A - numpy 2-d array of dimensions (size(data)-filter_order) x filter_order
    h - numpy 1-d array
    Ab = -h
    """
    L = filter_order
    rows = data.size - L-1
    dims = rows, L
    A = np.zeros(dims, dtype=data.dtype)
    for i in range(rows):
        A[i,:] = data[i+1:i+1+L]
    A = A.conj()
    h = data[:-L-1].conj()
    return A,h

def get_lp_roots(b):
    """ If b is the LS solution to the lp model,
    this will get the roots and return them as a list?
    """
    # GET ROOTS OF POLY 1 + b1 / z + b2/z^2
    b = np.insert(b, 0, 1)
    char_func = np.poly1d(b)
    roots = char_func.r
    return roots


class Sim1:
    """
    Simulation 1
    25 samples of data.
    Noise is wgn.
    Two (complex) frequencies. si = -alphai + 2pi*i*fi
    Two amplitudes, ai
    """
    def __init__(self, snr):
        """
        just initialize the simulation with the default values provided by
        the paper
        """
        alpha1, f1 = .1, .52
        alpha2, f2 = .2, .42
        a1, a2 = 1, 1
        s1 = -alpha1 + complex(0, 1)*2*np.pi*f1
        s2 = -alpha2 + complex(0, 1)*2*np.pi*f2
        self.a1 = a1
        self.a2 = a2
        self.s1 = s1
        self.s2 = s2
        self.snr = snr


    def gen_data(self,N):
        # GENERATE DATA
        # N is number of data points
        sigma = np.sqrt(np.power(10, -self.snr/10) / 2)
        noise = sigma*np.random.randn(N)
        domain = np.linspace(0,N-1, N)
        a1, a2, s1, s2 = self.a1, self.a2, self.s1, self.s2
        y = a1*np.exp(s1*domain) + a2*np.exp(s2*domain) + noise
        self.y = y
        return y

    def gen_uneven_data(self, N, pert_size):
        """
        Input:
        N : int
            number of data samples
        pert_size : float
            size of random perturbation to sampling point (std dev)
        generates some data with non-uniform domain
        """
        sigma = np.sqrt(np.power(10, -snr/10) / 2)
        noise = sigma*np.random.randn(N)
        domain = np.linspace(0,N-1, N)
        perts = pert_size*np.random.randn(N)
        domain += perts
        a1, a2, s1, s2 = self.a1, self.a2, self.s1, self.s2
        y = a1*np.exp(s1*domain) + a2*np.exp(s2*domain) + noise
        self.y = y

    def run_simple_lp(self, N, L):
        self.gen_data(N)
        # BUILD AR MODEL Ab = -h
        A,h = build_backward_model(self.y, L)
        A = np.matrix(A)

        # INVERT FOR b
        b = -np.linalg.inv(A.H@A)*A.H@h
        b = np.array(b)
        b = b.reshape(b.size)
        # GET ROOTS OF POLY 1 + b1 / z + b2/z^2
        estimates = get_lp_roots(b)
        self.simple_lp_ests = estimates
        return

    def run_uneven_simple_lp(self, N, L, pert_size):
        self.gen_uneven_data(N,pert_size)
        # BUILD AR MODEL Ab = -h
        A,h = build_backward_model(self.y, L)
        A = np.matrix(A)

        # INVERT FOR b
        b = -np.linalg.inv(A.H@A)*A.H@h
        b = np.array(b)
        b = b.reshape(b.size)
        # GET ROOTS OF POLY 1 + b1 / z + b2/z^2
        estimates = get_lp_roots(b)
        self.simple_lp_ests = estimates
        return

    def compare_results(self):
        e1, e2 = self.svd_lp_ests[0], self.svd_lp_ests[1]
        true1, true2 = np.exp(-np.conj(self.s1)), np.exp(-np.conj(self.s2))
        plt.scatter(e1.real, e1.imag, color='r')
        plt.scatter(e2.real, e2.imag, color='r')
        plt.scatter(true1.real, true1.imag, color='g')
        plt.scatter(true2.real, true2.imag, color='g')
        plt.show()

    def run_lp_ensemble(self, N, L, M):
        # M is number of runs
        ests = []
        for i in range(M):
            self.run_simple_lp(N, L)
            ests.append(self.simple_lp_ests)
        self.simple_lp_ensemble_ests = ests

    def run_svd_lp(self, N, L, num_eigs):
        """
        Input:
        N : integer
            number of data points
        L is the filter order
        """
        self.gen_data(N)
        A,h = build_backward_model(self.y, L)
        A = np.matrix(A)
        u, s, vh = np.linalg.svd(A)
        u1 = u[:, :num_eigs]
        vh1 = vh[:num_eigs, :]
        s1 = np.diag(1/s[:num_eigs])
        A_trunc = u1@s1@vh1 # approximate A
        b = -vh1.H @ s1 @ u1.H@h
        b = np.array(b)
        b = b.reshape(b.size)
        estimates = get_lp_roots(b)
        self.svd_lp_ests = estimates
        return

    def run_uneven_svd_lp(self, N, L, num_eigs, pert_size):
        """
        Run svd lp algorithm on unevenly sampled data
        Input:
        N : integer
            number of data points
        L is the filter order
        num_eigs - int
            number of frequencies
        pert_size - float
            std of normal perturbations to domain
        """
        self.gen_uneven_data(N, pert_size)
        A,h = build_backward_model(self.y, L)
        A = np.matrix(A)
        u, s, vh = np.linalg.svd(A)
        u1 = u[:, :num_eigs]
        vh1 = vh[:num_eigs, :]
        s1 = np.diag(1/s[:num_eigs])
        A_trunc = u1@s1@vh1 # approximate A
        b = -vh1.H @ s1 @ u1.H@h
        b = np.array(b)
        b = b.reshape(b.size)
        estimates = get_lp_roots(b)
        self.svd_lp_ests = estimates
        return

    def run_tls_lp(self, N, L, num_eigs, pert_size):
        """
        use TLS algorithm in Moon and Stirling pg. 383 to compute
        filter values and associated complex signal estimates
        Input:
        N - int
            number of data points
        L - int
            the filter order
        num_eigs - int
            number of frequencies to search for
        pert_size - float
            how uneven to make the samples (which are integer spaced).
            it's the std dev of random normal perturbations added to the
            integer valued domain
        Output:

        """
        self.gen_uneven_data(N,pert_size)
        A,h = build_backward_model(self.y, L)
        C = np.c_[A, h]
        C = np.matrix(C)
        u, s, vh = np.linalg.svd(C)
        if s[-1] == s[-2]: # nonunique
            raise ValueError("Non unique singular values...haven't implemented yet")
        s_delta = s[-1]
        u_delta = np.matrix(u[:,-1])
        vh_delta = np.matrix(vh[-1,:])
        Delta = -s_delta*u_delta@vh_delta
        if vh_delta[0, -1] == 0:
            raise ValueError("No solution")
        vh_delta = np.array(vh_delta).T
        b =  vh_delta[:h.size,0] / vh_delta[-1,0]
        estimates = get_lp_roots(b)
        self.tls_lp_ests = estimates
        self.Delta = Delta
        return

    def run_svd_ensemble(self, N, L, M, num_eigs):
        # M is number of runs
        ests = []
        for i in range(M):
            self.run_svd_lp(N, L, num_eigs)
            ests.append(self.svd_lp_ests)
        self.svd_lp_ensemble_ests = ests

    def run_uneven_svd_ensemble(self, N, L, M, num_eigs, pert_size):
        # M is number of runs
        ests = []
        for i in range(M):
            self.run_uneven_svd_lp(N, L, num_eigs, pert_size)
            ests.append(self.svd_lp_ests)
        self.svd_lp_ensemble_ests = ests

    def run_tls_ensemble(self, N, L, M, num_eigs, pert_size):
        # M is number of runs
        ests = []
        for i in range(M):
            self.run_tls_lp(N, L, num_eigs, pert_size)
            ests.append(self.tls_lp_ests)
        self.tls_lp_ensemble_ests = ests

    def plot_ensemble(self, ensemble_ests, color='r'):
        for est in ensemble_ests:
            for root in est:
                plt.scatter(root.real, root.imag, color=color)
                plt.scatter(root.real, root.imag, color=color)
        vals = np.exp(-np.conj(self.s1)), np.exp(-np.conj(self.s2))
        plt.scatter(vals[0].real, vals[0].imag, color='b')
        plt.scatter(vals[1].real, vals[1].imag, color='b')

    def plot_svd_ensemble(self):
        self.plot_ensemble(self.svd_lp_ensemble_ests)

    def plot_simple_ensemble(self):
        self.plot_ensemble(self.simple_lp_ensemble_ests, color='g')

    def plot_tls_ensemble(self):
        self.plot_ensemble(self.tls_lp_ensemble_ests, color='m')

if __name__ == '__main__':
    """
    M is number of ensembles
    L is filter length
    snr is 20
    Not sure how they compute snr here...I think it should be root 2 / variance of
    noise but whatever it's close to what I think it should be
    N is number of points in the data
    """
    N,L, M = 25, 2, 40
    snr = 20
    sim = Sim1(snr)
#    sim.run_lp_ensemble(N,L,40)
    sim.run_svd_lp(N, 8, 2)
    sim.run_uneven_svd_ensemble(N, 8, M, 2, .2)
    sim.plot_svd_ensemble()
    #sim.run_lp_ensemble(N, L,M)
#    sim.run_tls_lp(N, 8, M, .1)
#    sim.run_tls_ensemble(N,8,M,2,.00)
#    sim.plot_simple_ensemble()
#    sim.plot_tls_ensemble()
    plt.show()
