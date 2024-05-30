import neal
import numpy as np
import matplotlib.pyplot as plt

class discrete_CIM_bad:
    def __init__(self, 
                 alpha: float, 
                 b: float, 
                 J: list,
                 x0: list | None = None
                 ):
        self.alpha = alpha
        self.b = b
        self.k = 1
        self.J = np.array(J)
        self.i_max = len(self.J)
        if not x0:
            self.x0 = np.random.uniform(-0.5, 0.5, self.i_max)
        else:
            self.x0 = x0
    
    def anneal(self, iters):
        x_k = np.zeros([self.i_max, iters])
        x_k[:,0] = self.x0
        for k in range(1, iters):
            for i in range(self.i_max):
                x_k[i,k] = self.alpha * x_k[i,k-1] - 8/9 * ((self.alpha) * x_k[i, k-1])**3  + 192/120 * ((self.alpha) * x_k[i, k-1])**5 + self.b * (self.J[i].dot(x_k[:,k-1]))
        return x_k
    
    def adaptive_anneal(self, iters):
        x_k = np.zeros([self.i_max, iters])
        x_k[:,0] = self.x0
        for k in range(1, iters):
            alpha = 1 - k/(1.5*iters)
            for i in range(self.i_max):
                x_k[i,k] = alpha * x_k[i,k-1] - 8/9 * ((alpha) * x_k[i, k-1])**3  + 192/120 * ((alpha) * x_k[i, k-1])**5 + self.b * (self.J[i].dot(x_k[:,k-1]))
        return x_k

class discrete_CIM:
    def __init__(self, 
                 alpha1: float,
                 alpha2: float, 
                 b: float, 
                 J: list,
                 x0: list | None = None
                 ):
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.b = b
        self.k = 1
        self.J = np.array(J)
        self.i_max = len(self.J)
        if not x0:
            self.x0 = np.random.uniform(-0.5, 0.5, self.i_max)
        else:
            self.x0 = x0
    
    def anneal(self, iters, seed = 0):
        np.random.seed(seed = seed)
        x_k = np.zeros([self.i_max, iters])
        energy = np.zeros(iters)
        x_k[:,0] = self.x0
        energy[0] = x_k[:,0].dot(self.J).dot(x_k[:,0].T)
        for k in range(1, iters):
            x_k[:,k] = np.cos(self.k*(np.cos(self.alpha2 * x_k[:,k-1] 
                                             - np.pi/4)**2 - 1/2)+ self.alpha1 * x_k[:,k-1] 
                              + self.b * (self.J[:].dot(x_k[:,k-1])) - np.pi/4)**2 - 1/2 + np.random.normal(0, 1/np.e**2, self.i_max)
            vec = np.sign(x_k[:,k])
            energy[k] = vec.dot(self.J).dot(vec.T)
            
        return x_k, energy
    
    def adaptive_anneal(self, iters, seed):
        np.random.seed(seed = seed)
        x_k = np.zeros([self.i_max, iters])
        x_k[:,0] = self.x0
        delta_a = (self.alpha2 - 1)/iters
        alpha2 = self.alpha2
        for k in range(1, iters):
            alpha2 = alpha2 - delta_a
            for i in range(self.i_max):
                x_k[i,k] = np.cos((np.cos(alpha2 * x_k[i,k-1] - np.pi/4)**2 - 1/2)+ self.alpha1 * x_k[i,k-1] + self.b * (self.J[i].dot(x_k[:,k-1])) - np.pi/4)**2 - 1/2 
        return x_k

    def test_anneal(self, alpha1, alpha2, alpha2_end, b, noise, iters, seed):
        np.random.seed(seed = seed)
        x_k = np.zeros([self.i_max, iters])
        energy = np.zeros(iters)
        x_k[:,0] = self.x0
        energy[0] = x_k[:,0].dot(self.J).dot(x_k[:,0].T)
        for k in range(1, iters):
            alpha2 = alpha2 - (k/iters)*(alpha2 - alpha2_end)
            x_k[:,k] = np.cos(self.k*(np.cos(alpha2 * x_k[:,k-1] 
                                             - np.pi/4)**2 - 1/2)+ alpha1 * x_k[:,k-1]
                              + b * (self.J[:].dot(x_k[:,k-1])) - np.pi/4)**2 - 1/2 + np.random.normal(0, noise, self.i_max)
            vec = np.sign(x_k[:,k])
            energy[k] = vec.dot(self.J).dot(vec.T)
            
        return x_k, energy

def plot_spin_beh(x_k):
    """"
    function for ploting spins
    """
    iters = range(np.shape(x_k)[1])
    for i in range(np.shape(x_k)[0]):
        plt.scatter(iters, x_k[i,:])
    plt.show()

def dwave(matrix,  num_reads=10):
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample_ising(h = np.ones(np.shape(matrix)[0]), J = matrix, num_reads = num_reads)
    x = sampleset.samples()._samples
    return x[-1]

def parser(path):
    f = open(path, 'r')
    str1 = f.readline().split()
    str1 = [int(str1[0]), int(str1[1]) ]
    matrix = np.zeros([str1[0], str1[0]])
    with open(path, 'r') as file:
        next(file)  # Отпускаем заголовок
        for line in file:
            i,j, value = line.strip().split()
            matrix[int(i)-1,int(j)-1] = int(value)
    return matrix