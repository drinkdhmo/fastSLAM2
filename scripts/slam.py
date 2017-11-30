#! /usr/bin/python

from scipy.stats import multivariate_normal as mvn
import copy

def wrap_each(x):
    for i, y in enumerate(x):
        x[i] = wrap(y)
    return x
        
def wrap(x):
    while x < -np.pi:
        x += 2*np.pi
    while x > np.pi:
        x -= 2*np.pi
    return x

class Particle(object):
    def __init__(self, x0, map_params, f, del_f_u, Qu, Ts, Q=None):
        self.f = f
        self.del_f_u = del_f_u
        self.Qu = Qu
        self.mapper = Occ_Map(**map_params)
        self.x = x0
        self.Ts = Ts

        self.num_grid_pts = 5


    # def predict(self, u, z):
    #     uHat += np.random.multivariate_normal(np.zeros(self.m), self.Qu(u))
    #     self.x = self.f(self.x, uHat, self.Ts)
    #     if self.Q is not None:
    #         particle.x += np.random.multivariate_normal(np.zeros(self.n), self.Q)

    def update(self, u, z):

        x_bar = self.f(self.x, u, self.Ts)

        # need to get a "grid" covering "4-sigma" from my odometry proposal distribution
        # to then approximate my actual proposal distribution which is the product of 
        # the scan match pdf and the odometry pdf
        # Need the ith element from the diagonal of the inverse of the 
        # measurement covariance projected into the state space ( 6/(B*Qu*B.T)^-1_ii )^.5
        B = self.del_f_u(u, self.Ts)
        Qu = self.Qu(u)

        P = np.linalg.inv(B.dot(Qu).dot(B.T))
        bounds = 4./np.sqrt(np.diag(P))

        # throw down a grid around the local region to do some approximating and itegrating
        # x_idx = np.linspace(x_bar[0] - bounds[0], x_bar[0] + bounds[0], self.num_grid_pts)
        # y_idx = np.linspace(x_bar[1] - bounds[1], x_bar[1] + bounds[1], self.num_grid_pts)
        # theta_idx = np.linspace(x_bar[2] - bounds[2], x_bar[2] + bounds[2], self.num_grid_pts)

        grid = np.mgrid[x_bar[0] - bounds[0]:x_bar[0] + bounds[0]:2*bounds[0]/self.num_grid_pts, 
                        x_bar[1] - bounds[1]:x_bar[1] + bounds[1]:2*bounds[1]/self.num_grid_pts,
                        x_bar[2] - bounds[2]:x_bar[2] + bounds[2]:2*bounds[2]/self.num_grid_pts]

        grid_pts = grid.reshape(self.num_grid_pts**3, 3)
        
        tau_x = np.zeros(self.num_grid_pts**3)

        for i, pt in enumerate(grid_pts):
            p_odom = mvn.pdf(pt, mean=x_bar, cov=P)
            p_scan = self.mapper.match(pt, z)
            tau_x[i] = p_scan*p_odom

        eta = np.sum(tau_x)
        mu = np.sum(grid_pts*tau_x[:, None], axis=0)/eta
        Sigma = np.einsum(ki,kj,k->ij, grid_pts - mu[None, :], grid_pts - mu[None, :], tau_x)/eta

        self.x = np.random.randn(mu, Sigma)

        self.w = eta # self.mapper.match(self.x, z)

        self.mapper.update(self.x, z)

        return self.w
        



class FastSLAM(object):
    def __init__(self, x0, num_particles, map_params, f, del_f_u, Qu, Ts, Q=None, Qu=None):
        self.f = f
        self.del_f_u = del_f_u
        self.num_particles = num_particles

        self.Qu = Qu
        self.Q = Q

        # initialize the weights
        self.w = np.ones(self.num_particles)
        self.w = self.w/np.sum(self.w)
        
        # initialize the particles
        self.X = []
        for i in xrange(self.num_particles): 
            self.X.append(Particle(x0, map_params, f, del_f_u, Qu, Ts))
        self.best = self.X[0]
        self.Ts = Ts
        
    def lowVarSample(self, w):
        Xbar = []
        M = self.num_particles
        r = np.random.uniform(0, 1/M)
        c = w[0]
        i = 0
        last_i = i
        unique = 1
        for m in xrange(M):
            u = r + m/M
            while u > c:
                i += 1
                c = c + w[i]
            Xbar.append(copy.deepcopy(self.X[i]))
            if last_i != i:
                unique += 1
            last_i = i
        self.X = Xbar
        return unique

#     def predict(self, u, z):
#         self.u = u
        
#         # propagate the particles
# #         pdb.set_trace()
#         for particle in self.X:
#             particle.predict(u, z)
        
        
        

# #         self.x = np.mean(self.X, axis=1)[:, np.newaxis]
# #         self.P = np.cov(self.X, rowvar=True)
# #         print(self.X.shape)
# #         print(self.P.shape)
# #         print(self.x)
        
        
    def update(self, u, z, landmark_idx):

        
        for i, x in enumerate(self.X):
            self.w[i] = self.w[i]*x.update(u, z)
#             print(w)

        # for code simplicity, normalize the weights here
        self.w = self.w/np.sum(self.w)
        
        self.best_idx = np.argmax(w)        
        self.best = self.X[self.best_idx]
#         print("w: {}".format(w))
        
        self.n_eff = 1./np.sum(np.square(self.w))
        if self.n_eff < self.num_particles/2:
            unique = self.lowVarSample(w)
            self.w = np.ones(self.num_particles)
            print(unique)

        