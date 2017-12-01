#! /usr/bin/env python

import numpy as np
import time
from scipy.ndimage.filters import gaussian_filter

def wrap_each(thetas):
    for i, theta in enumerate(thetas):
        thetas[i] = wrap(theta)
    return thetas

def wrap(theta):
    while theta > np.pi:
        theta -= 2*np.pi
    while theta < -np.pi:
        theta += 2*np.pi
    return theta

def inv_sensor_model(loc, x, z, thk):
    z_max = 15000
    alpha = 1.
    beta = 5*np.pi/180.0
    p_occ = 0.7
    p_free = 0.3
    
    xi, yi = loc
    r = np.linalg.norm(loc - x[:2])
    phi = wrap(np.arctan2(loc[1] - x[1], loc[0] - x[0]) - x[2])
    k = np.argmin(np.abs(phi - thk))
#     print(np.abs(phi - thk[:, k]))
    l = 0.0
    if r > np.min([z_max, z[k] + alpha/2]) or np.abs(phi - thk[:, k]) > beta/2:
        l = 0.0
    elif z[k] < z_max and np.abs(r - z[k]) < alpha/2.:
        l = np.log(p_occ/(1-p_occ))
    elif r <= z[k]:
        l = np.log(p_free/(1-p_free))
    return l
        

def update_map(m, x, z, thk):
    height, width = m.shape
    for i in xrange(height):
        for j in xrange(width):
            loc = np.array([j, i])
            l = np.log(m[i, j]/(1 - m[i, j]))
            l += inv_sensor_model(loc, x, z, thk)
            m[i, j] = 1 - 1/(1 + np.exp(l))
    return m

class Occ_Map(object):
    def __init__(self, width=100, height=100, offset=(0., 0.), body_offset=(0., 0.), resolution=1.0, 
                 z_max = 150, alpha= 1., beta=2*np.pi/180, 
                 p_free = 0.4, p_occ = 0.6):
        m_width = int(width/resolution)
        m_height = int(height/resolution)
        self.resolution = resolution
        self.z_max = z_max
        self.alpha = alpha
        self.beta = beta
        self.l_free = np.log(p_free/(1 - p_free))
        self.l_occ = np.log(p_occ/(1 - p_occ))
#         self._m = np.zeros(m_width + 1, m_height + 1) + 0.5
        self._log_m = np.zeros((m_width + 1, m_height + 1))
        self._grid = np.mgrid[0:width + resolution:resolution, 0:height + resolution:resolution]
        self.body_offset = np.array(body_offset)
        self.map_offset = np.array(offset)
        self.free_mask = np.zeros_like(self._log_m, dtype=np.int)
        self.occ_mask = np.zeros_like(self._log_m, dtype=np.int)
        self.zero_mask = np.zeros_like(self._log_m, dtype=np.int)

        self.total_time = 0.
        self.iters = 0.
        
    def get_map(self):
        return 1.0 - 1.0/(1.0 + np.exp(self._log_m))
    
    def update(self, x, z):

        start = time.time()

        thk = z[1, :]
        thk = wrap_each(thk)

        z = z[0, :]

        body_off = np.array([[np.cos(x[2]), -np.sin(x[2])],[np.sin(x[2]), np.cos(x[2])]]).dot(self.body_offset[:, None])
        # print(body_off)
        rel_grid = self._grid - x[:2, None, None] - body_off[:, 0, None, None] - self.map_offset[:, None, None]
#         print(rel_grid.shape)
        
        r_grid = np.linalg.norm(rel_grid, axis=0)
        
        theta_grid = np.arctan2(rel_grid[1, :, :], rel_grid[0, :, :]) - x[2]
        # wrap
        theta_grid[theta_grid < -np.pi] += 2*np.pi
        theta_grid[theta_grid >  np.pi] -= 2*np.pi
        
        # generate an update mask
        # meas_mask = r_grid[:, :, np.newaxis] < z[np.newaxis, np.newaxis, :] - self.alpha/2.
        
        # max_mask = (r_grid < self.z_max)[:, :, np.newaxis]


        # max_mask = np.tile(max_mask, (1, 1, len(z)))
        
        # theta_mask = np.abs(theta_grid[:, :, np.newaxis] - thk[np.newaxis, :]) < self.beta/2.

        
        # free_mask = np.logical_or.reduce(max_mask & theta_mask & meas_mask, axis=-1)
        
        # # mask out the measurement range +/- the thickness alpha
        # meas_mask = r_grid[:, :, np.newaxis] < z[np.newaxis, np.newaxis, :] + self.alpha/2.
        # meas_mask = meas_mask & (r_grid[:, :, np.newaxis] > z[np.newaxis, np.newaxis, :] - self.alpha/2.)
        
        # occ_mask = np.logical_or.reduce(max_mask & theta_mask & meas_mask, axis=-1)
        
        # self._log_m += free_mask*self.l_free + occ_mask*self.l_occ


        # generate an update mask
        
        max_mask = (r_grid < self.z_max)

        np.copyto(self.free_mask, self.zero_mask)
        np.copyto(self.occ_mask, self.zero_mask)
        # self.free_mask = np.zeros_like(max_mask)
        # self.occ_mask = np.zeros_like(max_mask)

        for i, zi in enumerate(z):
            meas_mask = r_grid < zi - self.alpha/2.

            theta_mask = np.abs(theta_grid - thk[i]) < self.beta/2.

        
            self.free_mask = self.free_mask | (theta_mask & meas_mask)
    #         print("shape: {}".format(free_mask.shape))
            
            # mask out the measurement range +/- the thickness alpha
            meas_mask = r_grid < zi + self.alpha/2.
            meas_mask = meas_mask & (r_grid > zi - self.alpha/2.)
            
            self.occ_mask = self.occ_mask | (theta_mask & meas_mask)
        
        self._log_m += self.free_mask*self.l_free + self.occ_mask*self.l_occ

        t = time.time() - start
        self.total_time += t
        self.iters += 1.
        avg = self.total_time/self.iters
        print("it: {}, avg: {}".format(t, avg))

    def match(self, x, z):
        self.match_pts(x, z)

    def match_full(self, x, z):
        thk = wrap_each(thk)
        rel_grid = self._grid - x[:2, np.newaxis, np.newaxis] - body_off - self.offset[:, None, None]
#         print(rel_grid.shape)
        
        r_grid = np.linalg.norm(rel_grid, axis=0)
        
        theta_grid = np.arctan2(rel_grid[1, :, :], rel_grid[0, :, :]) - x[2]
        # wrap
        theta_grid[theta_grid < -np.pi] += 2*np.pi
        theta_grid[theta_grid >  np.pi] -= 2*np.pi
        
        # generate an update mask
        meas_mask = r_grid[:, :, np.newaxis] < z[np.newaxis, np.newaxis, :] - self.alpha/2.
        
        max_mask = (r_grid < self.z_max)[:, :, np.newaxis]
        max_mask = np.tile(max_mask, (1, 1, len(z)))
#         print("shape: {}".format(max_mask.shape))
        
        theta_mask = np.abs(theta_grid[:, :, np.newaxis] - thk[np.newaxis, :]) < self.beta/2.

        
        free_mask = np.logical_or.reduce(max_mask & theta_mask & meas_mask, axis=-1)
#         print("shape: {}".format(free_mask.shape))
        
        # mask out the measurement range +/- the thickness alpha
        meas_mask = r_grid[:, :, np.newaxis] < z[np.newaxis, np.newaxis, :] + self.alpha/2.
        meas_mask = meas_mask & (r_grid[:, :, np.newaxis] > z[np.newaxis, np.newaxis, :] - self.alpha/2.)
        
        occ_mask = np.logical_or.reduce(max_mask & theta_mask & meas_mask, axis=-1)

        raise("Not Yet Implemented")

    def match_pts(self, x, z):
        blurred = gaussian_filter(self._log_m, 0.1/self.resolution)

        scan_pts = x[:2, None] + z[0:1,:]*np.array([np.cos(z[1,:] + x[2]), np.sin(z[1,:] + x[2])]) + self.map_offset[:, None]

        idx = np.array(scan_pts/self.resolution, dtype=np.int)

        log_odds = np.sum(blurred[idx[0,:], idx[1,:]])
        print("log_odds: {}".format(log_odds))
        return 1.0 - 1.0/(1.0 + np.exp(log_odds))