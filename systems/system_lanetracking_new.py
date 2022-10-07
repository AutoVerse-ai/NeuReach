import sys
sys.path.append('..')
from ODEs.lanetracking import TC_Simulate
import numpy as np
import scipy.linalg
import copy
import matplotlib.pyplot as plt 

class LaneTrackingSystem():
    def __init__(
            self, 
            y_low, 
            y_high,
            theta_low,
            theta_high,
            r,
            TMAX = 0.5,
            dt = 0.01
        ):
        self.y_bound = [y_low, y_high]
        self.theta_bound = [theta_low, theta_high]
        self.r = r
        self.TMAX = TMAX
        self.dt = dt

    def sampleEllipsoid(self, S, z_hat, m_FA, Gamma_Threshold=1.0):
        # Sample random points from ellipsoid
        # https://www.onera.fr/sites/default/files/297/C013_-_Dezert_-_YBSTributeMonterey2001.pdf
        # S, the covariance matrix
        # z_hat, the center of matrix
        # m_FA, number of samples
        nz = S.shape[0]
        z_hat = z_hat.reshape(nz,1)

        X_Cnz = np.random.normal(size=(nz, m_FA))

        rss_array = np.sqrt(np.sum(np.square(X_Cnz),axis=0))
        kron_prod = np.kron( np.ones((nz,1)), rss_array)

        X_Cnz = X_Cnz / kron_prod       # Points uniformly distributed on hypersphere surface

        R = np.ones((nz,1))*( np.power( np.random.rand(1,m_FA), (1./nz)))

        unif_sph=R*X_Cnz;               # m_FA points within the hypersphere
        T = np.asmatrix(scipy.linalg.cholesky(S))    # Cholesky factorization of S => S=T’T


        unif_ell = T.H*unif_sph ; # Hypersphere to hyperellipsoid mapping

        # Translation and scaling about the center
        z_fa=(unif_ell * np.sqrt(Gamma_Threshold)+(z_hat * np.ones((1,m_FA))))

        return np.array(z_fa)

    def sample_X0(self):
        x_range = [0,0]
        y_range = copy.deepcopy(self.y_bound)
        theta_range = copy.deepcopy(self.theta_bound)
        r = self.r
        X0 = np.array(x_range+y_range+theta_range+[r])
        # print(X0)
        return X0

    def sample_t(self):
        return (np.random.randint(int(self.TMAX/self.dt))+1) * self.dt

    def sample_x0(self, X0):
        # if np.random.uniform()>0.5:
        #     x = np.random.choice([X0[0], X0[1]])
        #     y = np.random.choice([X0[2], X0[3]])
        #     theta = np.random.choice([X0[4], X0[5]])
        #     r = X0[6]

        #     d = np.random.choice([y-r,y+r])
        #     psi = np.random.choice([theta-r, theta+r])
        #     x0 = [x,y,theta,d,psi]

        # else:
        # print(X0)
        x = np.random.uniform(X0[0], X0[1])
        y = np.random.uniform(X0[2], X0[3])
        theta = np.random.uniform(X0[4],X0[5])
        r = X0[6]

        # C = np.eye(2)*(r**2)
        # center = np.array([y, theta])
        # x0 = self.sampleEllipsoid(C, center, 1).flatten().tolist()
        # x0 = [x,y,theta]+x0
        
        d = np.random.uniform(y-r,y+r)
        psi = np.random.uniform(theta-r, theta+r)
        x0 = [x,y,theta,d,psi]

        # print(x0)
        return x0

    def simulate(self, x0):
        return np.array(TC_Simulate("Default", x0, self.TMAX))

    def get_init_center(self, X0):
        center_x = (X0[0]+X0[1])/2
        center_y = (X0[2]+X0[3])/2
        center_theta = (X0[4]+X0[5])/2
        center_d = center_y
        center_psi = center_theta
        center = [center_x, center_y, center_theta, center_d, center_psi]
        return center

    def get_X0_normalization_factor(self):
        mean = np.zeros(len(self.sample_X0()))
        std = np.ones(len(self.sample_X0()))
        return [mean, std]

if __name__ == "__main__":
    config = LaneTrackingSystem(-0.9,-0.7,-0.17453292519943295,-0.10471975511965977,0.19897)
    X0 = config.sample_X0()
    print(X0)
    for i in range(10):
        x0 = config.sample_x0(X0)
        trace = config.simulate(x0)
        # print(x0)
        plt.plot(trace[:,2], trace[:,3],'r')
    plt.show()