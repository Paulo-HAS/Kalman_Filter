import numpy as np
import matplotlib.pyplot as plt


class KalmanFilter:
    def __init__(self, initial_x, initial_v, accel_variance):
        self._x = np.array([initial_x, initial_v])
        self._accel_variance = accel_variance
        self._P = np.eye(2)

    def predict(self, dt: float) -> None:
        # x = F * x
        # P = F P Ft + G Gt a
        F = np.array([[1, dt], [0, 1]])
        new_x = F.dot(self._x)
        G = np.array([0.5 * dt**2, dt]).reshape((2, 1))
        new_P = F.dot(self._P).dot(F.T) + G.dot(G.T) * self._accel_variance

        self._P = new_P
        self._x = new_x

    def update(self, meas_value, meas_variance):
        # y = z - H x
        # S = K P Ht + R
        #K = P Ht S^-1
        # x = x + K y
        # P = (I - K H) * P

        H = np.array([1, 0]).reshape((1, 2))

        z = np.array([meas_value])
        R = np.array([meas_variance])

        y = z - H.dot(self._x)
        S = H.dot(self._P).dot(H.T) + R

        K = self._P.dot(H.T).dot(np.linalg.inv(S))

        new_x = self._x + K.dot(y)
        new_P = (np.eye(2) - K.dot(H)).dot(self._P)

        self._P = new_P
        self._x = new_x


    @property
    def cov(self):
        return self._P
    
    @property
    def mean(self):
        return self._x

    @property
    def pos(self) -> float:
        return self._x[0]
    
    @property
    def vel(self) -> float:
        return self._x[1]


if __name__ =="__main__":
    plt.ion()
    plt.figure()

    kf = KalmanFilter(0.0, 1.0, 0.1)
    DT = 0.1
    num_steps = 1000
    meas_every_steps = 20

    real_x = 0.0
    real_v = 0.9
    meas_variance = 0.1 ** 2

    mus = []
    covs = []

    for step in range(num_steps):
        covs.append(kf.cov)
        mus.append(kf.mean)

        real_v *= 0.9

        real_x = real_x + DT * real_v

        kf.predict(DT)
        if step != 0 and step % meas_every_steps == 0:
            kf.update(meas_value= real_x+np.random.randn() * np.sqrt(meas_variance),
                      meas_variance=meas_variance)


    plt.subplot(2, 1, 1)
    plt.title('Position')
    plt.plot([mu[0] for mu in mus], 'r')
    plt.plot([mu[0] - 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], 'r--')
    plt.plot([mu[0] + 2*np.sqrt(cov[0,0]) for mu, cov in zip(mus, covs)], 'r--')

    plt.subplot(2, 1, 2)
    plt.title('Velocity')
    plt.plot([mu[1] for mu in mus], 'r')
    plt.plot([mu[1] + 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--')
    plt.plot([mu[1] - 2*np.sqrt(cov[1,1]) for mu, cov in zip(mus, covs)], 'r--')

    plt.show()
    plt.ginput(1)
