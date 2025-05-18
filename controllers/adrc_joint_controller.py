import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.model = ManipulatorModel(Tp)
        self.Tp = Tp
        self.b = b
        self.kp = kp
        self.kd = kd

        # page 10 Decentralized ADRC 
        A = np.array([[0, 1, 0],
             [0, 0, 1],
             [0, 0, 0]])
        B = np.array([[0],
             [self.b],
             [0]])
        L = np.array([[3*p],[3*p**2],[p**3]])
        W = np.array([[1, 0, 0]])
        self.eso = ESO(A, B, W, L, q0, Tp)
        self.last_u = 0

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        self.eso.set_B(np.array([[0], [self.b], [0]]))
        # return NotImplementedError

    def calculate_control(self, idx, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        q, q_dot = x
        self.eso.update(q, self.last_u)
        q_approx, q_approx_dot, f = self.eso.get_state()
        v = self.kp * (q_d - q) + self.kd * (q_d_dot - q_approx_dot) + q_d_ddot
        u = (v - f) / self.b
        self.last_u = u
        return u

        # with disturbance rejection
        # control_signal = (v - f) / self.b
        # self.eso.update(x[0], control_signal)
        # return control_signal
