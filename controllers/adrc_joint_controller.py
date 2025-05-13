import numpy as np
from observers.eso import ESO
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class ADRCJointController(Controller):
    def __init__(self, b, kp, kd, p, q0, Tp):
        self.model = ManiuplatorModel(Tp)
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

    def set_b(self, b):
        ### TODO update self.b and B in ESO
        self.b = b
        self.eso.set_B(np.array([[0], [self.b], [0]]))
        # return NotImplementedError

    def calculate_control(self, idx, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement ADRC
        Minv = np.linalg.inv(self.model.M([x[0], x[1], 0, 0]))
        # self.set_b(Minv[idx, idx]) # setting b (53) point 9
        _, q_approx_dot, f = self.eso.get_state()
        v = self.kp * (q_d - x[0]) + self.kd * (q_d_dot - q_approx_dot) + q_d_ddot
        
        self.eso.update(x[0], v)
        return v

        # with disturbance rejection
        # control_signal = (v - f) / self.b
        # self.eso.update(x[0], control_signal)
        # return control_signal
