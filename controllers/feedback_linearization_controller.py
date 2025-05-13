import numpy as np
from models.manipulator_model import ManipulatorModel
from .controller import Controller


class FeedbackLinearizationController(Controller):
    def __init__(self, Tp):
        self.model = ManipulatorModel(Tp)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        """
        Please implement the feedback linearization using self.model (which you have to implement also),
        robot state x and desired control v.
        """
        # v = q_r_ddot

        # PD feedback controller
        K_p = 0.02
        K_d = 0.2
        q_dot = x[2:4]
        q = x[0:2]
        #v = q_r_ddot + self.K_d * (q_r_dot - q_dot) - self.K_p * (q_r - q)
        v = q_r_ddot
        M = self.model.M(x)
        C = self.model.C(x)
        #tau = M.dot(v) +C.dot(q_dot)
        # print("BRUH")
        # print(M.shape, C.shape, v.shape, q_dot.shape)
        tau = M @ v + C @ q_dot
        return tau
