import numpy as np
from .controller import Controller
from models.manipulator_model import ManipulatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        Model_1 = ManipulatorModel(Tp = Tp)
        Model_1.m3 = 0.1
        Model_1.r3 = 0.05
        Model_2 = ManipulatorModel(Tp = Tp)
        Model_2.m3 = 0.01
        Model_2.r3 = 0.01
        Model_3 = ManipulatorModel(Tp = Tp)
        Model_3.m3 = 1.0
        Model_3.r3 = 0.3

        self.models = [Model_1, Model_2, Model_3]
        self.i = 0

        self.K_p = 30.50
        self.K_d = 20.0
        self.Tp = Tp
        self.u = np.zeros((2, 1))
        self.x = np.zeros((4, 1))

    def choose_model(self, x):
        M_inv = np.linalg.inv(self.models.M(x))
        err = []
        for model in self.models:
            M = model.M(x)
            C = model.C(x)
            q = x[:2]
            q_dot = x[2:]
            error = np.linalg.norm(M @ q[:, np.newaxis] + C @ q_dot[:, np.newaxis])
            err.append(error)
        self.i = np.argmin(err)
        print(f"Model {self.i + 1} chosen")

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        #v = q_r_ddot
        v = q_r_ddot + self.K_d * (q_r_dot - q_dot) + self.K_p * (q_r - q)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        # u = M.dot(v) + C.dot(q_dot)
        u = M @ v[:, np.newaxis] + C @ q_dot[:, np.newaxis]
        self.x = x
        self.u = u
        return u
