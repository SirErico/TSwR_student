import numpy as np
from .controller import Controller
from models.manipulator_model import ManiuplatorModel


class MMAController(Controller):
    def __init__(self, Tp):
        # TODO: Fill the list self.models with 3 models of 2DOF manipulators with different m3 and r3
        # I:   m3=0.1,  r3=0.05
        # II:  m3=0.01, r3=0.01
        # III: m3=1.0,  r3=0.3
        Model_1 = ManiuplatorModel(Tp = Tp)
        Model_1.m3 = 0.1
        Model_1.r3 = 0.05
        Model_2 = ManiuplatorModel(Tp = Tp)
        Model_2.m3 = 0.01
        Model_2.r3 = 0.01
        Model_3 = ManiuplatorModel(Tp = Tp)
        Model_3.m3 = 1.0
        Model_3.r3 = 0.3

        self.models = [Model_1, Model_2, Model_3]
        self.i = 0

        self.K_p = 10.0
        self.K_d = 10.0
        self.Tp = Tp
        self.u = np.zeros((2, 1))

    def choose_model(self, x):
        # TODO: Implement procedure of choosing the best fitting model from self.models (by setting self.i)
        x_p = np.zeros((2, 3))
        e = np.inf
        chosen_model_index = 0

        for index,model in enumerate(self.models):
            y = model.M(x).dot(self.u) + model.C(x).dot(np.reshape(x[2:], (2, 1)))
            x_p[0][index] = y[0]
            x_p[1][index] = y[1]

        for model_idx in range(len(self.models)):
            new_e = np.sum(abs(x[:2] - x_p[:, model_idx]))
            if e > new_e:
                e = new_e
                chosen_model_index = model_idx

        self.i = chosen_model_index
        print("Model: ",self.i)

    def calculate_control(self, x, q_r, q_r_dot, q_r_ddot):
        self.choose_model(x)
        q = x[:2]
        q_dot = x[2:]
        v = q_r_ddot
        #v = q_r_ddot - self.K_d * (q_dot - q_r_dot) - self.K_p * (q - q_r)
        M = self.models[self.i].M(x)
        C = self.models[self.i].C(x)
        # u = M.dot(v) + C.dot(q_dot)
        u = M @ v + C @ q_dot
        return u
