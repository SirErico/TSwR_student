import numpy as np

# from models.free_model import FreeModel
from observers.eso import ESO
from .adrc_joint_controller import ADRCJointController
from .controller import Controller
from models.manipulator_model import ManipulatorModel
# from models.ideal_model import IdealModel


class ADRFLController(Controller):
    def __init__(self, Tp, q0, Kp, Kd, p):
        self.model = ManipulatorModel(Tp)
        self.Kp = Kp
        self.Kd = Kd
        # page 11
        # A- 6x6
        A = np.array([[0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]])
        p1 = p[0]
        p2 = p[1]
        self.L = np.array([[3*p1, 0],[0, 3*p2],[3*p1**2, 0],[0, 3*p2**2],[p1**3, 0],[0, p2**3]])
        W = np.array([[1, 0, 0, 0, 0, 0],[0,1, 0, 0, 0, 0]])
        self.eso = ESO(A, B, W, self.L, q0, Tp)
        self.update_params(q0[:2], q0[2:])

    def update_params(self, q, q_dot):
        ### TODO Implement procedure to set eso.A and eso.B
        x = np.concatenate([q, q_dot], axis=0)
        M = self.model.M(x)
        C = self.model.C(x)
        M_inv = np.linalg.inv(M)
        A = np.array([[0, 0, 1, 0, 0, 0],
             [0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 1, 0],
             [0, 0, 0, 0, 0, 1],
             [0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0]])
        B = np.array([[0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0],
                [0, 0]])
        A[2:4, 2:4] = -M_inv @ C
        B[2:4, :2] = M_inv
        self.eso.A = A
        self.eso.B = B

    def calculate_control(self, x, q_d, q_d_dot, q_d_ddot):
        ### TODO implement centralized ADRFLC
        M = self.model.M(x)
        C = self.model.C(x)
        q1, q2, q1_dot, q2_dot = x
        q = np.array([q1, q2])
        erm = self.eso.get_state()
        q_approx = erm[0:2]
        q_approx_dot = erm[2:4]
        f = erm[4:6]
        # 61 equation
        v = self.Kp @ (q_d - q) + self.Kd @ (q_d_dot - q_approx_dot) + q_d_ddot
        u = M @ (v - f) + C @ q_approx_dot
        self.last_u = u
        self.update_params(q_approx, q_approx_dot)
        self.eso.update(q.reshape(len(q), 1), u.reshape(len(u), 1)) 
        return u

        # with disturbance rejection
        # control_signal = (v - f) / self.b
        # self.eso.update(x[0], control_signal)
        # return control_signal
