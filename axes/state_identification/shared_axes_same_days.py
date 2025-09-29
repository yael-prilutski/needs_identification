import numpy as np
from numpy.linalg import svd
from os.path import join
import matplotlib.pyplot as plt

from configurations import *
from mouse import Mouse
from Processor import Processor
from axes.axes_analyzer import AxesAnalyzer

WEEK = 'opto_water_week'
RELEVANT_DAYS = 'opto_water_days'
RELEVANT_MICE = ALL_MICE


class WithinDayAxis(Processor):

    def __init__(self, mouse, vector_name, *args, **kwargs):
        Processor.__init__(self, *args, **kwargs)
        self.mouse = mouse
        self.days = mouse.days
        self.vector_name = vector_name
        self.sec_hz = mouse.sec_hz
        self.week = mouse.week
        self.analyzer = AxesAnalyzer(mouse.sec_hz)

    def run(self):
        vectors_dict = np.load(join(self.week.data_dir_path, 'axes', self.vector_name), allow_pickle=True)[()]

        day1 = vectors_dict[self.days[0].name]['sub']
        day2 = vectors_dict[self.days[1].name]['sub']

        M = np.column_stack([day1, day2])  # shape (m, 2)

        # SVD
        U, S, Vt = np.linalg.svd(M, full_matrices=False)

        # U- directions in neuron space, S- singular values (how strong each pattern is), Vt- how much each day expresses each pattern

        u1 = U[:, 0]
        sigma1 = S[0]

        alphas = u1 @ M  # how much that day aligns with the shared pattern

        shared = np.outer(u1, alphas) # the portion of the data explained by the shared pattern

        # cleaned (remove shared component)
        M_clean = M - shared
        day1_clean = M_clean[:, 0]
        day2_clean = M_clean[:, 1]

        print("First singular value (strength of shared pattern):", sigma1)
        print("projection coefficients (how much each day uses the shared pattern):", alphas)
        print("||shared||_F (total shared energy):", np.linalg.norm(shared, ord='fro'))
        print("||original M||_F:", np.linalg.norm(M, ord='fro'))
        print("fraction of energy removed:",
              np.linalg.norm(shared, ord='fro') ** 2 / (np.linalg.norm(M, ord='fro') ** 2 + 1e-20))


def main():
    for mouse_path in RELEVANT_MICE:
        if WEEK in mouse_path.keys():
            mouse = Mouse(mouse_path, mouse_path[RELEVANT_DAYS], mouse_path[WEEK])
            process = WithinDayAxis(mouse, vector_name='axes_vector_neutral.npy')
            process.run()
            print(f'finished processing {mouse.name}')
            print('-------------------------------------')


'__main__' == __name__ and main()
