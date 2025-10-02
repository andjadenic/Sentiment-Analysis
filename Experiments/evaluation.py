import numpy as np
import torch


if __name__ == '__main__':
    a = np.array([1.3, 2.5, -.8])  # <class 'numpy.ndarray'>
    t = torch.tensor(a)
    print(type(t))
    print(t)