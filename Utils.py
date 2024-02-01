import torch
import numpy as np
from scipy.spatial.transform import Rotation as R



def meshgrid2D(h, w):
    """
        Generates a 2D meshgrid of size (h, w).
    """
    x_coords = torch.linspace(0.0, w - 1, w) 
    x_coords = torch.reshape(x_coords, [1, -1])
    x_coords = torch.tile(x_coords, [h, 1])
    x_coords = torch.reshape(x_coords, [1, -1])

    y_coords = torch.linspace(0.0, h - 1, h)
    y_coords = torch.reshape(y_coords, [-1, 1])
    y_coords = torch.tile(y_coords, [1, w])
    y_coords = torch.reshape(y_coords, [1, -1])

    # normalize to [-1, 1] for training patch size of 128^2
    x_coords /= 128
    x_coords -= 0.5
    x_coords *= 2.
    y_coords /= 128
    y_coords -= 0.5
    y_coords *= 2.
    # slice at z=0
    z_coords = torch.zeros_like(y_coords)

    # homogeneous coordinate (for translations)
    w_coords = torch.ones_like(y_coords)

    coords = torch.cat([x_coords, y_coords, z_coords, w_coords], 0)

    return coords


def get_random_slicing_matrices(BatchSize, random=False):
    """
        Samples a random transformation matrix, slice is at z=0.
        Rotate using random angles in [0, 90] and
        translate using random offsets in [-1, 1]
    """
    matrices = np.zeros((BatchSize, 4, 4)) # 4x4 transformation matrix
    for i in range(BatchSize):
        if not random:
            # only rotate around x
            angles = np.random.uniform(low=0., high=90.0, size=1)
            r = R.from_euler('x', angles, degrees=True)
        else:
            # rotate randomly around each axis
            angles = np.random.uniform(low=0., high=90.0, size=3)
            r = R.from_euler('xyz', angles, degrees=True)

        rot = np.identity(4, dtype=np.float32)
        rot[:3, :3] = r.as_matrix() # rotation matrix DCM

        # translate randomly in all 3 directions
        offset = np.random.uniform(low=-1., high=1., size=3)
        trans = np.identity(4, dtype=np.float32)
        trans[0:3, 3] = offset 

        # final matrix
        m = np.matmul(trans, rot)

        matrices[i, :, :] = m

    return matrices





"""
coords = meshgrid2D(128, 128) 
print('coords.shape:', coords.shape) # [4, 16384]
print('coords:', coords)
slicing_matrix_ph = get_random_slicing_matrices(16)
slicing_matrix_ph = torch.from_numpy(slicing_matrix_ph).float()
print('slicing_matrix_ph.shape:', slicing_matrix_ph.shape) # [16, 4, 4]
print('slicing_matrix_ph:', slicing_matrix_ph)
coords = torch.matmul(slicing_matrix_ph, coords)
print('coords.shape:', coords.shape) # [16, 4, 16384]
print('coords:', coords)
"""