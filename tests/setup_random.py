import numpy as np
import h5py
import os

meta_all = {
    "PDE": "The Everything Formula",
    "Fields Scheme": "aBBc",
    "Fields": ["Field1", "Field2a", "Field2b", "Field3"],
    "Constants": ["Const1"],
    "Dt": 0.01,
}


def setup():
    np.random.seed(1)

    os.makedirs("tests/datasets", exist_ok=True)

    with h5py.File("tests/datasets/random.hdf5", "w") as f:

        for i in range(3):
            # create random array with 1000 frames, 4 fields, and 128 x 64 frames
            data = np.random.random((1000, 4, 128, 64))
            sim = f.create_dataset("sims/sim" + str(i), data=data)
            sim.attrs["Const1"] = np.random.random()

        for key, value in meta_all.items():
            f["sims/"].attrs[key] = value


def teardown():
    os.remove("tests/datasets/random.hdf5")
