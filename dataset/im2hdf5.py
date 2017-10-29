import os
import glob
import h5py
import scipy.misc as misc
import numpy as np

dataset_dir = "DIV2K_train"

f = h5py.File("{}.h5".format(dataset_dir), "w")
dt = h5py.special_dtype(vlen=np.dtype('uint8'))

for subdir in ["HR", "x2", "x3", "x4"]:
    im_paths = glob.glob(os.path.join(dataset_dir, subdir, "*.png"))
    im_paths.sort()
    grp = f.create_group(subdir)

    for i, path in enumerate(im_paths):
        im = misc.imread(path)
        print(path)
        grp.create_dataset(str(i), data=im)
