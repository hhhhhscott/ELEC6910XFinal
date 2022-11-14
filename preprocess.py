import h5py
import os
import glob
import SimpleITK as sitk
import numpy as np

from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

root = os.getcwd()
data_dir = os.path.join(root, "problem1_datas")
mode = "train"
datapath = os.path.join(data_dir, mode, "*.h5")
all_files = sorted(glob.glob(datapath))
print(all_files)
def plot_3d(image, threshold=0):
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2, 1, 0)
    print(type(p))
    # p = image
    verts, faces,_,_ = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.70)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

for i, h5file in enumerate(all_files):
    f = h5py.File(h5file, 'r')
    image = f['image']
    image_arr = np.array(image)
    # plot_3d(image_arr)
    img = sitk.GetImageFromArray(image_arr)
    img.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(img, os.path.join(root, "problem1_nii_"+mode, str(i) + "_volume.nii.gz"))

for i, h5file in enumerate(all_files):
    f = h5py.File(h5file, 'r')
    image = f['label']
    image_arr = np.array(image)
    # plot_3d(image_arr)
    img = sitk.GetImageFromArray(image_arr)
    img.SetSpacing([1.0, 1.0, 1.0])
    sitk.WriteImage(img, os.path.join(root, "problem1_nii_"+mode, str(i) + "_seg.nii.gz"))

