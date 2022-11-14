from utils.segmentation_statistics import SegmentationStatistics
import pandas as pd
import os
import glob
import SimpleITK as sitk
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import config


def plot_3d(predict, label, text):
    threshold = 0
    # Position the scan upright,
    # so the head of the patient would be at the top facing the camera
    p1 = predict.transpose(2, 1, 0)
    p2 = label.transpose(2, 1, 0)

    verts1, faces1, _, _ = measure.marching_cubes_lewiner(p1, threshold)
    verts2, faces2, _, _ = measure.marching_cubes_lewiner(p2, threshold)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if text is not None:
        ax.text(-90, -90, -20, text, None, zorder=0, fontsize=15)
    # Fancy indexing: `verts[faces]` to generate a collection of triangles

    mesh1 = Poly3DCollection(verts1[faces1], alpha=0.40)
    mesh2 = Poly3DCollection(verts2[faces2], alpha=0.70)

    face_color1 = [0.75, 0.45, 0.45]
    face_color2 = [0.45, 0.45, 0.75]

    mesh1.set_facecolor(face_color1)
    mesh2.set_facecolor(face_color2)
    ax.add_collection3d(mesh1)
    ax.add_collection3d(mesh2)

    ax.set_xlim(0, p1.shape[0])
    ax.set_ylim(0, p1.shape[1])
    ax.set_zlim(0, p1.shape[2])


def evaluate(args, save_path):
    if not os.path.exists(os.path.join(save_path, '3d_result')): os.mkdir(os.path.join(save_path, '3d_result'))
    if not os.path.exists(os.path.join(save_path, '3d_result_text')): os.mkdir(os.path.join(save_path, '3d_result_text'))
    predict_path = os.path.join(save_path, 'seg_result', "*_predict.nii.gz")
    predict_all_list = sorted(glob.glob(predict_path))

    ground_path = os.path.join('problem1_nii_test', "*_seg.nii.gz")
    ground_all_list = sorted(glob.glob(ground_path))
    stat = []
    for i in range(len(predict_all_list)):
        label = sitk.ReadImage(ground_all_list[i])
        label = sitk.GetArrayFromImage(label).astype(bool)

        predict = sitk.ReadImage(predict_all_list[i])
        predict = sitk.GetArrayFromImage(predict).astype(bool)

        statdict = SegmentationStatistics(predict, label, [1, 1, 1]).to_dict()
        statdict['Filename'] = predict_all_list[i].split('/')[-1]
        # without text
        plot_3d(predict, label, text=None)
        plt.savefig(os.path.join(save_path, "3d_result", predict_all_list[i].split('/')[-1] + '.jpg'), dpi=300)
        title = ''
        for (key, value) in statdict.items():
            title = title + str(key) + ": " + str(value) + '\n'
        print(title)
        stat.append(statdict)
    df = pd.DataFrame(stat)
    csv_path = os.path.join(save_path, 'result.csv')
    df.to_csv(csv_path, index=False, header=True)


if __name__ == '__main__':
    netlist = ['proposed', 'r2attunet', 'r2unet', 'resunet', 'unet']
    for net in netlist:
        save_path = 'output/{}/Adam_DiceMeanLoss_lr_00001_bs_2'.format(net)
        args = config.args
        print(save_path)
        evaluate(args, save_path)
