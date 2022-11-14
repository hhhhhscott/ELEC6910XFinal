from utils.segmentation_statistics import SegmentationStatistics
import pandas as pd
import os
import glob
import SimpleITK as sitk
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import cv2
import config
from tensorboardX import SummaryWriter

netlist = ['proposed', 'r2attunet', 'r2unet', 'resunet', 'unet']
# netlist=['proposed']
for net in netlist:
    net_all_image_path = glob.glob(os.path.join("output", net, "Adam_DiceMeanLoss_lr_00001_bs_2", "3d_result","*.jpg"))
    writer = SummaryWriter(os.path.join("output", net, "Adam_DiceMeanLoss_lr_00001_bs_2"))
    for idx, image in enumerate(net_all_image_path):
        image = cv2.imread(image)
        writer.add_image('Test Sample/'+net, image.transpose(2,0,1), idx)
