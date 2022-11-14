from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
import glob
import SimpleITK as sitk
import random
import matplotlib.pyplot as plt

class heartDataset(Dataset):
    def __init__(self, crop_size, mode="train"):
        self.crop_size = crop_size
        self.root_path = os.getcwd()

        self.ct_path = os.path.join(self.root_path, "problem1_nii_" + mode, "*_volume.nii.gz")
        self.ct_path_list = sorted(glob.glob(self.ct_path))
    def __getitem__(self, index):
        image, label = self.get_data_batch_by_index(self.crop_size,index)
        return torch.from_numpy(image) ,torch.from_numpy(label)

    def __len__(self):
        return len(self.ct_path_list)

    def get_data_batch_by_index(self, crop_size, index):
        image, label = self.get_data_3d(self.ct_path_list[index])
        image, label = self.random_crop_3d(image, label, crop_size)
        return np.expand_dims(image, axis=0), label



    def get_data_3d(self,filepath):
        ct = sitk.ReadImage(filepath)
        ct = sitk.GetArrayFromImage(ct)

        label = sitk.ReadImage(filepath.replace('volume','seg'))
        label = sitk.GetArrayFromImage(label)
        return ct, label

    def random_crop_3d(self,img, label, crop_size):
        random_x_max = img.shape[0] - crop_size[0]
        random_y_max = img.shape[1] - crop_size[1]
        random_z_max = img.shape[2] - crop_size[2]

        if random_x_max < 0 or random_y_max < 0 or random_z_max < 0:
            return None

        x_random = random.randint(0, random_x_max)
        y_random = random.randint(0, random_y_max)
        z_random = random.randint(0, random_z_max)

        crop_img = img[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                   z_random:z_random + crop_size[2]]
        crop_label = label[x_random:x_random + crop_size[0], y_random:y_random + crop_size[1],
                     z_random:z_random + crop_size[2]]

        return crop_img, crop_label



if __name__ == '__main__':
    dataset = heartDataset([112,112,80],mode='train')  #batch size
    print(len(dataset))
    data_loader=DataLoader(dataset=dataset,batch_size=2,num_workers=1, shuffle=True)
    for data, target in (data_loader):
        print(data.shape, target.shape)
        plt.subplot(121)
        plt.imshow(data[0,0])
        plt.subplot(122)
        plt.imshow(target[0,0])
        plt.show()