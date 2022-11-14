from scipy import ndimage
import numpy as np
from torchvision import transforms as T
import torch, os
from torch.utils.data import Dataset, DataLoader
from glob import glob
import math
import SimpleITK as sitk
import os.path
import glob
import torch
import metrics as metrics
import config

from model.proposed import UNet, RecombinationBlock
from model.unet3d import UNet3D
from model.r2unet3d import R2UNet3D
from model.r2attunet3d import R2AttUNet3D
from model.resunet3d import ResUNet3D
from config import args


def sitk_read_raw(img_path, resize_scale=1):  # 读取3D图像并resale（因为一般医学图像并不是标准的[1,1,1]scale）
    nda = sitk.ReadImage(img_path)
    if nda is None:
        raise TypeError("input img is None!!!")
    nda = sitk.GetArrayFromImage(nda)  # channel first
    nda = ndimage.zoom(nda, [resize_scale, resize_scale, resize_scale], order=0)  # rescale
    return nda


def to_one_hot_3d(tensor, n_classes=2):  # shape = [batch, s, h, w]
    n, s, h, w = tensor.size()
    one_hot = torch.zeros(n, n_classes, s, h, w).scatter_(1, tensor.view(n, 1, s, h, w), 1)
    return one_hot


def test_Datasets(dataset_path, cut_param, resize_scale=1):
    data_list = glob.glob(os.path.join(dataset_path, '*_volume.nii.gz'))
    label_list = glob.glob(os.path.join(dataset_path, '*_seg.nii.gz'))
    data_list.sort()
    label_list.sort()
    print("The numbers of testset is ", len(data_list))
    for datapath, labelpath in zip(data_list, label_list):
        print("Start evaluate ", datapath)
        yield Mini_DataSet(datapath, labelpath, cut_param, resize_scale=resize_scale), datapath.split('-')[-1]


class Mini_DataSet(Dataset):
    def __init__(self, data_path, label_path, cut, resize_scale=1):
        self.resize_scale = resize_scale
        self.label_path = label_path
        self.data_path = data_path
        self.n_labels = 2
        # 读取一个data文件并归一化 shape:[s,h,w]
        self.data_np = sitk_read_raw(self.data_path, resize_scale=self.resize_scale)

        self.data_np = self.data_np.transpose(2, 1, 0)

        self.ori_shape = self.data_np.shape
        # 读取一个label文件 shape:[s,h,w]
        self.label_np = sitk_read_raw(self.label_path, resize_scale=self.resize_scale)
        self.label_np = self.label_np.transpose(2, 1, 0)
        # 扩展一定数量的slices，以保证卷积下采样合理运算
        self.cut = cut

        self.data_np = self.padding_img(self.data_np, self.cut)
        self.new_shape = self.data_np.shape
        self.data_np = self.extract_ordered_overlap(self.data_np, self.cut)  # 这里已经把一个fulldata变成了patchdata数组

    def __getitem__(self, index):
        data = self.data_np[index]
        # target = self.label_np[index]
        return torch.from_numpy(data)

    def __len__(self):
        return len(self.data_np)  # patch index

    def padding_img(self, img, C):  # 这个函数会调出一个3D的volume数据，然后向周围补0，为后面提取成块的3D数据做准备
        assert (len(img.shape) == 3)  # 3D array,第0维的长度
        #
        img_s, img_h, img_w = img.shape
        # imgs = 25, imgh = 256, imgw = 256

        # cut_param = {'patch_s': 32,
        #              'patch_h': 128,
        #              'patch_w': 128,
        #              'stride_s': 24,
        #              'stride_h': 96,
        #              'stride_w': 96}

        leftover_s = (img_s - C['patch_s']) % C['stride_s']
        # 17, (25-32)%24 -7+24=17
        leftover_h = (img_h - C['patch_h']) % C['stride_h']
        # 32, (256-128)%96=128%96=32
        leftover_w = (img_w - C['patch_w']) % C['stride_w']
        # 32, (256-128)%96=128%96=32
        if (leftover_s != 0):
            s = img_s + (C['stride_s'] - leftover_s)
        else:
            s = img_s
            # 32,25+(24-17)
        if (leftover_h != 0):
            h = img_h + (C['stride_h'] - leftover_h)
        else:
            h = img_h
            # 320,256+(96-32)
        if (leftover_w != 0):
            w = img_w + (C['stride_w'] - leftover_w)
        else:
            w = img_w
            # 320,256+(96-32)
        tmp_full_imgs = np.zeros((s, h, w))
        tmp_full_imgs[:img_s, :img_h, 0:img_w] = img
        # print("new images shape: \n" + str(img.shape))#我感觉是作者写错了，应该是tmp_full_imgs
        print("new images shape: \n" + str(tmp_full_imgs.shape))
        return tmp_full_imgs

    # Divide all the full_imgs in pacthes
    def extract_ordered_overlap(self, img, C):  # 把全部的完整3D数据切割成微小的patch数据，并且把一个3Dvolume变成一个个数据数组
        # 传进来的img是经过padding的
        assert (len(img.shape) == 3)  # 3D arrays
        img_s, img_h, img_w = img.shape
        # 32,320,320
        # cut_param = {'patch_s': 32,
        #              'patch_h': 128,
        #              'patch_w': 128,
        #              'stride_s': 24,
        #              'stride_h': 96,
        #              'stride_w': 96}
        assert ((img_h - C['patch_h']) % C['stride_h'] == 0
                and (img_w - C['patch_w']) % C['stride_w'] == 0
                and (img_s - C['patch_s']) % C['stride_s'] == 0)
        # 前面的好像是让他这里能整除？不过这是干什么
        N_patches_s = (img_s - C['patch_s']) // C['stride_s'] + 1
        # (32-32)/24+1=1
        N_patches_h = (img_h - C['patch_h']) // C['stride_h'] + 1
        # (320-128)/96+1=3
        N_patches_w = (img_w - C['patch_w']) // C['stride_w'] + 1
        # (320-128)/96+1=3
        N_patches_img = N_patches_s * N_patches_h * N_patches_w
        # 9
        print("Number of patches s/h/w(patched) : ", N_patches_s, N_patches_h, N_patches_w)
        print("number of patches per image: " + str(N_patches_img))
        patches = np.empty((N_patches_img, C['patch_s'], C['patch_h'], C['patch_w']))
        iter_tot = 0  # iter over the total number of patches (N_patches)
        for s in range(N_patches_s):  # loop over the full images
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    patch = img[s * C['stride_s']: s * C['stride_s'] + C['patch_s'],
                            h * C['stride_h']: h * C['stride_h'] + C['patch_h'],
                            w * C['stride_w']: w * C['stride_w'] + C['patch_w']]
                    # patch = [pathc_s,patch_h,patch_w]，一共N_patches_img个patch,每个patch之间是有重叠的
                    # cut_param = {'patch_s': 32,
                    #              'patch_h': 128,
                    #              'patch_w': 128,
                    #              'stride_s': 24,
                    #              'stride_h': 96,
                    #              'stride_w': 96}
                    patches[iter_tot] = patch
                    iter_tot += 1  # total
        assert (iter_tot == N_patches_img)
        return patches  # array with all the full_imgs divided in patches


class Recompone_tool():
    def __init__(self, save_path, filename, img_ori_shape, img_new_shape, C):
        # 这里的result好像是啥也没有？
        self.result = None
        self.save_path = save_path
        self.filename = filename
        self.ori_shape = img_ori_shape
        self.new_shape = img_new_shape
        self.C = C

    def add_result(self, tensor):  # reslut参数是通过这个东西传进来的
        # tensor = tensor.detach().cpu() # shape: [N,class,s,h,w]
        # tensor_np = np.squeeze(tensor_np,axis=0)
        if self.result is not None:
            self.result = torch.cat((self.result, tensor), dim=0)
            # ([2, 3, 32, 128, 128]) cat ([2, 3, 32, 128, 128]) cat ([2, 3, 32, 128, 128]) cat ([2, 3, 32, 128, 128])
            # = ([9, 3, 32, 128, 128])
        else:
            self.result = tensor

    def recompone_overlap(self):  # 网络输出的小块的结果，这个函数是把每一个数据分成batch后分割的结果进行拼接
        """
        :param preds: output of model  shape：[N_patchs_img,3,patch_s,patch_h,patch_w]
        :return: result of recompone output shape: [3,img_s,img_h,img_w]
        """
        patch_s = self.result.shape[2]
        patch_h = self.result.shape[3]
        patch_w = self.result.shape[4]
        N_patches_s = (self.new_shape[0] - patch_s) // self.C['stride_s'] + 1
        N_patches_h = (self.new_shape[1] - patch_h) // self.C['stride_h'] + 1
        N_patches_w = (self.new_shape[2] - patch_w) // self.C['stride_w'] + 1
        N_patches_img = N_patches_s * N_patches_h * N_patches_w
        print("N_patches_s/h/w:", N_patches_s, N_patches_h, N_patches_w)
        print("N_patches_img: " + str(N_patches_img))
        assert (self.result.shape[0] == N_patches_img)

        full_prob = torch.zeros((2, self.new_shape[0], self.new_shape[1],
                                 self.new_shape[2]))  # itialize to zero mega array with sum of Probabilities
        full_sum = torch.zeros((2, self.new_shape[0], self.new_shape[1], self.new_shape[2]))
        k = 0  # iterator over all the patches
        for s in range(N_patches_s):
            for h in range(N_patches_h):
                for w in range(N_patches_w):
                    full_prob[:, s * self.C['stride_s']:s * self.C['stride_s'] + patch_s,
                    h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                    w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += self.result[k]
                    full_sum[:, s * self.C['stride_s']:s * self.C['stride_s'] + patch_s,
                    h * self.C['stride_h']:h * self.C['stride_h'] + patch_h,
                    w * self.C['stride_w']:w * self.C['stride_w'] + patch_w] += 1
                    k += 1
        assert (k == self.result.size(0))
        assert (torch.min(full_sum) >= 1.0)  # at least one
        final_avg = full_prob / full_sum  # 重叠部分多算的概率会在这里被平均掉，重叠次数已经通过full_sum给出
        print('final_avg:', final_avg.size())
        assert (torch.max(final_avg) <= 1.0)  # max value for a pixel is 1.0
        assert (torch.min(final_avg) >= 0.0)  # min value for a pixel is 0.0
        img = final_avg[:, :self.ori_shape[0], :self.ori_shape[1], :self.ori_shape[2]]
        return img


def test(model, dataset, save_path, filename,device):
    dataloader = DataLoader(dataset=dataset, batch_size=4, num_workers=0, shuffle=False)
    model.eval()
    save_tool = Recompone_tool(save_path, filename, dataset.ori_shape, dataset.new_shape, dataset.cut)
    target = torch.from_numpy(np.expand_dims(dataset.label_np, axis=0)).long()
    target = to_one_hot_3d(target)
    with torch.no_grad():
        for data in dataloader:
            data = data.unsqueeze(1)
            data = data.float().to(device)
            output = model(data)
            save_tool.add_result(output.detach().cpu())

    pred = save_tool.recompone_overlap()
    pred = torch.unsqueeze(pred, dim=0)
    val_loss = metrics.DiceMeanLoss()(pred, target)
    val_dice0 = metrics.dice(pred, target, 0)
    val_dice1 = metrics.dice(pred, target, 1)

    pred_img = torch.argmax(pred, dim=1)
    pred_img = pred_img.transpose(1, 3)
    print(pred_img.shape, "=============================")
    img = sitk.GetImageFromArray(np.squeeze(np.array(pred_img.numpy(), dtype='uint8'), axis=0))
    sitk.WriteImage(img, os.path.join(save_path, filename))

    # save_tool.save(filename)
    print('\nAverage loss: {:.4f}\tdice0: {:.4f}\tdice1: {:.4f}\t'.format(
        val_loss, val_dice0, val_dice1))
    return val_loss, val_dice0, val_dice1


def inference(args, save_path,device,model):
    device = torch.device('cuda:0')
    ckpt = torch.load(os.path.join(save_path, 'best_model.pth'))
    model.load_state_dict(ckpt['net'])
    model=model.to(device)

    test_data_path = "problem1_nii_test"
    result_save_path = os.path.join(save_path, "seg_result")
    if not os.path.exists(result_save_path): os.mkdir(result_save_path)
    cut_param = {'patch_s': 80,
                 'patch_h': 112,
                 'patch_w': 112,
                 'stride_s': 60,
                 'stride_h': 84,
                 'stride_w': 84}
    datasets = test_Datasets(test_data_path, cut_param, resize_scale=1)
    for dataset, file_idx in datasets:
        test(model, dataset, result_save_path, file_idx.split('/')[1].replace('volume', 'predict'),device)


if __name__ == '__main__':
    args = config.args
