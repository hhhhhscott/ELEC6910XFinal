import argparse
import os

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--n_threads', type=int, default=6, help='number of threads for data loading')

parser.add_argument('--gpu', action='store_true', help='use gpu only')

parser.add_argument('--seed', type=int, default=1, help='random seed')

# data/dataset setting
parser.add_argument('--crop_size', type=list, default=[112, 112, 80], help='patch size of train samples after resize')

parser.add_argument('--batch_size', type=list, default=2, help='batch size of trainset')

# train setting
parser.add_argument('--net', type=str, default="proposed",
                    help='Specify the network (unet / resunet / r2unet / r2attunet / proposed)')

parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.0001, metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')

parser.add_argument('--early-stop', default=50, type=int, help='early stopping (default: 20)')
args = parser.parse_args()

if __name__ == '__main__':
    save_path = "result/"
    a = os.path.join(save_path, "best_model_lr_{}_bs_{}.pth".format(str(args.lr).replace(".", ""), args.batch_size))
