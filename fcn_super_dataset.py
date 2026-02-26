import os
import torch
import tarfile
import torchvision
from torch import nn
from d2l import torch as d2l
from torch.nn import functional as F


data_dir = r'D:\data'
super_dir = os.path.join(data_dir, 'archive', 'supervisely_person_clean_2667_img')

def read_super_images(super_dir, is_train=True):

    img_root = os.path.join(super_dir, 'images')
    mask_root = os.path.join(super_dir, 'masks')

    images = sorted([x for x in os.listdir(img_root) if x.endswith('.jpg') or x.endswith('.png')])
    masks = sorted([x for x in os.listdir(mask_root) if x.endswith('.png')])

    val_idx = int(len(images) * 0.9)
    if is_train:
        images_names = images[:val_idx]
        mask_names = masks[:val_idx]
    else:
        images_names = images[val_idx:]
        mask_names = masks[val_idx:]
    
    features = [os.path.join(img_root, fname) for fname in images_names]
    labels = [os.path.join(mask_root, fname) for fname in mask_names]

    return features, labels

train_features, train_labels = read_super_images(super_dir, True)
n = 5
modeRGB = torchvision.io.ImageReadMode.RGB
modeGRAY = torchvision.io.ImageReadMode.GRAY
# imgs = []
# for i in range(n):
#     feature_img = torchvision.io.read_image(train_features[i], mode=torchvision.io.ImageReadMode.RGB)
#     label_img = torchvision.io.read_image(train_labels[i], mode=torchvision.io.ImageReadMode.GRAY)
#     imgs.append(feature_img)
#     imgs.append(label_img)

# imgs = [img.permute(1, 2, 0) for img in imgs]

# d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
# d2l.plt.show()

SUPER_COLORMAP = [[0, 0, 0], [128, 0, 0]]

SUPER_CLASSES = ['background', 'person']


def super_rand_crop(feature, label, height, width):
    rect = torchvision.transforms.RandomCrop.get_params(
        feature, (height, width)
    )
    feature = torchvision.transforms.functional.crop(feature, *rect)
    label = torchvision.transforms.functional.crop(label, *rect)
    return feature, label

# imgs = []
# for _ in range(n):
#     feature_img = torchvision.io.read_image(train_features[i], mode=torchvision.io.ImageReadMode.RGB)
#     label_img = torchvision.io.read_image(train_labels[i], mode=torchvision.io.ImageReadMode.GRAY)
#     imgs += super_rand_crop(feature_img, label_img, 200, 300)
#     imgs.append(feature_img)
#     imgs.append(label_img)

# imgs = [img.permute(1, 2, 0) for img in imgs]
# d2l.show_images(imgs[::2] + imgs[1::2], 2, n)
# d2l.plt.show()

class SUPERSegDataset(torch.utils.data.Dataset):

    def __init__(self, is_train, crop_size, super_dir):
        self.transform = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.crop_size = crop_size
        features, labels = read_super_images(super_dir, is_train=is_train)
        self.features = features
        self.labels = labels
        print('read ' + str(len(self.features)) + ' examples')

    def normalize_image(self, img):
        return self.transform(img.float() / 255)

    def filter(self, imgs):
        return imgs

    def __getitem__(self, idx):
        feature, label = (torchvision.io.read_image(self.features[idx], modeRGB),
                          torchvision.io.read_image(self.labels[idx], modeGRAY))
        feature, label = super_rand_crop(feature, label,
                                       *self.crop_size)
        label_idx = (label > 128).long().squeeze(0)
        return self.normalize_image(feature), label_idx

    def __len__(self):
        return len(self.features)

def load_data_SUPER(batch_size, crop_size):
    num_workers = d2l.get_dataloader_workers()
    train_iter = torch.utils.data.DataLoader(
        SUPERSegDataset(True, crop_size, super_dir), batch_size,
        shuffle=True, drop_last=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(
        SUPERSegDataset(False, crop_size, super_dir), batch_size,
        drop_last=True, num_workers=num_workers)
    return train_iter,test_iter