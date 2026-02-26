import os
import torch
import tarfile
import torchvision
from torch import nn
from d2l import torch as d2l
import fcn_super_dataset as Super
from torch.nn import functional as F


data_dir = r'D:\data'
super_dir = os.path.join(data_dir, 'archive', 'supervisely_person_clean_2667_img')

pretrained_net = torchvision.models.resnet18(pretrained = True)
# print(list(pretrained_net.children())[-3:])
net = nn.Sequential(*list(pretrained_net.children())[:-2])
X = torch.rand(size=(1, 3, 320, 480))
num_classes = 2
net.add_module('first_cow', nn.Conv2d(512, 256, kernel_size=1))
net.add_module('relu_cow', nn.ReLU())
net.add_module('final_cow', nn.Conv2d(256, num_classes, kernel_size=1))
net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                    kernel_size=64, padding=16, stride=32))

def bilinear_kernel(in_channels, out_channels, kernel_size):
    factor = (kernel_size + 1) // 2
    if kernel_size % 2 == 1:
        center = factor - 1
    else:
        center = factor - 0.5
    og = (torch.arange(kernel_size).reshape(-1, 1),
          torch.arange(kernel_size).reshape(1, -1))
    filt = (1 - torch.abs(og[0] - center) / factor) * \
           (1 - torch.abs(og[1] - center) / factor)
    weight = torch.zeros((in_channels, out_channels,
                          kernel_size, kernel_size))
    weight[range(in_channels), range(out_channels), :, :] = filt
    return weight

conv_trans = nn.ConvTranspose2d(3, 3, kernel_size=4, padding=1, stride=2,
                                bias=False) 
conv_trans.weight.data.copy_(bilinear_kernel(3, 3, 4))
W = bilinear_kernel(num_classes, num_classes, 64)
net.transpose_conv.weight.data.copy_(W)
batch_size, crop_size = 32, (320, 320)
train_iter, test_iter = Super.load_data_SUPER(batch_size, crop_size)

def loss(inputs, targets):
    return F.cross_entropy(inputs, targets, reduction='none').mean(1).mean(1)

num_epochs, lr, wd, devices = 10, 0.0025, 1e-3, d2l.try_all_gpus()
trainer = torch.optim.SGD(net.parameters(), lr=lr, weight_decay=wd)
d2l.train_ch13(net, train_iter, test_iter, loss, trainer, num_epochs, devices)
d2l.plt.show()

def predict(img):
    X = test_iter.dataset.normalize_image(img).unsqueeze(0)
    pred = net(X.to(devices[0])).argmax(dim=1)
    return pred.reshape(pred.shape[1], pred.shape[2])

def label2image(pred):
    colormap = torch.tensor(Super.SUPER_COLORMAP, device=devices[0])
    X = pred.long()
    return colormap[X, :]

test_images, test_labels = Super.read_super_images(super_dir, False)
n, imgs = 4, []

for i in range(n):
    crop_rect = (0, 0, 320, 320)
    X_full = torchvision.io.read_image(test_images[i], mode=torchvision.io.image.ImageReadMode.RGB)
    Y_full = torchvision.io.read_image(test_labels[i], mode=torchvision.io.image.ImageReadMode.GRAY)
    X = torchvision.transforms.functional.crop(X_full, *crop_rect)
    Y = torchvision.transforms.functional.crop(Y_full, *crop_rect)
    
    pred = label2image(predict(X))
    imgs += [X.permute(1, 2, 0), pred.cpu(), Y.permute(1, 2, 0)]

d2l.show_images(imgs[::3] + imgs[1::3] + imgs[2::3], 3, n, scale=2)
d2l.plt.show()

print("saving model...")
torch.save(net.state_dict(), 'my_fcn2_model.pth')
print("save successfully")

import matplotlib.pyplot as plt
from PIL import Image
img_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406], 
        std=[0.229, 0.224, 0.225]
    )
])

def predict_my_photo(image_path, model, device):
    model.eval()
    
    original_img = Image.open(image_path).convert('RGB')
    
    X = img_transform(original_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(X)
        pred_mask = output.argmax(dim=1).squeeze(0)
    
    return original_img, pred_mask.cpu()

def mask_to_colormap(pred_mask, device):
    colormap = torch.tensor(Super.SUPER_COLORMAP, device=device)
    return colormap[pred_mask.long(), :]

def run_test(img_path):
    device = d2l.try_all_gpus()[0]
    
    print(f"processing: {img_path} ...")
    
    org_img, pred_mask = predict_my_photo(img_path, net, device)
    
    color_mask = mask_to_colormap(pred_mask.to(device), device).cpu().numpy()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(org_img)
    plt.title("Original Photo")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(color_mask)
    plt.title("Segmentation Result")
    plt.axis('off')
    
    plt.show()
    
    return org_img, pred_mask

my_image_path = r"D:\DeepLearningProject\hanhan.jpg" 
org_img, mask = run_test(my_image_path)

import numpy as np
import cv2  

def apply_portrait_mode(original_img, pred_mask):
    import cv2
    import numpy as np
    import matplotlib.pyplot as plt

    img_np = np.array(original_img)  
    mask_np = pred_mask.numpy()      

    h, w = img_np.shape[:2]
    if mask_np.shape != (h, w):
        mask_np = cv2.resize(mask_np.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)

    person_class_index = 1
    person_mask = (mask_np == person_class_index).astype(np.uint8)
    
    if person_mask.sum() == 0:
        print("no")
        plt.imshow(img_np)
        plt.show()
        return

    blurred_img = cv2.GaussianBlur(img_np, (51, 51), 0)
    
    mask_3d = person_mask[:, :, np.newaxis]
    
    final_img = img_np * mask_3d + blurred_img * (1 - mask_3d)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(final_img.astype(np.uint8)) 
    plt.title("Portrait Mode (Background Blur)")
    plt.axis('off')
    plt.show()

apply_portrait_mode(org_img, mask)