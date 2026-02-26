import torch
import torchvision
from torch import nn
from d2l import torch as d2l
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2 

def get_model():
    num_classes = 2
    
    pretrained_net = torchvision.models.resnet18(weights=None) 
    net = nn.Sequential(*list(pretrained_net.children())[:-2])
    
    net.add_module('first_cow', nn.Conv2d(512, 256, kernel_size=1))
    net.add_module('relu_cow', nn.ReLU())
    net.add_module('final_cow', nn.Conv2d(256, num_classes, kernel_size=1))
    net.add_module('transpose_conv', nn.ConvTranspose2d(num_classes, num_classes,
                                                        kernel_size=64, padding=16, stride=32))
    return net

def load_my_model(model_path, device):
    print(f"正在加载模型: {model_path} ...")
    
    net = get_model()
    
    checkpoint = torch.load(model_path, map_location=device)
    
    net.load_state_dict(checkpoint)
    
    net = net.to(device)
    net.eval()  
    
    print("模型加载成功！")
    return net

def predict_and_blur(net, img_path, device):
    img_transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    original_img = Image.open(img_path).convert('RGB')
    w, h = original_img.size
    
    scale = 640 / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    img_resized = original_img.resize((new_w, new_h))
    
    X = img_transform(img_resized).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = net(X)
        output = torch.nn.functional.interpolate(output, size=(h, w), mode='bilinear', align_corners=False)
        pred_mask = output.argmax(dim=1).squeeze(0).cpu().numpy()
        
    person_mask = (pred_mask == 1).astype(np.uint8)
    
    if person_mask.sum() < 100:
        print("没检测到人")
        plt.imshow(original_img)
        plt.show()
        return

    img_np = np.array(original_img)
    blurred = cv2.GaussianBlur(img_np, (51, 51), 0) # 模糊力度
    mask_3d = person_mask[:, :, np.newaxis]
    final_img = img_np * mask_3d + blurred * (1 - mask_3d)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(final_img.astype(np.uint8))
    plt.axis('off')
    plt.title("Loaded Model Result")
    plt.show()

if __name__ == '__main__':
    model_path = r'D:\DeepLearningProject\fcn\my_fcn2_model.pth' 
    
    test_img = r"D:\DeepLearningProject\fcn\hanhan.jpg"
    
    device = d2l.try_all_gpus()[0]
    
    net = load_my_model(model_path, device)
    
    predict_and_blur(net, test_img, device)