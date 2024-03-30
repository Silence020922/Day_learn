import torch.nn as nn
import torch
from PIL import Image
from torchvision import transforms
from model import ResNetGenerator

# 数据处理
preprocess = transforms.Compose([transforms.Resize(256), # 将最小的维度扩展到256
                                 transforms.ToTensor()])
Img_dir = "/home/surplus/Documents/python/Deep-learning/CycleGAN/data/horse.jpg"
Img = Image.open(Img_dir)
# Img.show()
Img_t = preprocess(Img)
batch_t = torch.unsqueeze(Img_t,0) # B C H W

# 模型加载
netG = ResNetGenerator()
model_path = 'Deep-learning/CycleGAN/data/horse2zebra_0.4.0.pth'
model_data = torch.load(model_path)
netG.load_state_dict(model_data)

# 推理
netG.eval()
batch_out = netG(batch_t)
print(batch_out)
out_t = (batch_out.data.squeeze()  + 1.0)  / 2.0
out_img = transforms.ToPILImage()(out_t)
out_img.show()

