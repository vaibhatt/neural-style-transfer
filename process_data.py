import torchvision
import torch.nn as nn
import torch
import torchvision.transforms as transforms
from PIL import Image
import config


class data_preprocess(nn.Module):
    def __init__(self, resize = None, transforms  = None):
        super(data_preprocess, self).__init__()
        self.resize = resize
        self.transforms = transforms

    def forward(self, content_image_path, style_image_path):
        content = Image.open(content_image_path)
        style = Image.open(style_image_path)
        if self.resize is not None:
            content = content.resize(self.resize)
            style = style.resize(self.resize)
        
        if self.transforms is not None:
            content = self.transforms(content).unsqueeze(0)
            style = self.transforms(style).unsqueeze(0)


        return {
            "content" : content.to(config.device, torch.float),
            "style" : style.to(config.device, torch.float)
        }

transform = transforms.Compose([
    transforms.ToTensor()
])



            

