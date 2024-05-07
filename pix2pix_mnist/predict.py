import torch
import torchvision.transforms as transforms
from PIL import Image
from models import *
from torch.autograd import Variable
import argparse
import cv2
import numpy as np
import glob


transforms_ = [
    transforms.Resize((256, 256), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
]
transform = transforms.Compose(transforms_)
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default='./saved_models/mnist/generator_Uet_100.pth', help="inference model checkpoint")
parser.add_argument("--input", type=str, default='./mnist/test/0_4.jpg', help="image for predict")
parser.add_argument("--output", type=str, default="./results/0_4_out.jpg", help="predict result")
opt = parser.parse_args()

# load checkpoint
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
generator = GeneratorUNet()
if cuda:
    generator = generator.cuda()
generator.load_state_dict(torch.load(opt.checkpoint))
generator.eval()

#open image
for image_name in glob.glob('./mnist/test/*.jpg'):
    print(image_name)
    img = Image.open(image_name)
    print(img.size)
    w, h = img.size
    img = img.crop((0, 0, w / 2, h))
    img = transform(img)
    img = Variable(img.type(Tensor).unsqueeze(0))

    result = generator(img).squeeze(0).permute(1,2,0).detach().cpu().numpy() * 255
    result = np.clip(result,0,255)
    result_name = './results/' + image_name.split('\\')[-1]
    cv2.imwrite(result_name,result.astype(np.uint8))

