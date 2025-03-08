import numpy as np
import torch, os
from utils.common import merge_config, get_model
import cv2
from data.dali_data import get_image_results
import matplotlib.pyplot as plt
from torchvision import transforms

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    image_path="images/test.jpg"

    args, cfg = merge_config()
    net = get_model(cfg)
    state_dict = torch.load(cfg.test_model, map_location='cpu')
    if 'model' in state_dict:
        state_dict = state_dict['model']
    compatible_state_dict = {}
    for k, v in state_dict.items():
        if 'module.' in k:
            compatible_state_dict[k[7:]] = v
        else:
            compatible_state_dict[k] = v

    net.load_state_dict(compatible_state_dict, strict = True)
    net.eval() 

    image = cv2.imread(image_path)
    image = cv2.resize(image, (cfg.train_width, cfg.train_height))
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Mean và Std chuẩn của ImageNet
])
    image_tensor = transform(image).unsqueeze(0).cuda()
    # print(image_tensor)
    out_seg = net(image_tensor)
    out_seg = out_seg[0].argmax(dim=0).cpu().numpy()
    # print(out_seg.shape)
    im_seg = get_image_results(image, out_seg, cfg.train_height, cfg.train_width)
    # plt.imshow(im_seg)
    # plt.show()
    cv2.imwrite("/content/KhoaLuanTotNghiep/Cascade-LD/images/show.jpg", np.array(im_seg))
    