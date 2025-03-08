import torch, os, cv2
from utils.common import merge_config, get_model
import tqdm
import torchvision.transforms as transforms
from utils.common import get_test_loader
from utils.eval import eval_lane

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True

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
    eval_lane(net, cfg)