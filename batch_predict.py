"""
Uses MANIQA to predict the quality of a batch of images.
MANIQA by Yang et al, 2022 - https://github.com/IIGROUP/MANIQA
Adapted by: JeS24 for Tirtha.

"""
import os, random
import torch, cv2, numpy as np

from pathlib import Path
from torchvision import transforms

# Tirtha: Local imports - NOTE: Script won't work with dot-syntax imports
from .models.maniqa import MANIQA
from .config import Config


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class Normalize(object):
    def __init__(self, mean, var):
        self.mean = mean
        self.var = var

    def __call__(self, img):
        return (img - self.mean) / self.var


class Image(torch.utils.data.Dataset):
    def __init__(self, image_path, transform, num_crops=20):
        super(Image, self).__init__()
        self.img_name = image_path.split('/')[-1]
        self.img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        self.img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        new_h = new_w = 224
        top = np.random.randint(0, h - new_h, size=num_crops)
        left = np.random.randint(0, w - new_w, size=num_crops)
        patches = torch.empty((num_crops, c, new_h, new_w)) ## FloatTensor

        for i in range(num_crops):
            patches[i] = torch.from_numpy(self.img[:, top[i]: top[i] + new_h, left[i]: left[i] + new_w])
            if self.transform:
                patches[i] = self.transform(patches[i])
        self.img_patches = patches


class MANIQAScore:
    def __init__(self, ckpt_pth: str, cpu_num: int = 16, num_crops: int = 20, seed: int = 42):
        setup_seed(seed)
        os.environ['OMP_NUM_THREADS'] = str(cpu_num)
        os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
        os.environ['MKL_NUM_THREADS'] = str(cpu_num)
        os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
        os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
        torch.set_num_threads(cpu_num)

        ## Config
        self.config = config = Config({
            # CPU num
            "cpu_num": cpu_num,

            # valid times
            "num_crops": num_crops,

            # model
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.8,

            # checkpoint path
            # Tirtha: Passed from Tirtha
            "ckpt_path": str(Path(ckpt_pth).resolve().parent),
        })

        # model definition
        net = MANIQA(
            embed_dim=config.embed_dim,
            num_outputs=config.num_outputs,
            dim_mlp=config.dim_mlp,
            patch_size=config.patch_size,
            img_size=config.img_size,
            window_size=config.window_size,
            depths=config.depths,
            num_heads=config.num_heads,
            num_tab=config.num_tab,
            scale=config.scale
        )
        net.load_state_dict(torch.load(config.ckpt_path), strict=False)
        self.net = net.to('cuda')
        self.net.eval()

    def predict_patch(self, patch_sample):
        with torch.no_grad():
            patch = patch_sample.to('cuda')
            patch = patch.unsqueeze(0)
            score = self.net(patch)
            return score

    def predict_one(self, img_path: str) -> float:
        img = Image(
            image_path=img_path,
            transform=transforms.Compose([Normalize(0.5, 0.5)]),
            num_crops=self.config.num_crops
        )

        score = 0
        for patch in img.img_patches:
            score += self.predict_patch(patch)
        avg_score = sum(score) / self.config.num_crops

        return avg_score

# NOTE: Code below won't work with dot-syntax imports
if __name__ == '__main__':
    # Create a MANIQAScore object
    # CHANGEME:
    manr = MANIQAScore(ckpt_pth="<PATH/TO/CKPT/HERE", cpu_num=16, num_crops=20)

    # Create a list of images
    img_root = Path("../../iops_data_test/")
    imgs = [str(img) for img in img_root.glob('*.jpg')]

    # Predict the quality of each image
    scores = dict()
    for img in imgs:
        scores[img] = manr.predict_one(img).detach().cpu().numpy()
    print(scores)
