import os
import random

import torch
import torchvision
import albumentations as A
import numpy as np
import torchvision.transforms.functional as TF
from torch import nn
from tqdm import tqdm
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.models import vgg19
from albumentations.pytorch import ToTensorV2

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


class VGGLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = vgg19(pretrained=True).features[:36].eval().to(torch.device("cuda"))
        self.loss = nn.MSELoss()

        for param in self.vgg.parameters():
            param.requires_grad = False

    def forward(self, input, target):
        vgg_input_features = self.vgg(input)
        vgg_target_features = self.vgg(target)
        return self.loss(vgg_input_features, vgg_target_features)


class DDPMDataset(Dataset):
    def __init__(self, data_paths, label_paths, img_size):
        self.data_paths = data_paths
        self.label_paths = label_paths

        self.transform1 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size, interpolation=torchvision.transforms.InterpolationMode.BICUBIC),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5)
        ])

    def __getitem__(self, index):
        x = Image.open(self.data_paths[index])
        if self.transform1:
            x = self.transform1(x)

        y = Image.open(self.label_paths[index]).convert('L')
        if self.transform2:
            y = self.transform2(y)

        if random.random() < 0.5:
            x, y = TF.hflip(x), TF.hflip(y)

        if random.random() > 0.5:
            x, y = TF.vflip(x), TF.vflip(y)

        return x, y

    def __len__(self):
        return len(self.data_paths) if len(self.data_paths) < len(self.label_paths) else len(self.label_paths)


class MaskDataset(Dataset):
    def __init__(self, data_paths, label_paths, img_size):
        self.data_paths = data_paths
        self.label_paths = label_paths

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        x = Image.open(self.data_paths[index])
        x = self.transform(x)

        y = Image.open(self.label_paths[index])
        y = self.transform(y)

        if random.random() < 0.5:
            x, y = TF.hflip(x), TF.hflip(y)

        if random.random() > 0.5:
            x, y = TF.vflip(x), TF.vflip(y)

        return x, y

    def __len__(self):
        return len(self.data_paths) if len(self.data_paths) < len(self.label_paths) else len(self.label_paths)


class SegmDataset(Dataset):
    def __init__(self, data_paths, label_paths, img_size):
        self.data_paths = data_paths
        self.label_paths = label_paths

        self.transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(0.5, 0.5,)
        ])

    def __getitem__(self, index):
        x = Image.open(self.data_paths[index]).convert('L')
        x = self.transform(x)

        y = Image.open(self.label_paths[index]).convert('L')
        y = self.transform(y)

        if random.random() < 0.5:
            x, y = TF.hflip(x), TF.hflip(y)

        if random.random() > 0.5:
            x, y = TF.vflip(x), TF.vflip(y)

        return x, y

    def __len__(self):
        return len(self.data_paths) if len(self.data_paths) < len(self.label_paths) else len(self.label_paths)


class GANDataset(Dataset):
    def __init__(self, data1_paths, data2_paths, img_size):
        self.data1_paths = data1_paths
        self.data2_paths = data2_paths

        self.transform1 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size*4),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.transform2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(img_size),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __getitem__(self, index):
        x = Image.open(self.data1_paths[index])
        if self.transform1:
            x = self.transform1(x)

        y = Image.open(self.data2_paths[index])
        if self.transform2:
            y = self.transform2(y)

        if random.random() < 0.5:
            x, y = TF.hflip(x), TF.hflip(y)

        if random.random() > 0.5:
            x, y = TF.vflip(x), TF.vflip(y)

        return x, y

    def __len__(self):
        return len(self.data1_paths) if len(self.data1_paths) < len(self.data2_paths) else len(self.data2_paths)


class ABDataset(Dataset):
    def __init__(self, root_lr, root_hr):
        self.root_lr = root_lr
        self.root_hr = root_hr

        self.lr_images = os.listdir(root_lr)
        self.hr_images = os.listdir(root_hr)
        self.lr_len = len(self.lr_images)
        self.hr_len = len(self.hr_images)
        self.length_dataset = max(self.lr_len, self.hr_len)

        self.lr_transforms = A.Compose(
            [
                A.Resize(width=64, height=64, interpolation=Image.BICUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ])

        self.hr_transforms = A.Compose(
            [
                A.Resize(width=512, height=512, interpolation=Image.BICUBIC),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ])

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        lr_img = self.lr_images[index % self.lr_len]
        lr_path = os.path.join(self.root_lr, lr_img)
        lr_img = np.array(Image.open(lr_path).convert("RGB"))

        hr_img = self.hr_images[index % self.hr_len]
        hr_path = os.path.join(self.root_hr, hr_img)
        hr_img = np.array(Image.open(hr_path).convert("RGB"))

        lr_img = self.lr_transforms(image=lr_img)["image"]
        hr_img = self.hr_transforms(image=hr_img)["image"]

        return lr_img, hr_img


class ABCDDataset(Dataset):
    def __init__(self, root_lr_img, root_lr_msk, root_hr_img, root_hr_msk):
        self.root_lr_img = root_lr_img
        self.root_lr_msk = root_lr_msk
        self.root_hr_img = root_hr_img
        self.root_hr_msk = root_hr_msk

        self.lr_images = os.listdir(root_lr_img)
        self.lr_masks = os.listdir(root_lr_msk)
        self.hr_images = os.listdir(root_hr_img)
        self.hr_masks = os.listdir(root_hr_msk)
        self.lr_len = max(len(self.lr_images), len(self.lr_masks))
        self.hr_len = max(len(self.hr_images), len(self.hr_masks))
        self.length_dataset = max(self.lr_len, self.hr_len)

        self.lr_img_transforms = A.Compose(
            [
                A.Resize(width=64, height=64, interpolation=Image.BICUBIC),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ])

        self.hr_img_transforms = A.Compose(
            [
                A.Resize(width=512, height=512, interpolation=Image.BICUBIC),
                A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255),
                ToTensorV2(),
            ])

        self.lr_msk_transforms = A.Compose(
            [
                A.Resize(width=64, height=64, interpolation=Image.BICUBIC),
                A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
                ToTensorV2(),
            ])

        self.hr_msk_transforms = A.Compose(
            [
                A.Resize(width=512, height=512, interpolation=Image.BICUBIC),
                A.Normalize(mean=0.5, std=0.5, max_pixel_value=255),
                ToTensorV2(),
            ])

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        lr_img = self.lr_images[index % self.lr_len]
        lr_img = os.path.join(self.root_lr_img, lr_img)
        lr_img = np.array(Image.open(lr_img).convert("RGB"))

        lr_msk = self.lr_images[index % self.lr_len]
        lr_msk = os.path.join(self.root_lr_msk, lr_msk)
        lr_msk = np.array(Image.open(lr_msk).convert("L"))

        hr_img = self.hr_images[index % self.hr_len]
        hr_img = os.path.join(self.root_hr_img, hr_img)
        hr_img = np.array(Image.open(hr_img).convert("RGB"))

        hr_msk = self.hr_images[index % self.hr_len]
        hr_msk = os.path.join(self.root_hr_msk, hr_msk)
        hr_msk = np.array(Image.open(hr_msk).convert("L"))

        lr_img = self.lr_img_transforms(image=lr_img)["image"]
        hr_img = self.hr_img_transforms(image=hr_img)["image"]

        lr_msk = self.lr_msk_transforms(image=lr_msk)["image"]
        hr_msk = self.hr_msk_transforms(image=hr_msk)["image"]

        return lr_img, lr_msk, hr_img, hr_msk


def gradient_penalty(critic, real, fake, device):
    BATCH_SIZE, C, H, W = real.shape
    alpha = torch.rand((BATCH_SIZE, 1, 1, 1)).repeat(1, C, H, W).to(device)
    interpolated_images = real * alpha + fake.detach() * (1 - alpha)
    interpolated_images.requires_grad_(True)
    mixed_scores = critic(interpolated_images)

    gradient = torch.autograd.grad(
        inputs=interpolated_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    # print(gradient.shape)
    gradient = gradient.reshape(gradient.shape[0], -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = torch.mean((gradient_norm - 1) ** 2)
    return gradient_penalty


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device
        self.fixed_noise = torch.randn(1, 3, img_size, img_size).to(device)

        self.beta = self.prepare_linear_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_linear_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        eps = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x

    def generate(self, model, n):
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size)).to(self.device)
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                else:
                    noise = torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (
                        x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(
                    beta) * noise
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)
        return x


def plot_images(images):
    plt.figure(figsize=(32, 32))
    plt.imshow(torch.cat([
        torch.cat([i for i in images.cpu()], dim=-1),
    ], dim=-2).permute(1, 2, 0).cpu())
    plt.show()


def save_images(images, path, **kwargs):
    grid = torchvision.utils.make_grid(images, **kwargs)
    ndarr = grid.permute(1, 2, 0).to('cpu').numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


def get_data(args):
    transforms1 = torchvision.transforms.Compose([
        torchvision.transforms.Resize(args.hr_image_size),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset1 = torchvision.datasets.ImageFolder(args.dataset_path, transform=transforms1)
    dataloader1 = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    if args.generated_path is not None:
        transforms2 = torchvision.transforms.Compose([
            torchvision.transforms.Resize(args.image_size),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomVerticalFlip(),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        dataset2 = torchvision.datasets.ImageFolder(args.generated_path, transform=transforms2)
        dataloader2 = DataLoader(dataset2, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        return dataloader1, dataloader2

    return dataloader1


def setup_logging(run_name):
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=torch.device("cuda"))
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
