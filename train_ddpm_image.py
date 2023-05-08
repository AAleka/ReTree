import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch import optim
from torch.utils.data import DataLoader
from utils import load_checkpoint, save_images, save_checkpoint, DDPMDataset, MaskDataset
from DDPM_model import DDPM, Discriminator
torch.manual_seed(1)


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
        return torch.abs(torch.cos(torch.linspace(0, torch.pi / 2, self.noise_steps)) * self.beta_end -
                         (self.beta_end - self.beta_start))
        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t):  # self, x, labels, t
        # labels = labels[:x.shape[0]]
        # labels = labels.expand(x.shape[0], *labels.shape[1:])
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        eps = torch.randn_like(x)
        # eps[labels > 0] = x[labels > 0]
        img = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        # img[labels > 0] = x[labels > 0]
        return img, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, labels, n):  # self, model, images, labels, n
        # images = images[:n]
        labels = labels[:n]
        # labels = labels.expand(n, *labels.shape[1:])
        model.eval()
        with torch.no_grad():
            x = torch.randn((labels.shape[0], 3, self.img_size, self.img_size)).to(self.device)
            # x[labels > 0] = labels[labels > 0]  #
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(labels.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                    # noise[labels > 0] = labels[labels > 0]  #
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
                # x[labels > 0] = labels[labels > 0]  #
                # xs = (((x.clamp(-1, 1) + 1) / 2) * 255).type(torch.uint8)
                # save_images(xs, os.path.join("results/steps", f"{i}_fake.png"))
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        labels = (labels.clamp(-1, 1) + 1) / 2
        labels = (labels * 255).type(torch.uint8)

        # images = (images.clamp(-1, 1) + 1) / 2
        # images = (images * 255).type(torch.uint8)

        save_images(x, os.path.join("results", f"experiment 5 seg2img/{counter}_fake.png"))
        save_images(labels, os.path.join("results", f"experiment 5 seg2img/{counter}_label.png"))
        # save_images(images, os.path.join("results", f"{counter}_real.png"))

    def generate(self, model, labels, n, counter):  # self, model, images, labels, n, counter
        # images = images[:n]
        labels = labels[:n]
        # labels = labels.expand(n, *labels.shape[1:])
        model.eval()
        with torch.no_grad():
            x = torch.randn((labels.shape[0], 3, self.img_size, self.img_size)).to(self.device)
            # xs[labels > 0] = images[labels > 0]  # images[labels > 0]
            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = (torch.ones(labels.shape[0]) * i).long().to(self.device)
                predicted_noise = model(x, t, labels)
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                if i > 1:
                    noise = torch.randn_like(x)
                    # noise[labels > 0] = images[labels > 0]  # images[labels > 0]
                else:
                    noise = torch.zeros_like(x)
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
                # x[labels > 0] = images[labels > 0]  # images[labels > 0]

        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        # labels[labels > 0] = images[labels > 0]
        labels = (labels.clamp(-1, 1) + 1) / 2
        labels = (labels * 255).type(torch.uint8)

        for img, lab in zip(x, labels):
            save_images(img, os.path.join("results/experiment 5 seg2img", "images", f"{counter}_image.png"))
            save_images(lab, os.path.join("results/experiment 5 seg2img", "labels", f"{counter}_image.png"))
            counter += 1
        return counter


def train_classifier(args):
    global counter
    device = args.device
    real_paths = glob("datasets/generated/real/*.png")
    fake_paths = glob("datasets/generated/fake/*.png")
    dataset = MaskDataset(data_paths=real_paths, label_paths=fake_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=args.num_workers)

    disc = Discriminator(in_channels=3).to(device)
    optimizer = optim.AdamW(disc.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, "disc3.pth.tar"), disc, optimizer, args.lr,
        )
        n1 = 0
        n2 = 0
        pbar = tqdm(dataloader)
        for i, (_, fake) in enumerate(pbar):
            fake = fake.to(device)

            D_fake = disc(fake)

            if torch.mean(D_fake).item() > 0.1:
                fake = (fake.clamp(-1, 1) + 1) / 2
                fake = (fake * 255).type(torch.uint8)
                save_images(fake, f"datasets/generated/good/{i}_image.png")
                n1 += 1
            else:
                fake = (fake.clamp(-1, 1) + 1) / 2
                fake = (fake * 255).type(torch.uint8)
                save_images(fake, f"datasets/generated/bad/{i}_image.png")
                n2 += 1

            pbar.set_postfix(n1=n1, n2=n2, mean=torch.mean(D_fake).item())

    mse = nn.MSELoss()
    min_avg_loss = float("inf")

    for epoch in range(1, args.epochs):
        pbar = tqdm(dataloader)
        avg_loss = 0
        for i, (real, fake) in enumerate(pbar):
            real = real.to(device)
            fake = fake.to(device)

            D_real = disc(real)
            D_fake = disc(fake)

            D_loss_real = mse(D_real, torch.ones_like(D_real) - 0.001*torch.randn_like(D_real))
            D_loss_fake = mse(D_fake, torch.zeros_like(D_fake))

            loss = D_loss_real + D_loss_fake

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()
            pbar.set_postfix(epoch=epoch, AVG_LOSS=avg_loss / (i + 1), MIN_LOSS=min_avg_loss)

            if i % ((len(dataloader) - 1) // 2) == 0 and i != 0:
                counter += 1

        if min_avg_loss > avg_loss / len(dataloader):
            min_avg_loss = avg_loss / len(dataloader)
            save_checkpoint(disc, optimizer, filename=os.path.join(args.checkpoints, f"disc{epoch}.pth.tar"))


def test(args):
    global counter
    device = args.device
    data_paths = sorted(glob("datasets/eyeq/images/*.png"), key=len)
    label_paths = sorted(glob("results/experiment 7 noise2seg/images/good/*.png"), key=len)
    dataset = DDPMDataset(data_paths=data_paths, label_paths=label_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    pbar = tqdm(dataloader)

    ddpm = DDPM(time_dim=args.emb_dim).to(device)
    diffusion = Diffusion(noise_steps=1000, img_size=args.image_size, device=device)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoint, "ddpm25.pth.tar"), ddpm, None, None,
        )

    for i in range(test_args.num_iters):
        for j, (images, labels) in enumerate(pbar):
            # images = images.to(device)
            labels = labels.to(device)
            counter = diffusion.generate(ddpm, labels, args.batch_size, counter)


def train(args):
    global counter
    device = args.device
    data_paths = glob("datasets/segmentation/image/*.png")
    label_paths = glob("datasets/segmentation/mask/*.png")
    dataset = DDPMDataset(data_paths=data_paths, label_paths=label_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ddpm = DDPM(img_channels=4, time_dim=args.emb_dim).to(device)
    optimizer = optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoint, "ddpm3.pth.tar"), ddpm, optimizer, args.lr,
        )

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    min_avg_loss = float("inf")

    diffusion = Diffusion(noise_steps=1000, img_size=args.image_size, device=device)

    for epoch in range(1, args.epochs):
        pbar = tqdm(dataloader)
        avg_loss = 0
        count1 = 0
        count2 = 0
        for i, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)

            t = diffusion.sample_timesteps(images.shape[0]).to(device)

            x_t, noise = diffusion.noise_images(images, t)

            for rep in range(5):
                if rep == 1:
                    count1 += 1
                predicted_noise = ddpm(x_t, t, labels)
                loss = mse(noise, predicted_noise) + l1(noise, predicted_noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if loss < min_avg_loss:
                    if rep == 4:
                        count2 += 1
                    break

            avg_loss += loss.item()
            pbar.set_postfix(epoch=epoch, AVG_MSE=avg_loss / (i+1), count1=count1, count2=count2, MIN_MSE=min_avg_loss)

            if i % ((len(dataloader)-1)//2) == 0 and i != 0:
                # images = (images.clamp(-1, 1) + 1) / 2
                # images = (images * 255).type(torch.uint8)

                # save_images(images, os.path.join("results", f"experiment 5 seg2img/{counter}_real.png"))

                diffusion.sample(ddpm, labels, n=8)
                counter += 1

        if min_avg_loss > avg_loss / len(dataloader):
            min_avg_loss = avg_loss / len(dataloader)
            save_checkpoint(ddpm, optimizer, filename=os.path.join(args.checkpoints, f"ddpm{epoch}.pth.tar"))


if __name__ == '__main__':
    training = False
    counter = 18869
    if training:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.load_model = False
        args.epochs = 500
        args.batch_size = 1
        args.emb_dim = 256 * 1
        args.image_size = 64 * 1
        args.num_workers = 4
        args.checkpoints = "results/experiment 5 seg2img/checkpoints"
        args.dataset_path = None
        args.generated_path = None
        args.device = "cuda"
        args.lr = 3e-4
        train(args)
        # train_classifier(args)
    else:
        test_parser = argparse.ArgumentParser()
        test_args = test_parser.parse_args()
        test_args.load_model = True
        test_args.emb_dim = 256 * 1
        test_args.num_iters = 1000
        test_args.batch_size = 64
        test_args.image_size = 64 * 1
        test_args.num_workers = 4
        test_args.checkpoint = "results/experiment 5 seg2img/checkpoints"
        test_args.device = "cuda"
        test(test_args)
