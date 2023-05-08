import torch


import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch import optim
from torch.utils.data import DataLoader
from utils import load_checkpoint, save_images, save_checkpoint, SegmDataset
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
                xs = (((x.clamp(-1, 1) + 1) / 2) * 255).type(torch.uint8)
                save_images(xs, os.path.join("results/steps", f"{i}_fake.png"))
        model.train()
        x = (x.clamp(-1, 1) + 1) / 2
        x = (x * 255).type(torch.uint8)

        labels = (labels.clamp(-1, 1) + 1) / 2
        labels = (labels * 255).type(torch.uint8)

        # images = (images.clamp(-1, 1) + 1) / 2
        # images = (images * 255).type(torch.uint8)

        save_images(x, os.path.join("results", f"{counter}_fake.png"))
        save_images(labels, os.path.join("results", f"{counter}_label.png"))
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
            save_images(lab, os.path.join("results/experiment 5 seg2img", "labels", f"{counter}_label.png"))
            counter += 1
        return counter


def train(args, training):
    global counter
    device = args.device
    real_paths = glob("datasets/generated/real/*.png")
    fake_paths = glob("datasets/generated/fake/*.png")
    valid_path = glob("datasets/generated/validation/*.png")
    dataset = SegmDataset(data_paths=real_paths, label_paths=fake_paths, img_size=args.image_size)
    validation_dataset = SegmDataset(data_paths=real_paths, label_paths=valid_path, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    validation_dataloader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False,
                                       num_workers=1)

    disc = Discriminator(in_channels=1).to(device)
    optimizer = optim.AdamW(disc.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, "disc8.pth.tar"), disc, optimizer, args.lr,
        )

    if training is False:
        n1 = 0
        n2 = 0
        pbar = tqdm(validation_dataloader)
        for i, (_, fake) in enumerate(pbar):
            fake = fake.to(device)

            D_fake = disc(fake)

            if torch.mean(D_fake).item() > 0.05:
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
    else:
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

                D_loss_real = mse(D_real, torch.ones_like(D_real))
                D_loss_fake = mse(D_fake, torch.zeros_like(D_fake))

                loss = D_loss_real + D_loss_fake

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                avg_loss += loss.item()
                pbar.set_postfix(epoch=epoch, AVG_LOSS=avg_loss / (i + 1), MIN_LOSS=min_avg_loss)

                if i % ((len(dataloader) - 1) // 2) == 0 and i != 0:
                    counter += 1

            n1 = 0
            n2 = 0
            avg_value = 0
            pbar = tqdm(validation_dataloader)
            for i, (_, fake) in enumerate(pbar):
                fake = fake.to(device)

                D_fake = disc(fake)

                if torch.mean(D_fake).item() > 0.05:
                    fake = (fake.clamp(-1, 1) + 1) / 2
                    fake = (fake * 255).type(torch.uint8)
                    save_images(fake, f"datasets/generated/good/{i}_image.png")
                    n1 += 1
                else:
                    fake = (fake.clamp(-1, 1) + 1) / 2
                    fake = (fake * 255).type(torch.uint8)
                    save_images(fake, f"datasets/generated/bad/{i}_image.png")
                    n2 += 1

                avg_value += torch.mean(D_fake).item()
                pbar.set_postfix(n1=n1, n2=n2, mean=round(avg_value/(i + 1), 3))

            if min_avg_loss > avg_loss / len(dataloader):
                min_avg_loss = avg_loss / len(dataloader)

            save_checkpoint(disc, optimizer, filename=os.path.join(args.checkpoints, f"disc{epoch}.pth.tar"))


if __name__ == '__main__':
    training = True
    counter = 1

    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    args.load_model = False
    args.epochs = 500
    args.batch_size = 1
    args.image_size = 64 * 1
    args.num_workers = 4
    args.checkpoints = "results/experiment 7 noise2seg/checkpoints"
    args.dataset_path = None
    args.generated_path = None
    args.device = "cuda"
    args.lr = 1e-5
    train(args, training)
