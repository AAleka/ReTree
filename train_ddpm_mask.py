import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from glob import glob
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import load_checkpoint, save_images, save_checkpoint, DDPMDataset, MaskDataset
from DDPM_model import DDPM, Discriminator

torch.manual_seed(9)


class Diffusion:
    def __init__(self, noise_steps=1000, beta_start=1e-4, beta_end=0.02, img_size=256, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_linear_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def prepare_linear_noise_schedule(self):
        return torch.abs(torch.cos(torch.linspace(0, torch.pi/2, self.noise_steps))*self.beta_end -
                         (self.beta_end-self.beta_start))
        # return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, labels, t):
        labels = labels[:labels.shape[0]]

        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]

        eps = torch.randn_like(labels)
        img = sqrt_alpha_hat * labels + sqrt_one_minus_alpha_hat * eps
        return img, eps

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def sample(self, model, disc, epoch, n):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
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
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
                # xs = ((x.clamp(-1, 1) + 1) / 2)
                # xs[xs >= 0.25] = torch.ones_like(xs[xs >= 0.25])
                # xs[xs < 0.25] = torch.zeros_like(xs[xs < 0.25])
                # xs = (xs * 255).type(torch.uint8)
                # save_images(xs, os.path.join("results/steps", f"{i}_fake.png"))
        model.train()

        # score = disc(torch.clone(x.clamp(-1, 1)))

        x = (x.clamp(-1, 1) + 1) / 2
        # x[x >= 0.25] = torch.ones_like(x[x >= 0.25])
        # x[x < 0.25] = torch.zeros_like(x[x < 0.25])
        x = (x * 255).type(torch.uint8)
        save_images(x, os.path.join("results/experiment 7 noise2seg", f"{counter}_{epoch}_image.png"))
        # return score

    def generate(self, model, disc, n, counter1, counter2):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
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
                x = 1/torch.sqrt(alpha)*(x-((1-alpha)/(torch.sqrt(1-alpha_hat)))*predicted_noise)+torch.sqrt(beta)*noise
        model.train()
        x = x.clamp(-1, 1)
        # x[x >= 0.25] = torch.ones_like(x[x >= 0.25])
        # x[x < 0.25] = torch.zeros_like(x[x < 0.25])
        # x = (x * 255).type(torch.uint8)
        avg_score = 0
        for img in x:
            img = img[None, :]
            score = torch.mean(disc(img)).item()
            if score > 0.8:
                img = (img + 1) / 2
                img = (img * 255).type(torch.uint8)
                save_images(img, os.path.join(test_args.image_path, f"good/{counter1}_image.png"))
                counter1 += 1
            else:
                img = (img + 1) / 2
                img = (img * 255).type(torch.uint8)
                save_images(img, os.path.join(test_args.image_path, f"bad/{counter2}_image.png"))
                counter2 += 1
            # avg_score += score
        # print(avg_score / n)
        return counter1, counter2


def test(args):
    global counter1, counter2
    device = args.device

    disc = Discriminator(in_channels=1).to(device)
    load_checkpoint(os.path.join(args.checkpoints, "disc6.pth.tar"), disc, None, None)

    ddpm = DDPM(img_channels=1, time_dim=args.emb_dim).to(device)
    diffusion = Diffusion(noise_steps=800, img_size=args.image_size, device=device)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, "ddpmbest64.pth.tar"), ddpm, None, None,
        )

    for i in range(test_args.num_iters):
        counter1, counter2 = diffusion.generate(ddpm, disc, args.batch_size, counter1, counter2)


def train(args):
    global counter
    device = args.device
    data_paths = glob("datasets/eyeq/images/*.png")
    label_paths = glob("datasets/eyeq/labels/*.png")
    # dataset = DDPMDataset(data_paths=data_paths, label_paths=label_paths, img_size=args.image_size)
    dataset = MaskDataset(data_paths=data_paths, label_paths=label_paths, img_size=args.image_size)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ddpm = DDPM(img_channels=3, time_dim=args.emb_dim).to(device)  # channels = 1
    optimizer = optim.AdamW(ddpm.parameters(), lr=args.lr, weight_decay=0.01)

    if args.load_model:
        load_checkpoint(
            os.path.join(args.checkpoints, "ddpm4.pth.tar"), ddpm, optimizer, args.lr,
        )

    disc = Discriminator(in_channels=1).to(device)
    load_checkpoint(os.path.join(args.checkpoints, "disc6.pth.tar"), disc, None, None)

    mse = nn.MSELoss()
    l1 = nn.L1Loss()
    bce = nn.BCELoss()

    m = nn.Sigmoid()
    min_avg_loss = float("inf")

    diffusion = Diffusion(noise_steps=800, img_size=args.image_size, device=device)
    score = torch.tensor([0]).item()
    for epoch in range(1, args.epochs):
        pbar = tqdm(dataloader)
        avg_loss = 0
        count1 = 0
        count2 = 0
        for i, (_, labels) in enumerate(pbar):
            # images = images.to(device)
            labels = labels.to(device)
            # labels[labels > 0] = images[labels > 0]

            t = diffusion.sample_timesteps(labels.shape[0]).to(device)
            x_t, noise = diffusion.noise_images(labels, t)

            for rep in range(5):
                if rep == 1:
                    count1 += 1
                predicted_noise = ddpm(x_t, t)
                loss = mse(noise, predicted_noise) + l1(noise, predicted_noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                if loss < min_avg_loss:
                    if rep == 4:
                        count2 += 1
                    break

            avg_loss += loss.item()
            # kl_div_loss = kl_div.item()
            pbar.set_postfix(epoch=epoch, AVG_LOSS=avg_loss / (i + 1), count1=count1,
                             count2=count2, MIN_LOSS=min_avg_loss, classifier_score=score)

            if i % ((len(dataloader)-1)//2) == 0 and i != 0:
                diffusion.sample(ddpm, disc, epoch, n=8)  # score =
                # score = torch.mean(score).item()
                counter += 1

        if min_avg_loss > avg_loss / len(dataloader):
            min_avg_loss = avg_loss / len(dataloader)

        save_checkpoint(ddpm, optimizer, filename=os.path.join(args.checkpoints, f"ddpm{epoch}.pth.tar"))


if __name__ == '__main__':
    training = False
    counter = 1

    if training:
        parser = argparse.ArgumentParser()
        args = parser.parse_args()
        args.load_model = False
        args.epochs = 500
        args.batch_size = 8
        args.emb_dim = 256 * 4
        args.image_size = 64 * 2
        args.num_workers = 4
        args.checkpoints = "results/experiment 7 noise2seg/checkpoints"
        args.dataset_path = None
        args.generated_path = None
        args.device = "cuda"
        args.lr = 3e-4
        train(args)
    else:
        counter1 = 20943
        counter2 = 1
        test_parser = argparse.ArgumentParser()
        test_args = test_parser.parse_args()
        test_args.load_model = True
        test_args.num_iters = 1000
        test_args.batch_size = 64
        test_args.emb_dim = 256 * 1
        test_args.image_size = 64 * 1
        test_args.num_workers = 4
        test_args.checkpoints = "results/experiment 7 noise2seg/checkpoints"
        test_args.image_path = "results/experiment 7 noise2seg/images"
        test_args.device = "cuda"
        test(test_args)
