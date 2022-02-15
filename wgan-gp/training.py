import os

import torch
from tqdm import tqdm

from utils import tensor_image_grid


class Trainer:
    def __init__(
        self,
        generator,
        critic,
        optim_gen,
        optim_critic,
        use_cuda=False,
        wandb_logger=None,
    ):
        super(Trainer, self).__init__()
        self.generator = generator
        self.critic = critic
        self.optim_gen = optim_gen
        self.optim_critic = optim_critic
        self.device = torch.device("cuda" if use_cuda else "cpu")
        self.wandb_logger = wandb_logger
        self.gen_losses = []
        self.crit_losses = []
        self.step = 0

    def _calc_grad_penalty(self, real, fake, epsilon):
        mixed_image = real * epsilon + fake * (1 - epsilon)
        mixed_scores = self.critic(mixed_image)
        gradient = torch.autograd.grad(
            outputs=mixed_scores,
            inputs=mixed_image,
            create_graph=True,
            retain_graph=True,
            grad_outputs=torch.ones_like(mixed_scores),
        )[0]
        gradient = gradient.view(gradient.size(0), -1)
        gradient_penalty = torch.mean((gradient.norm(2, dim=1) - 1) ** 2)
        return gradient_penalty

    @staticmethod
    def _get_gen_loss(crit_fake_scores):
        loss = -torch.mean(crit_fake_scores)
        return loss

    @staticmethod
    def _get_crit_loss(crit_real_scores, crit_fake_scores, gp, gp_lambda):
        loss = (
            -(torch.mean(crit_real_scores) - torch.mean(crit_fake_scores))
            + gp * gp_lambda
        )
        return loss

    def train_step(self, real_img, crit_rep=5, gp_lambda=10):
        self.generator.train()
        self.critic.train()

        real_img = real_img.to(self.device)
        bs = real_img.size(0)

        # Train Critic
        running_iter_critic_loss = 0
        for _ in range(crit_rep):
            z = self.generator.sample_noise_vector(bs).to(self.device)
            fake = self.generator(z)
            crit_real_scores = self.critic(real_img)
            crit_fake_scores = self.critic(fake.detach())
            epsilon = torch.rand(bs, 1, 1, 1, requires_grad=True).to(self.device)
            gp = self._calc_grad_penalty(real_img, fake.detach(), epsilon)
            crit_loss = self._get_crit_loss(
                crit_real_scores, crit_fake_scores, gp, gp_lambda
            )
            self.optim_critic.zero_grad()
            crit_loss.backward(retain_graph=True)
            self.optim_critic.step()
            running_iter_critic_loss += crit_loss.item()

        crit_loss_scalar = running_iter_critic_loss / crit_rep
        self.crit_losses.append(crit_loss_scalar)

        # Train Generator
        z = self.generator.sample_noise_vector(bs).to(self.device)
        fake = self.generator(z)
        crit_fake_scores = self.critic(fake)
        gen_loss = self._get_gen_loss(crit_fake_scores)
        self.optim_gen.zero_grad()
        gen_loss.backward()
        self.optim_gen.step()

        gen_loss_scalar = gen_loss.item()
        self.gen_losses.append(gen_loss_scalar)

        self.step += 1
        if self.wandb_logger is not None:
            self.wandb_logger.log(
                {
                    "Generator Loss": gen_loss_scalar,
                    "Critic Loss": crit_loss_scalar,
                },
                step=self.step,
            )

    def train_epoch(
        self,
        dataloader,
        log_step_interval=100,
        img_log_path="./img_log",  # Doesn't logs if it's set to None
        crit_rep=5,
        gp_lambda=10,
    ):
        if img_log_path is not None:
            if not os.path.exists(img_log_path):
                os.mkdir(img_log_path)

        fixed_noise = self.generator.sample_noise_vector(25).to(self.device)

        for real, _ in tqdm(dataloader):
            self.train_step(real, crit_rep, gp_lambda)

            if self.step % log_step_interval == 0:
                if img_log_path is not None:
                    img_save_path = os.path.join(img_log_path, f"{self.step}.png")
                else:
                    img_save_path = None

                fake = self.visualise_gen_images(fixed_noise, save_path=img_save_path)
                if self.wandb_logger is not None:
                    self.wandb_logger.log(
                        {
                            "Generated Images": [
                                self.wandb_logger.Image(image) for image in fake
                            ]
                        },
                        step=self.step,
                    )

                print(
                    f"Step {self.step}, "
                    f"Generator Loss: {self.gen_losses[-1]:.4f}, "
                    f"Critic Loss: {self.crit_losses[-1]:.4f}"
                )

    def train(
        self,
        dataloader,
        epochs=10,
        log_step_interval=100,
        img_log_path="./img_log",  # Doesn't logs if it's set to None
        crit_rep=5,
        gp_lambda=10,
    ):
        for epoch in range(epochs):
            self.train_epoch(
                dataloader,
                log_step_interval,
                img_log_path,
                crit_rep,
                gp_lambda,
            )

    @staticmethod
    def export_model_gif(images_dir, output_path="gen_vis.gif"):
        import imageio

        images = []
        for filename in sorted(
            os.listdir(images_dir), key=lambda x: int(x.split(".")[0])
        ):
            images.append(imageio.imread(os.path.join(images_dir, filename)))

        imageio.mimsave(output_path, images)

    @torch.no_grad()
    def visualise_gen_images(self, noise=None, num_images=25, nrow=5, save_path=None):
        if noise is None:
            noise = self.generator.sample_noise_vector(num_images).to(self.device)
        fake = self.generator(noise)
        tensor_image_grid(fake, num_images, nrow, save_path)
        return fake

    def save_state(self, save_path):
        torch.save(
            {
                "step": self.step,
                "gen_state_dict": self.generator.state_dict(),
                "crit_state_dict": self.critic.state_dict(),
                "optim_gen_state_dict": self.optim_gen.state_dict(),
                "optim_crit_state_dict": self.optim_critic.state_dict(),
                "gen_losses": self.gen_losses,
                "crit_losses": self.crit_losses,
            },
            save_path,
        )

    def load_state(self, save_path):
        state = torch.load(save_path)
        self.step = state["step"]
        self.generator.load_state_dict(state["gen_state_dict"])
        self.critic.load_state_dict(state["crit_state_dict"])
        self.optim_gen.load_state_dict(state["optim_gen_state_dict"])
        self.optim_critic.load_state_dict(state["optim_critic_state_dict"])
        self.gen_losses = state["gen_losses"]
        self.crit_losses = state["crit_losses"]
