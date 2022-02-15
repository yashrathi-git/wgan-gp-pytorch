# WGAN-GP
Pytorch implementation of [Improved Training of Wasserstein GANs
](https://arxiv.org/abs/1704.00028)

## Examples
### MNIST (8 epochs)
* lr = 1e-4, betas = (0.0, 0.9), batch size = 32
* Resized to 32 x 32

![Generator Training](https://user-images.githubusercontent.com/57002207/153932231-378f4a67-8091-4976-8254-110e0c315cad.gif)

## Setup
```shell
git clone https://github.com/yashrathi-git/wgan-gp-pytorch
```
and then copy the model files to current directory, or change current working directory:
```shell
cp wgan-gp-pytorch/wgan-gp/* .
```

## Usage
This is generalized to work on any image size, but you have to resize it to a multiple of 16, 
and the image should have equal height and width.
```python
from models import Generator, Critic


device = "cuda"

gen = Generator(z_dim=128, img_size=32, n_channels=1, hidden_dim=32).to(device)
critic = Critic(img_size=32, n_channels=1, hidden_dim=32).to(device)

from torch import optim
opt_gen = optim.Adam(gen.parameters(), lr=1e-4, betas=(0.0, 0.9))
opt_critic = optim.Adam(critic.parameters(), lr=1e-4, betas=(0.0, 0.9))

from training import Trainer
trainer = Trainer(generator=gen, 
                  critic=critic, 
                  optim_gen=opt_gen, 
                  optim_critic=opt_critic, 
                  use_cuda=True
                  )

trainer.train(dataloader, epochs=8, log_step_interval=100, 
              img_log_path="./img_logs", crit_rep=5,
              gp_lambda=10)

# Make gif of training
trainer.export_model_gif(images_dir="./img_logs")
```

**Visualise output**

```py
trainer.visualise_gen_images(num_images = 25)
```

**Save state and load state for training later**

Save state: `trainer.save_state(save_path)`

Load state: `trainer.load_state(save_path)`

## Inspiration
https://github.com/EmilienDupont/wgan-gp
