import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid


def tensor_image_grid(img_tensor, n_images=25, nrow=5, save_image_path=None):
    img_tensor = (img_tensor + 1) / 2  # Bring it between 0 and 1
    img_tensor = img_tensor.detach().cpu()
    img_grid = make_grid(img_tensor[:n_images], nrow=nrow)
    # Move n_C to the end
    img_grid = torch.moveaxis(img_grid, 0, -1)
    plt.imshow(torch.squeeze(img_grid))

    if save_image_path is not None:
        plt.savefig(save_image_path)
    plt.show()

    return img_grid
