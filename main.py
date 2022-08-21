import os
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path
from datasets import load_dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.optim import Adam
from src.diffusion_from_scratch.diffusion import p_losses, sample, create_diffusion_params
from src.diffusion_from_scratch.unet import Unet
from src.diffusion_from_scratch.utils import img_to_tensor_pipeline, num_to_groups

# settings
image_size = 28
channels = 1
batch_size = 128
timesteps = 200
results_folder = Path("./results")
model_folder = Path("./models")

# Checks
results_folder.mkdir(exist_ok=True)
model_folder.mkdir(exist_ok=True)

# load dataset from the hub
dataset = load_dataset("fashion_mnist")

# transform the images to (pixel) tensors
transformed_dataset = dataset.with_transform(img_to_tensor_pipeline).remove_columns(["label"])

# create dataloader
dataloader = DataLoader(transformed_dataset["train"], batch_size=batch_size, shuffle=True)

# define the diffusion process parameters
diffusion_params = create_diffusion_params(timesteps=timesteps)

# Run on GPU, CPU or M1
if torch.cuda.is_available():
    device = 'cuda'
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = 'mps'
    # aten::native_group_norm_backward not implemented for MPS
    # https://github.com/pytorch/pytorch/issues/77764
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
else:
    device = 'cpu'

# Create model
model = Unet(dim=image_size, channels=channels, dim_mults=(1, 2, 4,))
model.to(device)

# Optimzer and training parameters
optimizer = Adam(model.parameters(), lr=1e-3)
epochs = 10
save_and_sample_every = 1000

# Train run
for epoch in range(epochs):
    for step, batch in enumerate(dataloader):

        optimizer.zero_grad()

        batch_size = batch["pixel_values"].shape[0]
        batch = batch["pixel_values"].to(device)

        # Algorithm 1 line 3: sample t uniformly for every example in the batch
        t = torch.randint(0, timesteps, (batch_size,), device=device).long()

        # Calculate the losses between predicted and actual noise
        loss = p_losses(model, batch, t, diff_dict=diffusion_params, loss_type="huber")

        if step % 100 == 0:
            print(f"Loss @Epoch {epoch} - Step {step}: {loss.item()}")

        loss.backward()
        optimizer.step()

        # save generated images
        if step != 0 and step % save_and_sample_every == 0:
            milestone = step // save_and_sample_every
            batches = num_to_groups(4, batch_size)
            all_images_list = list(map(lambda n: sample(model, batch_size=n, channels=channels), batches))
            all_images = torch.cat(all_images_list, dim=0)
            all_images = (all_images + 1) * 0.5
            save_image(all_images, str(results_folder / f'sample-{milestone}.png'), nrow=6)

# Save model
torch.save(model, str(model_folder / "diffusion_model.pt"))

# sample 64 images (reverse diffusion)
samples = sample(model, diff_dict=diffusion_params, image_size=image_size, batch_size=64, channels=channels)

# show a random one
random_index = 2
plt.imshow(samples[-1][random_index].reshape(image_size, image_size, channels), cmap="gray")

# Create a gif
random_index = 53

fig = plt.figure()
ims = []
for i in range(timesteps):
    im = plt.imshow(samples[i][random_index].reshape(image_size, image_size, channels), cmap="gray", animated=True)
    ims.append([im])

animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
animate.save('diffusion.gif')
plt.show()
