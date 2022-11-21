import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from model import Discriminator, Generator, init_weights
import time
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 2e-4
BATCH_SIZE = 128
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 3
FEATURES_DISC = 64
FEATURES_GEN = 64

transforms = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5 for _ in (range(CHANNELS_IMG))], [0.5 for _ in (range(CHANNELS_IMG))])
    ]
)

dataset = datasets.MNIST(root="datasets/",train=True,transform=transforms, download=True)
loader = DataLoader(dataset,BATCH_SIZE,shuffle=True)
gen = Generator(Z_DIM, CHANNELS_IMG, FEATURES_GEN).to(DEVICE)
disc = Discriminator(CHANNELS_IMG,FEATURES_DISC).to(DEVICE)
init_weights(gen)
init_weights(disc)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))
opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5,0.999))
criterion = nn.BCELoss()

fixed_noise = torch.randn(32,Z_DIM,1,1).to(DEVICE)
#WRITER = SUMMARY WRITER TENSORBOARD
#step = 0
real_batch = next(iter(loader))
print(real_batch[0].shape)
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0][:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

print(gen)
print(disc)

gen.train()
disc.train()
img_list = []
for epoch in range(NUM_EPOCHS):
    start = time.time()
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(DEVICE)
        noise = torch.randn((BATCH_SIZE,Z_DIM,1,1)).to(DEVICE)
        fake = gen(noise)

        ### Train Disc, max log(D(X)) + log(1-D(G(x)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real= criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake).reshape(-1)
        loss_disc_fake=criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake)/2
        disc.zero_grad()
        loss_disc.backward(retain_graph=True)
        opt_disc.step()

        ### Train Generator, min log(1 - D(F(z))) <--> max log(D(G(z)))
        output = disc(fake).reshape(-1)
        loss_gen = criterion(output, torch.ones_like(output))
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 50 == 0:
            print(                                                          #TAB IS HERE
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \t\
                Loss D: {loss_disc:.4f}, Loss G: {loss_gen:.4f} Time: {time.time()-start}s:"
            )
            start = time.time()

    with torch.no_grad():
        fake = gen(fixed_noise)
        #img_grid_real = vutils.make_grid(real[:32].cpu(), normalize=True)
        #img_grid_real = np.transpose(vutils.make_grid(fake.cpu(), padding=2, normalize=True).cpu(),(1,2,0))
        img_list.append(vutils.make_grid(fake.cpu(), padding=2, normalize=True))
        #img_grid_fake = vutils.make_grid(fake[:32].cpu(), normalize=True)

            #step += 1
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(img_list[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
#plt.imshow(img_grid_fake)
#plt.show()