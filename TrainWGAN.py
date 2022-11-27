import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from modelWGAN import Critic, Generator, init_weights
import time
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from utils import gradient_penalty

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
IMAGE_SIZE = 64
CHANNELS_IMG = 1
Z_DIM = 100
NUM_EPOCHS = 4
FEATURES_DISC = 64
FEATURES_GEN = 64
CRITIC_ITERATIONS = 5
LAMBDA_GP = 10

LOAD_PARAMS = False
SAVE_PARAMS = True

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
critic = Critic(CHANNELS_IMG, FEATURES_DISC).to(DEVICE)
init_weights(gen)
init_weights(critic)

opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.0,0.9))
opt_critic = optim.Adam(critic.parameters(), lr=LEARNING_RATE, betas=(0.0,0.9))

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
print(critic)

print(f"LOAD_PARAMS is set to: {LOAD_PARAMS} \nSAVE_PARAMS is set to: {SAVE_PARAMS}")
if LOAD_PARAMS:
    try:
        gen.load_state_dict(torch.load("wgangp_genP.pt"))
        critic.load_state_dict(torch.load("wgangp_criticP.pt"))
        print("Params loaded")
    except FileNotFoundError:
        print("No loadable parameters found, Initializing new params instead")



gen.train()
critic.train()
img_list = []
for epoch in range(NUM_EPOCHS):
    start = time.time()
    for batch_idx, (real, _) in enumerate(loader):
        real = real.to(DEVICE)
        curr_batch_size = real.shape[0]
        loss_critic = 0

        for _ in range(CRITIC_ITERATIONS):
            noise = torch.randn((curr_batch_size, Z_DIM, 1, 1)).to(DEVICE)
            fake = gen(noise)
            critic_real = critic(real).reshape(-1)
            critic_fake = critic(fake).reshape(-1)
            gp = gradient_penalty(critic,real, fake, device=DEVICE)
            loss_critic = -(torch.mean(critic_real) - torch.mean(critic_fake)) + LAMBDA_GP * gp
            critic.zero_grad()
            loss_critic.backward(retain_graph=True)
            opt_critic.step()


        output = critic(fake).reshape(-1)
        loss_gen = -torch.mean(output)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

        if batch_idx % 50 == 0:
            print(                                                          #TAB IS HERE
                f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {batch_idx}/{len(loader)} \t\
                Loss D: {loss_critic:.4f}, Loss G: {loss_gen:.4f} Time: {time.time()-start}s:"
            )
            start = time.time()

    with torch.no_grad():
        fake = gen(fixed_noise)
        #img_grid_real = vutils.make_grid(real[:32].cpu(), normalize=True)
        #img_grid_real = np.transpose(vutils.make_grid(fake.cpu(), padding=2, normalize=True).cpu(),(1,2,0))
        img_list.append(vutils.make_grid(fake.cpu(), padding=2, normalize=True))
        if SAVE_PARAMS:
            torch.save(gen.state_dict(), "wgangp_genP.pt")
            torch.save(critic.state_dict(), "wgangp_criticP.pt")
            print("Current params saved")
        #img_grid_fake = vutils.make_grid(fake[:32].cpu(), normalize=True)

            #step += 1
    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(vutils.make_grid(img_list[:64], padding=2, normalize=True).cpu(),(1,2,0)))
    plt.show()
#plt.imshow(img_grid_fake)
#plt.show()