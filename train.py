import torch
from model import Generator,Discriminator
import config
from helper import train_fn
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_loader import dataloader

params = {k:v  for k,v in config.__dict__.items() if "__" not in k}
print(f"Params: {params}")

device = params['DEVICE']
adversarial_loss = torch.nn.BCELoss()
generator = Generator(params)
discriminator = Discriminator(params)
generator.to(device)
discriminator.to(device)
adversarial_loss.to(device)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=params['LEARNING_RATE'], betas=(params['B1'], params["B2"]))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=params['LEARNING_RATE'], betas=(params['B1'], params["B2"]))
losses_g = []  # List to store generator losses
losses_d = [] 

def main(params,adversarial_loss):
    for epoch in tqdm(range(params["NUM_EPOCHS"])):
        (losses_G,losses_D) = train_fn(epoch,params,generator,discriminator,adversarial_loss,dataloader,optimizer_G,optimizer_D)
        losses_g.extend(losses_G)
        losses_d.extend(losses_D)


main(params,adversarial_loss)
# Plot and save losses
plt.figure(figsize=(12, 5))
plt.plot(losses_g, label="Generator loss")
plt.plot(losses_d, label="Discriminator loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.legend()
plt.savefig("loss_plot.png")
plt.show()