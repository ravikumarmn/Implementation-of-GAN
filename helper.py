import torch
import os
from torchvision.utils import save_image


 # List to store discriminator losses

def train_fn(epoch,params,generator,discriminator,criterion_loss,dataloader,optimizer_G,optimizer_D):
    os.makedirs("images", exist_ok=True)
    losses_G = []  # List to store generator losses
    losses_D = [] 
    for i, (imgs, _) in enumerate(dataloader):
        imgs = imgs.to(params["DEVICE"])
        valid = torch.ones_like(torch.randn(imgs.size(0), 1), requires_grad=False).to(params["DEVICE"])
        fake = torch.zeros_like(torch.randn(imgs.size(0), 1), requires_grad=False).to(params["DEVICE"])
        # ----------------
        # Train Generator
        # ----------------
        z = torch.normal(0, 1, (imgs.shape[0], params["LATENT_DIM"])).to(params["DEVICE"])

        gen_imgs = generator(z)
        g_loss = criterion_loss(discriminator(gen_imgs), valid)
        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()
        # --------------------
        # Train Discriminator
        # --------------------
        optimizer_D.zero_grad()
        real_loss = criterion_loss(discriminator(imgs), valid)
        fake_loss = criterion_loss(discriminator(gen_imgs.detach()), fake)

        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        optimizer_D.step()
        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
            % (epoch + 1, params["NUM_EPOCHS"], i, len(dataloader), d_loss.item(), g_loss.item())
        )
        batches_done = epoch * len(dataloader) + i
        if batches_done % params["SAMPLE_INTERVAL"] == 0:
            save_image(gen_imgs.data[:25], "images/%d.png" % batches_done, nrow=5, normalize=True)

        # Save losses
        losses_G.append(g_loss.item())
        losses_D.append(d_loss.item())
    return (losses_G,losses_D)
