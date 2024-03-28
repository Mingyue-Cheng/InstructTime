import torch.nn as nn

class MSE:
    def __init__(self, model, latent_loss_weight=0.25):
        self.model = model
        self.latent_loss_weight = latent_loss_weight
        self.mse = nn.MSELoss()

    def compute(self, batch):
        seqs = batch
        out, latent_loss, _ = self.model(seqs)
        recon_loss = self.mse(out, seqs)
        latent_loss = latent_loss.mean()
        loss = recon_loss + self.latent_loss_weight * latent_loss
        return loss