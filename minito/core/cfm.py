from tqdm import tqdm
import torch
#from pdb import set_trace as st
import lightning as pl
from minito.core.dataloaders import BaseDensity

# adapted from TITO repo

class CFM(pl.pytorch.LightningModule):
    def __init__(self, velocity, lr=1e-3):
        super().__init__()
        self.vf = velocity
        self.sigma = 0.01
        self.save_hyperparameters()
        self.learning_rate = lr
        self.tsampler = torch.distributions.Beta(2.0, 1.0) 
    

    def training_step(self, batch, batch_idx):
        #t = torch.rand((len(batch['cond']['x']),)).to(batch['cond']['x'].device)        
        t = self.tsampler.sample((len(batch['cond']['x']),)).to(batch['cond']['x'].device)
        loss = self.get_loss(t, batch)
        self.log("train/loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        #t = torch.rand((len(batch['cond']['x']),)).to(batch['cond']['x'].device)
        t = self.tsampler.sample((len(batch['cond']['x']),)).to(batch['cond']['x'].device)
        loss = self.get_loss(t, batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss            

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    def get_loss(self, t, batch):
        x0 = batch['target']['xbase']
        x1 = batch['target']['x']
        xt = self.sample_conditional_pt(t, x0, x1)
        ut = x1 - x0  # Simplified call: compute_conditional_vector_field
        vt = self.vf(t, batch, xt)

        return torch.nn.functional.mse_loss(vt, ut)

    def sample_conditional_pt(self, t, x0, x1):
        epsilon = torch.normal(0, 1, size=x0.shape, device=x0.device)
        mu_t = (t * x1.T + (1 - t) * x0.T).T
        return mu_t + self.sigma * epsilon

    def compute_conditional_vector_field(self, x0, x1):
        return x1 - x0

    def _forward(self, t, batch):
        xt = batch['corr']['x']
        # t comes in from SampleHandler as (B,)
        return self.vf(t, batch, xt)

    def sample(self, batch, ode_steps=50, nested_samples=1, base_distribution=None):
        self.eval()
        device = self.device
        sh = SampleHandler(self._forward)
        
        # Initialize x_current from noise (x0 at t=0)
        # Using 1.0 std if no distribution provided
        if 'corr' not in batch: batch['corr'] = {}
        
        x_current = torch.randn_like(batch['cond']['x']).to(device)
        if base_distribution is not None:
            x_current = base_distribution.sample_as(batch['cond']['x']).to(device)
            
        batch['corr']['x'] = x_current
        dt = 1.0 / ode_steps
        traj = [x_current.clone()]

        with torch.no_grad():
            for i_nested in range(nested_samples):
                for i_ode in range(ode_steps):
                    # Compute t for the current step
                    t_val = torch.tensor([i_ode * dt], device=device)
                    
                    # Euler step
                    v_t = sh(t_val, batch)
                    x_current = x_current + dt * v_t
                    
                    # Update state for next VF call
                    batch['corr']['x'] = x_current
                
                traj.append(x_current.clone())
                
                # For nested/autoregressive loops:
                if i_nested < nested_samples - 1:
                    batch["cond"]['x'] = x_current.clone()
                    x_current = torch.randn_like(x_current).to(device)
                    batch['corr']['x'] = x_current

            batch["traj_x"] = torch.stack(traj, dim=0)
            return batch
    
class SampleHandler:
    def __init__(self, sample_forward):
        self.sample_forward = sample_forward

    def __call__(self, t, batch):
        t = t.repeat((len((batch['cond']['x'])), ))
        return self.sample_forward(t, batch)