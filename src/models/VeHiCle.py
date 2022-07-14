import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl


import yaml

################################################ VEHICLE LOSS FUNCTION #####################################################################

class VAE_Model(pl.LightningModule):

    def __init__(self,
            batch_size=-9,
            kld_weight=0.0,
            kld_weight_inc=0.000,
            lr=0.0001,
            gamma=0.99,
            latent_dim=100,
            pre_latent=4608,
            condensed_latent=3,
            ):
        super(VAE_Model, self).__init__()
        torch.manual_seed(0)
        self.batch_size       = batch_size
        self.epoch_num        = 0
        self.kld_weight_inc   = kld_weight_inc
        self.kld_weight       = kld_weight
        self.lr               = lr
        self.gamma            = gamma
        self.latent_dim       = latent_dim
        self.PRE_LATENT       = pre_latent
        self.CONDENSED_LATENT = condensed_latent
        hidden_dims      = [32, 64, 128, 256, 256, 512, 512]
        modules               = []

        self.save_hyperparameters()

        in_channels = 1 
        #Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                                kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU())
            )
            in_channels = h_dim
        self.encoder = nn.Sequential(*modules)

        self.fc_mu    = nn.Linear(self.PRE_LATENT, self.latent_dim)
        self.fc_var   = nn.Linear(self.PRE_LATENT, self.latent_dim)

        #Build Decoder
        modules = []
        self.decoder_input = nn.Linear(self.latent_dim, self.PRE_LATENT)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) -1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i +1],
                                       kernel_size=3,
                                       stride =2,
                                       padding=1),
                                       #output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i+1]),
                    nn.LeakyReLU())
                )
        
        self.decoder = nn.Sequential(*modules)
        self.final_layer = nn.Sequential(
                        nn.ConvTranspose2d(hidden_dims[-1],
                                        hidden_dims[-1],
                                        kernel_size=3,
                                        stride=2,
                                        padding=1),
                                        #output_padding=1),
                        nn.BatchNorm2d(hidden_dims[-1]),
                        nn.LeakyReLU(),
                        nn.Conv2d(hidden_dims[-1], out_channels=1,
                                kernel_size=3, padding=1),
                        #nn.Tanh())
                        nn.Sigmoid())

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z):
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [B x D]
        :return: (Tensor) [B x C x H x W]
        """
        rez  = self.decoder_input(z)
        rez  = rez.view(-1,
                512,
                self.CONDENSED_LATENT,
                self.CONDENSED_LATENT)
        result  = list(self.decoder.children())[0](rez)
        result  = list(self.decoder.children())[1](result)
        result  = list(self.decoder.children())[2](result)
        result  = list(self.decoder.children())[3](result)
        result  = list(self.decoder.children())[4](result)
        result  = list(self.decoder.children())[5](result)
        #result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def get_z(self, x):
        mu, log_var = self.encode(x)
        z           = self.reparameterize(mu, log_var)
        return z, mu, log_var

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        recon = self.decode(z)
        return  [recon, x, mu, log_var]

    def loss_function(self, *args):
        recons  = args[0]
        x       = args[1]
        mu      = args[2]
        log_var = args[3]
        
        kld_weight = self.kld_weight 

        recon_loss = F.mse_loss(recons, x)
        kld_loss   = torch.mean(-0.5 * torch.sum(1 + log_var - mu **2 - log_var.exp(), dim = 1), dim = 0)
        loss = recon_loss + (kld_weight*kld_loss)
        self.log('kld_weight', kld_weight)
        self.log('train_loss', loss)
        self.log('recon_loss', recon_loss)
        self.log('kld_loss', kld_loss)
        return loss, recon_loss, kld_loss

    def training_step(self, batch, batch_idx):
        data, target, _               = batch
        results                    = self.forward(target)
        loss, recon_loss, kld_loss = self.loss_function(*results) 
        return loss

    def training_epoch_end(self, training_step_outputs):
        print(self.epoch_num)
        self.epoch_num = self.epoch_num+1
        if self.epoch_num > 0:
           self.kld_weight = self.kld_weight+self.kld_weight_inc

    def validation_step(self, batch, batch_idx):
        data, target, _            = batch
        results                    = self.forward(target)
        loss, recon_loss, kld_loss = self.loss_function(*results) 
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                    lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                gamma=self.gamma)
        return [optimizer], [scheduler]


class VaeLoss(torch.nn.Module):
    def __init__(self):
        super(VaeLoss, self).__init__()
        op      = open('/home/murtaza/Documents/GrapHiC/weights/vehicle/vehicle_vae_hparams.yaml')
        hparams = yaml.load(op)
        model   = VAE_Model(
                condensed_latent=hparams['condensed_latent'],
                gamma=hparams['gamma'],
                kld_weight=hparams['kld_weight'],
                latent_dim=hparams['latent_dim'],
                lr=hparams['lr'],
                pre_latent=hparams['pre_latent'])
        self.pretrained_model = model.load_from_checkpoint('/home/murtaza/Documents/GrapHiC/weights/vehicle/vehicle_vae.ckpt')
        self.hparams = hparams

    def forward(self, output, target):
        latent_output, mu_out, var_out       = self.pretrained_model.get_z(output)
        latent_target, mu_target, var_target = self.pretrained_model.get_z(target)
        loss          = F.mse_loss(mu_target, mu_out)
        return loss







class computeInsulation(torch.nn.Module):
    def __init__(self, window_radius=10, deriv_size=10):
        super(computeInsulation, self).__init__()
        self.window_radius = window_radius
        self.deriv_size  = deriv_size
        self.di_pool     = torch.nn.AvgPool2d(kernel_size=(2*window_radius+1), stride=1) #51
        self.top_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
    
    def forward(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)       
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,self.deriv_size:])
        bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
        dv     = (top-bottom)
        left   = torch.cat([torch.zeros(dv.shape[0], dv.shape[1],2), dv], dim=2)
        right  = torch.cat([dv, torch.zeros(dv.shape[0], dv.shape[1],2)], dim=2)
        band   = ((left<0) == torch.ones_like(left)) * ((right>0) == torch.ones_like(right))
        band   = band[:,:,2:-2]
        boundaries = []
        for i in range(0, band.shape[0]):
            cur_bound = torch.where(band[i,0])[0]+self.window_radius+self.deriv_size
            boundaries.append(cur_bound)
        return iv, dv, boundaries

class InsulationLoss(torch.nn.Module):
    def __init__(self, window_radius=10, deriv_size=10):
        super(InsulationLoss, self).__init__()
        self.deriv_size     = deriv_size
        self.window_radius  = window_radius=10
        self.di_pool        = torch.nn.AvgPool2d(kernel_size=window_radius, stride=1)
        self.top_pool       = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)
        self.bottom_pool    = torch.nn.AvgPool1d(kernel_size=deriv_size, stride=1)

    def indivInsulation(self, x):
        iv     = self.di_pool(x)
        iv     = torch.diagonal(iv, dim1=2, dim2=3)       
        iv     = torch.log2(iv/torch.mean(iv))
        top    = self.top_pool(iv[:,:,self.deriv_size:])
        bottom = self.bottom_pool(iv[:,:,:-self.deriv_size])
        dv     = (top-bottom)
        return dv

    def forward(self, output, target):
        out_dv = self.indivInsulation(output)
        tar_dv = self.indivInsulation(target)
        loss   = F.mse_loss(tar_dv, out_dv)
        return loss







################################################ VEHICLE GAN CLASSES #####################################################################
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.relu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res

class Generator(nn.Module):
    def __init__(self, num_res_blocks=5):
        super(Generator, self).__init__()

        self.pre_res_block = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3),
                nn.ReLU(),
                )

        res_blocks = [ResidualBlock(64) for _ in range(num_res_blocks)]
        self.res_blocks = nn.Sequential(*res_blocks)

        self.post_res_block = nn.Sequential(
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64)
                )

        self.final_block = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3),
                nn.Conv2d(128, 128, kernel_size=3),
                nn.Conv2d(128, 256, kernel_size=3),
                nn.Conv2d(256, 256, kernel_size=3),
                nn.Conv2d(256, 1, kernel_size=3),
                )

    def forward(self, x):
        first_block = self.pre_res_block(x)
        res_blocks = self.res_blocks(first_block)
        post_res_block = self.post_res_block(res_blocks)
        final_block = self.final_block(first_block + post_res_block)
        #return torch.tanh(final_block)
        return torch.sigmoid(final_block)

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2, padding=1, bias=False),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(512, 1, kernel_size=1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.conv(x)
        return x

    def init_params(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight.data, 0.0, 0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight.data, 1.0, 0.02)
                nn.init.constant_(module.bias.data, 0)




class VeHiCle(pl.LightningModule):
    def __init__(self):
        super(VeHiCle, self).__init__()
        self.mse_lambda     = 1
        self.tad_lambda     = 1
        self.vae_lambda     = 1e-3 
        self.gan_lambda     = 2.5e-3
        self.G_lr           = 1e-5
        self.D_lr           = 1e-5
        self.beta_1         = 0.9  
        self.beta_2         = 0.99
        self.num_res_blocks = 15
        self.generator      = Generator(num_res_blocks=self.num_res_blocks)
        self.discriminator  = Discriminator() 
        self.generator.init_params()
        self.discriminator.init_params()
        self.bce             = nn.BCEWithLogitsLoss()
        self.mse             = nn.L1Loss()
        self.vae             = VaeLoss()
        self.tad             = InsulationLoss()
    
    def forward(self, x):
        fake = self.generator(x)
        return fake

    def tad_loss(self, target, output):
        return self.tad(target, output)

    def vae_loss(self, target, output):
        return self.vae(target, output)

    def adversarial_loss(self, target, output):
        return self.bce(target, output)

    def meanSquaredError_loss(self, target, output):
        return self.mse(target, output)

    def training_step(self, batch, batch_idx, optimizer_idx):
        data, full_target, info = batch
        target = full_target[:,:,6:-6,6:-6]

        #Generator
        if optimizer_idx == 0:
            self.generator.zero_grad()
            output      = self.generator(data)
            MSE_loss    = self.meanSquaredError_loss(output, target)
            VAE_loss    = self.vae_loss(output, target)
            TAD_loss    = self.tad_loss(output, target)
            pred_fake   = self.discriminator(output)
            labels_real = torch.ones_like(pred_fake, requires_grad=False)
            GAN_loss    = self.adversarial_loss(pred_fake, labels_real)
            
            total_loss_G = (self.mse_lambda*TAD_loss)+(self.vae_lambda*VAE_loss)+(self.mse_lambda * MSE_loss)+(self.gan_lambda *GAN_loss)
            self.log("total_loss_G", total_loss_G)
            return total_loss_G
        
        #Discriminator
        if optimizer_idx == 1:
            self.discriminator.zero_grad()
            #train on real data
            pred_real       = self.discriminator(target)
            labels_real     = torch.ones_like(pred_real, requires_grad=False)
            pred_labels_real = (pred_real>0.5).float().detach()
            acc_real        = (pred_labels_real == labels_real).float().sum()/labels_real.shape[0]
            loss_real       = self.adversarial_loss(pred_real, labels_real)
            
            #train on fake data
            output           = self.generator(data)
            pred_fake        = self.discriminator(output.detach())
            labels_fake      = torch.zeros_like(pred_fake, requires_grad=False)
            pred_labels_fake = (pred_fake > 0.5).float()
            acc_fake         = (pred_labels_fake == labels_fake).float().sum()/labels_fake.shape[0]
            loss_fake        = self.adversarial_loss(pred_fake, labels_fake)

            total_loss_D = loss_real + loss_fake
            self.log("total_loss_D",total_loss_D)
            return total_loss_D



    def validation_step(self, batch, batch_idx):
        data, full_target, info  = batch
        output       = self.generator(data)
        target       = full_target[:,:,6:-6,6:-6]
        MSE_loss     = self.meanSquaredError_loss(output, target)
        return MSE_loss

       
    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.G_lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.D_lr)
        return [opt_g, opt_d]













































