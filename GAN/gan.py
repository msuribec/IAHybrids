import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.datasets as datasets
import matplotlib
from torchvision.utils import make_grid, save_image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import imageio
import numpy
import os
matplotlib.style.use('ggplot')


class Generator(nn.Module):
    """Class to define the generator network
    Parameters:
        nz: int
            size of the input noise vector
    """
    def __init__(self, nz):
        super(Generator, self).__init__()
        self.nz = nz
        self.main = nn.Sequential(
            nn.Linear(self.nz, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )
    def forward(self, x):
        return self.main(x).view(-1, 1, 28, 28)
    
class Discriminator(nn.Module):
    """Class to define the discriminator network
    """
    def __init__(self):
        super(Discriminator, self).__init__()
        self.n_input = 784
        self.main = nn.Sequential(
            nn.Linear(self.n_input, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x = x.view(-1, 784)
        return self.main(x)


class GAN:
    """Class to define the GAN
    Parameters:
        batch_size: int
            batch size
        max_epochs: int
            maximum number of epochs
        sample_size: int
            number of sample images to be generated
        nz: int
        length of the noise vector
        k: int
            number of steps to apply to the discriminator
        eta: float
            learning rate
    """

    
    def __init__(self, batch_size, max_epochs, sample_size, nz, k, eta):

        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.sample_size = sample_size
        self.nz = nz
        self.k = k
        self.eta = eta

        self.input_path = 'Data'
        self.results_path = 'Results'
        self.imgs_path = f'{self.results_path}/Images'
        
        self.create_folders_if_not_exist([self.input_path,self.results_path, self.imgs_path])


        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,),(0.5,)),])
        self.to_pil_image = transforms.ToPILImage()

        self.train_data = datasets.MNIST(
            root=self.input_path,
            train=True,
            download=True,
            transform=transform
        )
        self.train_loader = DataLoader(self.train_data, batch_size= self.batch_size, shuffle=True)

        self.TOTAL_STEPS = int(len(self.train_data)/self.train_loader.batch_size)

        self.generator = Generator(self.nz).to(self.device)
        self.discriminator = Discriminator().to(self.device)

        # optimizers
        self.optim_g = optim.Adam(self.generator.parameters(), lr=self.eta)
        self.optim_d = optim.Adam(self.discriminator.parameters(), lr=self.eta)

        # loss function
        self.criterion = nn.BCELoss()

        self.generator_loss = [] # to store generator loss after each epoch
        self.discriminator_loss = [] # to store discriminator loss after each epoch
        self.images = [] # to store images generatd by the generator

        self.noise = self.create_noise(self.sample_size)

        self.generator.train()
        self.discriminator.train()


    def create_folders_if_not_exist(self,folder_paths):
        """Function to create folders if they don't exist
        Parameters:
            folder_paths: list
                list of paths of the folders to be created
        """
        for folder_path in folder_paths:
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

    def create_noise(self,sample_size):
        """Function to create the noise vector
        Parameters:
            sample_size: int
                size of the noise vector
        """
        return torch.randn(sample_size, self.nz).to(self.device)


    def train_discriminator(self, b_size, data_real):
        """Function to train the discriminator network
        Parameters:
            b_size: int
                batch size
            data_real: tensor
                real data
        Returns:
            loss: float
                loss of the discriminator network on the real and fake data
        """ 

        real_label = torch.ones(b_size, 1).to(self.device)
        fake_label = torch.zeros(b_size, 1).to(self.device)

        latent_v = self.create_noise(b_size)
        fake_image = self.generator(latent_v).detach()
        
        self.optim_d.zero_grad()

        y_real = self.discriminator(data_real)
        y_fake = self.discriminator(fake_image)

        loss_real = self.criterion(y_real, real_label)
        loss_fake = self.criterion(y_fake, fake_label)

        loss_real.backward()
        loss_fake.backward()

        self.optim_d.step()
        return loss_real + loss_fake


    def train_generator(self, b_size):
        """Function to train the generator network
        Parameters:
            b_size: int
                batch size
        Returns:
            loss: float
                loss of the generator network
        """

        real_label = torch.ones(b_size, 1).to(self.device)
        latent_v = self.create_noise(b_size)
        fake_image = self.generator(latent_v)
        
        self.optim_g.zero_grad()
        y = self.discriminator(fake_image)
        loss = self.criterion(y, real_label)
        
        loss.backward()
        
        self.optim_g.step()
        
        return loss


    def train(self):
        """Function to train the GAN and save results
        """

        for epoch in range(self.max_epochs):
            gen_loss, disc_loss = 0.0, 0.0
            for size_batch, data in tqdm(enumerate(self.train_loader), total=self.TOTAL_STEPS):

                real_image = (data[0]).to(self.device)
                b_size = len(real_image)

                for s in range(self.k):
                    disc_loss += self.train_discriminator(b_size, real_image)

                gen_loss += self.train_generator(b_size)

            generated_img = make_grid(self.generator(self.noise).cpu().detach())

            save_image(generated_img, f"{self.imgs_path}/{epoch}.png")

            self.images.append(generated_img)

            epoch_loss_g = gen_loss / size_batch
            epoch_loss_d = disc_loss / size_batch

            self.generator_loss.append(epoch_loss_g)
            self.discriminator_loss.append(epoch_loss_d)
    
            print(f"epoch: {epoch+1}/{self.max_epochs}  |  loss(gen): {epoch_loss_g:.8f}  |  loss(disc): {epoch_loss_d:.8f}")

    def get_numpy_array(self, ar):
        """Function to get numpy array from the tensors
        Parameters:
            ar: list
                list of tensors
        Returns:
            list
                list of numpy arrays
        """
        #we move tensors from gpu to cpu and detach them from any gradient calculations
        return [ar[i].cpu().detach().numpy() for i in range (len(ar))] 

    def save_results(self):
        """Function to save the results
        """
        plt.figure(figsize=(15,7))
        gen_loss = self.get_numpy_array(self.generator_loss)
        disc_loss = self.get_numpy_array(self.discriminator_loss)
        
        plt.plot(gen_loss, label='Generator')
        plt.plot(disc_loss, label='Discriminator')
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.legend()
        plt.savefig(f'{self.results_path}/loss.png')

if __name__ == "__main__":
    g1 = GAN(512,3,64,128,1, 0.0002)
    g1.train()
    g1.save_results()
