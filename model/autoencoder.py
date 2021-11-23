import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(in_features=32768, out_features=2048)
        )
        self.linear_decoding = nn.Linear(in_features=2048, out_features=32768)
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.linear_decoding(x)
        x = torch.reshape(x, (x.shape[0], 32, 32, 32))
        out = self.decoder(x)
        out[out < 0.5] = 0.0
        out[out >= 0.5 ] = 1.0
        # for i in range(out.shape[0]):
        #     top5k_values, top_5k_indices = torch.topk(-out[i].flatten(), 330)
        #     #out = nn.LeakyReLU()(out)
        #     batch_indices = []
        #     c_indices = []
        #     row_indices = []
        #     col_indices = []

        #     for j in list(top_5k_indices):
        #         batch_indices.append(i)
        #         c_indices.append(0)
        #         row_indices.append(j % 256)
        #         col_indices.append(j // 256)
        #     out[batch_indices, c_indices, row_indices, col_indices] = 0.0
        #     out[i, :, :, :][out[i, :, :, :] != 0.0] = 1.0
        return out