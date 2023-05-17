import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import os


class Reconstructor(nn.Module):
    def __init__(self, pos_encoder='sin_cos', *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        input_dim = 40 if pos_encoder == 'sin_cos' else 2
        self.process = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

    def forward(self, x):
        x = self.process(x)
        return x


class PositionalEncoder(Dataset):
    def __init__(self, image, pos_encoder='sin_cos') -> None:
        self.image = image
        self.pos_encoder = pos_encoder

        H, W, C = image.shape

        L = 10

        # data
        self.input_for_model = []
        self.original_index = []
        self.gt = []

        for y_i in range(H):
            for x_i in range(W):
                tmp = []  # it will contains 40 values for 'sin_cos'

                xdash = (x_i / W) * 2 - 1
                ydash = (y_i / H) * 2 - 1

                # for the gt
                r, g, b = image[y_i, x_i]
                r = r*2 - 1
                g = g*2 - 1
                b = b*2 - 1
                if not self.pos_encoder == 'raw':
                    for l in range(L):
                        value = 2 ** l
                        sinx = np.sin(value * np.pi * xdash)
                        cosx = np.cos(value * np.pi * xdash)

                        siny = np.sin(value * np.pi * ydash)
                        cosy = np.cos(value * np.pi * ydash)
                        tmp.extend([sinx, cosx, siny, cosy])
                else:
                    tmp.extend([xdash, ydash])

                self.input_for_model.append(tmp)
                self.gt.append([r, g, b])
                self.original_index.append([x_i, y_i])

    def __len__(self):
        return len(self.input_for_model)

    def __getitem__(self, idx):
        return torch.tensor(self.input_for_model[idx], dtype=torch.float32), torch.tensor(self.gt[idx], dtype=torch.float32), torch.tensor(self.original_index[idx])


if __name__ == '__main__':
    im = Image.open('input.jpeg')
    im = np.array(im)
    im = im / 255.0

    POS_ENCODER = 'sin_cos'  # or 'raw'
    pos_encoder = PositionalEncoder(im, pos_encoder=POS_ENCODER)
    pos_encoder_valid = PositionalEncoder(
        im, pos_encoder=POS_ENCODER)  # test the model

    EPOCHS = 200
    DEVICE = 'cuda:0'
    BATCH_SIZE = 256

    model = Reconstructor(pos_encoder=POS_ENCODER)
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    dl = torch.utils.data.DataLoader(
        pos_encoder, batch_size=BATCH_SIZE, shuffle=True)
    dl_valid = torch.utils.data.DataLoader(
        pos_encoder_valid, batch_size=BATCH_SIZE, shuffle=True)

    # create folder for visualization
    if os.path.exists(f'outputs/{POS_ENCODER}'):
        os.system(f'rm -rf outputs/{POS_ENCODER}')
    os.mkdir(f'outputs/{POS_ENCODER}')

    # training loop
    for epochs in tqdm(range(EPOCHS), desc='Epochs', leave=True):
        for batch, (x, y, _) in enumerate((pbar := tqdm(dl, desc='Batches', leave=False))):
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad()

            preds = model(x)
            loss = criterion(preds, y)

            loss.backward()
            pbar.set_description(f"Loss {loss.item():.4f}")
            optimizer.step()

        if epochs % 2 == 0:  # batch > 0 and batch % 10000 == 0:
            # reconstruct the image

            out_image = np.zeros_like(im)
            model.eval()

            for x, _, index in dl_valid:
                index = index.detach().cpu().numpy()
                x = x.to(DEVICE)
                preds = model(x)
                preds = preds.detach().cpu().numpy()
                colors = np.clip((preds + 1)/2.0, 0, 1)
                out_image[index[:, 1], index[:, 0]] = colors

            out_image = (out_image * 255).astype('uint8')
            pil_img = Image.fromarray(out_image)

            pil_img.save(
                f'outputs/{POS_ENCODER}/{str(epochs).zfill(4)}_output_image__batch-{batch}.png')
