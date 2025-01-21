import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, num_classes):
        super().__init__()

        d = 256

        d -= 2
        embd = 64

        # encoder
        # encoder level 1
        self.e11 = nn.Sequential(
            # 3 channels for rgb
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.e12 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        # max pool 1
        self.mp1 = nn.MaxPool2d(kernel_size=2, stride=2)
        d //= 2
        d -= 2
        embd *= 2

        # encoder level 2
        self.e21 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d -= 2
        self.e22 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        # max pool 2
        self.mp2 = nn.MaxPool2d(kernel_size=2, stride=2)
        d //= 2
        d -= 2
        embd *= 2

        # encoder level 3
        self.e31 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.e32 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        # max pool 3
        self.mp3 = nn.MaxPool2d(kernel_size=2, stride=2)
        d //= 2
        d -= 2
        embd *= 2

        # encoder level 4
        self.e41 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.e42 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        # max pool 4
        self.mp4 = nn.MaxPool2d(kernel_size=2, stride=2)
        d //= 2
        d -= 2
        embd *= 2

        # encoder level 5
        self.e51 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.e52 = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d*=2
        embd //= 2

        # up conv 4
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2),
            nn.LayerNorm([embd,d,d]),
        )

        d -= 2

        # decoder level 4
        self.d41 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.d42 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d *= 2
        embd //= 2

        # up conv 3
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.LayerNorm([embd,d,d]),
        )

        d -= 2

        # decoder level 3
        self.d31 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.d32 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=0),
            nn.LayerNorm([embd,d,d]),
            nn.ReLU(inplace=True),
        )

        d *= 2
        embd //= 2

        # up conv 2
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.LayerNorm([embd,d,d]),
        )

        d -= 2

        # decoder level 2
        self.d21 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.d22 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        d *= 2
        embd //= 2

        # up conv 1
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.LayerNorm([embd,d,d]),
        )

        d -= 2

        # decoder level `
        self.d11 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        d -= 2

        self.d12 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(inplace=True),
        )

        # final conv for output segmentation
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def crop(self, enc_out, dec_in):
        _, _, h_dec, w_dec = dec_in.size()
        _, _, h_enc, w_enc = enc_out.size()

        # Calculate cropping margins
        crop_top = (h_enc - h_dec) // 2
        crop_left = (w_enc - w_dec) // 2

        # Crop the encoder feature map
        return enc_out[:, :, crop_top:crop_top + h_dec, crop_left:crop_left + w_dec]

    def forward(self, x):
        # encoder
        e1_out = self.e12(self.e11(x))

        e2_out = self.e22(self.e21(self.mp1(e1_out)))

        e3_out = self.e32(self.e31(self.mp2(e2_out)))

        e4_out = self.e42(self.e41(self.mp3(e3_out)))

        out =  self.mp4(e4_out)
        e5_out = self.e52(self.e51(out))

        # decoder
        up4_out = self.up4(e5_out)
        d4_out = self.d42(self.d41(torch.cat((self.crop(e4_out, up4_out), up4_out), dim=1)))

        up3_out = self.up3(d4_out)
        d3_out = self.d32(self.d31(torch.cat((self.crop(e3_out, up3_out), up3_out), dim=1)))

        up2_out = self.up2(d3_out)
        d2_out = self.d22(self.d21(torch.cat((self.crop(e2_out, up2_out), up2_out), dim=1)))

        up1_out = self.up1(d2_out)
        d1_out = self.d12(self.d11(torch.cat((self.crop(e1_out, up1_out), up1_out), dim=1)))

        return self.final_conv(d1_out)