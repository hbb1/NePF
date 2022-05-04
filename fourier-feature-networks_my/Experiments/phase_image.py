import cv2
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import copy


def time2freq(gridSize):
    ffSize = copy.deepcopy(gridSize)
    ffSize[-1] = ffSize[-1] // 2 + 1
    return ffSize


def dft(phasors, inputs, T=None, dim=-1):
    # inputs should be in [0,1]                      # F(f(ax)) = 1/|a| P(w/a)
    phasors = phasors.transpose(dim, -1)
    device = phasors.device
    inputs = inputs * (T - 1) / T  # to match torch.fft.fft
    N = phasors.shape[-1]  # frequency domain scaling
    pf = torch.arange(0, (N + 1) // 2).to(device)  # positive freq
    nf = torch.arange(-(N - 1) // 2, 0).to(device)  # negative freq
    fk = torch.concat([pf, nf])  # sampling frequencies
    inputs = inputs.reshape(-1, 1).to(device)
    M = torch.exp(2j * np.pi * inputs * fk).to(device)
    out = F.linear(phasors, M)  # integrate phasors
    out = out.transpose(dim, -1)  # transpose back
    return out


def rdft(phasors, inputs, T=None, dim=-1):
    phasors = phasors.transpose(dim, -1)
    device = phasors.device
    inputs = inputs * (T - 1) / T  # to match torch.fft.fft
    N = phasors.shape[-1]
    pf = torch.arange(N).to(device)  # positive freq only
    fk = pf  # sampling frequencies
    inputs = inputs.reshape(-1, 1).to(device)
    M = torch.exp(2j * np.pi * inputs * fk).to(device)
    # index in pytorch is slow
    # M[:, 1:] = M[:, 1:] * 2                          # Hermittion symmetry
    M = M * ((fk > 0) + 1)[None]
    out = F.linear(phasors.real, M.real) - F.linear(phasors.imag, M.imag)
    out = out.transpose(dim, -1)
    return out


def irdft3d(phasors, gridSize):
    device = phasors.device
    ifft_crop = phasors
    Nx, Ny, Nz = gridSize
    xx, yy, zz = [torch.linspace(0, 1, N).to(device) for N in gridSize]
    ifft_crop = dft(ifft_crop, xx, Nx, dim=2)
    ifft_crop = dft(ifft_crop, yy, Ny, dim=3)
    ifft_crop = rdft(ifft_crop, zz, Nz, dim=4)
    return ifft_crop


def irdft2d(phasors, gridSize):
    device = phasors.device
    ifft_crop = phasors
    Nx, Ny = gridSize
    xx, yy = [torch.linspace(0, 1, N).to(device) for N in gridSize]
    ifft_crop = dft(ifft_crop, xx, Nx, dim=0)
    ifft_crop = rdft(ifft_crop, yy, Ny, dim=1)
    return ifft_crop


class PhaseImage(nn.Module):
    def __init__(self, image=None, url=None, H=0, W=0, device="cuda"):
        super(PhaseImage, self).__init__()
        self.device = device
        # load image
        if image is not None:
            self.image = image
        elif url is not None:
            self.image = cv2.imread(url) / 255.0
        else:
            raise ValueError("image or url must be set")
        self.img_ten = torch.from_numpy(self.image * 2 - 1).to(self.device)
        self.H, self.W = self.image.shape[:2]
        # init phasor space
        self.kspace = torch.nn.Parameter(
            torch.zeros(*time2freq([self.H, self.W]), 3).to(torch.complex64).to(device)
        )

    def normal_fft2(self, shift=False):
        phase = torch.fft.rfft(self.img_ten, dim=1) / self.W
        phase = torch.fft.fft(phase, dim=0) / self.H
        if shift:
            phase = torch.fft.fftshift(phase, dim=[0])
        return phase

    def normal_ifft2(self, kspace=None, image_size=None):
        kspace = torch.fft.ifftshift(kspace, dim=0)
        out = torch.fft.ifft(kspace, dim=0) * image_size[0]
        out = torch.fft.irfft(out, dim=1) * image_size[1]
        return out

    def infer_image(self, image_size=None, kspace=None):
        out = self.forward(image_size, kspace)
        image = np.array(255.0 * (out.cpu().numpy() + 1) / 2, dtype=np.uint8)
        return image

    def forward(self, image_size=None, kspace=None):
        if image_size is None:
            image_size = [self.H, self.W]
        if kspace is None:
            kspace = self.kspace
        out = []
        for k in torch.split(kspace, 1, dim=-1):
            out.append(irdft2d(k.squeeze(-1), image_size))
            # out.append(self.normal_ifft2(k.squeeze(-1), image_size))
        out = torch.stack(out, dim=-1)
        return out

    def upsample(self, rate=2, image_size=None, kspace=None):
        if image_size is None:
            image_size = [self.H, self.W]
        if kspace is None:
            kspace = self.kspace
        Nx, Ny = map(lambda x: x * rate, image_size)
        xx = torch.linspace(0, 1, int(Nx))[
            (int(Nx) - image_size[0]) // 2 : (int(Nx) + image_size[0]) // 2
        ].to(self.device)
        yy = torch.linspace(0, 1, int(Ny))[
            (int(Ny) - image_size[1]) // 2 : (int(Ny) + image_size[1]) // 2
        ].to(self.device)
        ifft_crop = kspace
        ifft_crop = dft(ifft_crop, xx, Nx, dim=0)
        ifft_crop = rdft(ifft_crop, yy, Ny, dim=1)
        return ifft_crop


def main():
    torch.set_default_dtype(torch.float32)
    phase_img = PhaseImage(url="./cat.jpg")
    # kspace = phase_img.normal_fft2()
    # img = phase_img.infer_image(kspace=kspace)
    # cv2.imwrite("./res.png", img)
    idx = 0
    for i in np.linspace(1, 16, num=100):
        image = 255 * (phase_img.upsample(rate=i).detach().cpu().numpy() + 1) / 2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"x{i}")
        cv2.imwrite("./results/x16/%05d.png" % (idx), image)
        idx += 1


if __name__ == "__main__":
    main()
