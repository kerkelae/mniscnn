import math

from dipy.core.gradients import gradient_table
import healpy as hp
import numpy as np
from scipy.special import sph_harm
import torch


l_max = 8
n_coeffs = int(0.5 * (l_max + 1) * (l_max + 2))

ls = torch.zeros(n_coeffs, dtype=int)
l0s = torch.zeros(n_coeffs, dtype=int)
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        ls[int(0.5 * l * (l + 1) + m)] = l
        l0s[int(0.5 * l * (l + 1) + m)] = int(0.5 * l * (l + 1))


def sh(l, m, thetas, phis):
    """Real and symmetric spherical harmonic basis function.

    Parameters
    ----------
    l : array_like
        Degree of the harmonic.
    m : array_like
        Order of the harmonic.
    thetas : array_like
        Polar coordinate.
    phis : array_like
        Azimuthal coordinate.

    Returns
    -------
    float or numpy.ndarray
        The harmonic sampled at `phi` and `theta`.
    """
    if l % 2 == 1:
        return np.zeros(len(phis))
    if m < 0:
        return np.sqrt(2) * sph_harm(-m, l, phis, thetas).imag
    if m == 0:
        return sph_harm(m, l, phis, thetas).real
    if m > 0:
        return np.sqrt(2) * sph_harm(m, l, phis, thetas).real


_n_sides = 2**3
_x, _y, _z = hp.pix2vec(_n_sides, np.arange(12 * _n_sides**2))
vertices = np.vstack((_x, _y, _z)).T
_thetas = np.arccos(vertices[:, 2])
_phis = np.arctan2(vertices[:, 1], vertices[:, 0]) + np.pi
isft = np.zeros((len(vertices), n_coeffs))
for l in range(0, l_max + 1, 2):
    for m in range(-l, l + 1):
        isft[:, int(0.5 * l * (l + 1) + m)] = sh(l, m, _thetas, _phis)
sft = np.linalg.pinv(isft.T @ isft) @ isft.T
sft = torch.tensor(sft).float()
isft = torch.tensor(isft).float()

_rf_btens_lte = torch.tensor(
    gradient_table(np.ones(len(vertices)), vertices, btens="LTE").btens
).float()
_rf_btens_pte = torch.tensor(
    gradient_table(np.ones(len(vertices)), vertices, btens="PTE").btens
).float()


def compartment_model_simulation(
    bval, bvecs_isft, ads, rds, fs, odfs, bten_shape, device
):
    """Simulate diffusion-weighted MR measurements.

    Parameters
    ----------
    bval : int
        Simulated b-value.
    bvecs_isft : torch.tensor
        Tensor with shape (number of simulated b-vectors, number of
        coefficients) for evaluating signal values from the spherical harmonic
        expansion coefficients.
    ads : torch.tensor
        Tensor with shape (number of simulations, number of compartments)
        containing axial diffusivities.
    rds : torch.tensor
        Tensor with shape (number of simulations, number of compartments)
        containing radial diffusivities.
    fs : torch.tensor
        Tensor with shape (number of simulations, number of compartments)
        containing compartment signal fractions.
    odfs : torch.tensor
        Tensor with shape (number of simulations, number of coefficients)
        containing the SH coefficients of the ODFs.
    bten_shape : {'linear', 'planar'}
        Shape of the simulated b-tensor.
    device : {'cpu', 'cuda'}
        Device on which to run the simulation.

    Returns
    -------
    torch.tensor
        Simulated signals.
    """
    n_simulations = ads.shape[0]
    n_compartments = ads.shape[1]
    Ds = torch.zeros(n_simulations, n_compartments, 3, 3).to(device)
    Ds[:, :, 2, 2] = ads  # aligned with the z-axis
    Ds[:, :, 1, 1] = rds
    Ds[:, :, 0, 0] = rds
    if bten_shape == "linear":
        response = torch.sum(
            fs.to(device).unsqueeze(2)
            * torch.exp(
                -torch.sum(
                    bval
                    * _rf_btens_lte.to(device).unsqueeze(0).unsqueeze(1)
                    * Ds.unsqueeze(2),
                    dim=(3, 4),
                )
            ),
            dim=1,
        )
    elif bten_shape == "planar":
        response = torch.sum(
            fs.to(device).unsqueeze(2)
            * torch.exp(
                -torch.sum(
                    bval
                    * _rf_btens_pte.to(device).unsqueeze(0).unsqueeze(1)
                    * Ds.unsqueeze(2),
                    dim=(3, 4),
                )
            ),
            dim=1,
        )
    response_sh = (sft.to(device) @ response.unsqueeze(-1)).squeeze(-1)
    convolution_sh = (
        torch.sqrt(4 * math.pi / (2 * ls.to(device) + 1))
        * odfs.to(device)
        * response_sh[:, l0s]
    )
    simulated_signals = bvecs_isft.to(device) @ convolution_sh.unsqueeze(-1)
    return simulated_signals


class SphConv(torch.nn.Module):
    """Layer for performing a spherical convolution with a zonal filter."""

    def __init__(self, c_in, c_out):
        """Create a new layer.

        Parameters
        -----------
        c_in : int
            Number of input channels.
        c_out : int
            Number of output channels.
        """
        super().__init__()
        self.l_max = l_max
        self.c_in = c_in
        self.c_out = c_out
        ls = torch.zeros(n_coeffs, dtype=int)
        for l in range(0, l_max + 1, 2):
            for m in range(-l, l + 1):
                ls[int(0.5 * l * (l + 1) + m)] = l
        self.register_buffer("ls", ls)
        self.weights = torch.nn.Parameter(
            torch.Tensor(self.c_out, self.c_in, int(self.l_max / 2) + 1)
        )
        torch.nn.init.uniform_(self.weights)

    def forward(self, x):
        """Make a forward pass.

        Parameters
        ----------
        x : torch.tensor
            Signal to be convoluted in a tensor with shape (batch_size, number
            of channels, number of coefficients).

        Returns
        -------
        torch.tensor
            Convoluted signals.
        """
        weights_exp = self.weights[:, :, (self.ls / 2).long()]
        ys = torch.sum(
            torch.sqrt(
                math.pi / (2 * self.ls.unsqueeze(0).unsqueeze(0).unsqueeze(0) + 1)
            )
            * weights_exp.unsqueeze(0)
            * x.unsqueeze(1),
            dim=2,
        )
        return ys
