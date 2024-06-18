import torch
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

def ellipse_ndim(mean, cov, ax, edgecolor):
    for m, c in zip(mean, cov):
        ellipse(m, c, ax, edgecolor=edgecolor)

def ellipse(mean, cov, ax, n_std=1.0, facecolor='none', **kwargs):

    pearson = cov[0, 1]/torch.sqrt(cov[0, 0] * cov[1, 1])
    
    ell_radius_x = torch.sqrt(1 + pearson)
    ell_radius_y = torch.sqrt(1 - pearson)
    ellipse = Ellipse((0, 0), width=ell_radius_x * 2, height=ell_radius_y * 2,
                      facecolor=facecolor, **kwargs)

    scale_x = torch.sqrt(cov[0, 0]) * n_std
    mean_x = mean[0]

    scale_y = torch.sqrt(cov[1, 1]) * n_std
    mean_y = mean[1]

    transf = transforms.Affine2D() \
        .rotate_deg(45) \
        .scale(scale_x, scale_y) \
        .translate(mean_x, mean_y)

    ellipse.set_transform(transf + ax.transData)
    return ax.add_patch(ellipse)

def camera_intr(x_0, y_0, f_x, f_y, s):
    transl = torch.tensor([
        [1., 0., x_0],
        [0., 1., y_0],
        [0., 0., 1  ],
    ])
    scale = torch.tensor([
        [f_x, 0. , 0.],
        [0. , f_y, 0.],
        [0. , 0. , 1.],
    ])
    shear = torch.tensor([
        [1., s/f_x, 0.],
        [0., 1.   , 0.],
        [0., 0.   , 1.],
    ])

    return transl @ scale @ shear