import torch
from tqdm import tqdm

### Project ##################################################################

def P(f_x, f_y, h, w, n, f):
    P = torch.tensor([
        [2.*f_x/w, 0., 0., 0.],
        [0., 2.*f_y/h, 0., 0.],
        [0., 0., (f+n)/(f-n), -2*f*n/(f-n)],
        [0., 0., 1., 0.],
    ], device='cuda')

    return P

def J(f_x, f_y, t_x, t_y, t_z):
    N = len(t_x)
    J = torch.zeros((N,2,3), device='cuda')
    J[:,0,0] = f_x/t_z
    J[:,1,1] = f_y/t_z
    J[:,0,2] = -f_x*t_x/t_z**2
    J[:,1,2] = -f_y*t_y/t_z**2

    return J

def quat_to_rot(quaternion):
    N = quaternion.shape[0]
    x, y, z, w = quaternion[:,0],quaternion[:,1],quaternion[:,2],quaternion[:,3],

    R = torch.empty((N,3,3), device='cuda')
    
    R[:,0,0] = 1-2*(y**2+z**2)
    R[:,0,1] = 2*(x*y-w*z)
    R[:,0,2] = 2*(x*z+w*y)

    R[:,1,0] = 2*(x*y+w*z)
    R[:,1,1] = 1-2*(x**2-z**2)
    R[:,1,2] = 2*(y*z-w*x)

    R[:,2,0] = 2*(x*z+w*y)
    R[:,2,1] = 2*(y*z+w*x)
    R[:,2,2] = 1-2*(x**2+y**2)

    return R

def project_gaussians(
        means3d, 
        scales, 
        # glob_scale, 
        quats, 
        viewmat, 
        fx, 
        fy, 
        cx, 
        cy, 
        img_height, 
        img_width, 
        # block_width, 
        clip_thresh=0.01
    ):
    N = means3d.shape[0]

    # Project Means
    t = viewmat @ torch.cat((means3d.T, torch.ones(1, N, device='cuda')), dim=0) # (4, 4) x (4, N) = (4, N)
    t_ = P(fx, fy, img_height, img_width, clip_thresh, 10) @ t # (4, 4) x (4, N) = (4, N)

    xys = torch.vstack((
        (img_width*t_[0]/t_[3])/2+cx, # old: (img_width*t_[0]/t_[3]+1.)/2+cx,
        (img_height*t_[1]/t_[3])/2+cy, # (img_height*t_[1]/t_[3]+1.)/2+cy
    )).T
    depths = t_[2]

    # Scale + Rot. to Cov.
    R = quat_to_rot(quats); S = torch.cat([torch.diag(s)[None] for s in scales])
    RS =  R @ S
    Sigma = RS @ RS.permute(0,2,1)

    # Project Cov
    J_ = J(fx, fy, t[0], t[1], t[2])
    R_cw = viewmat[:3,:3]
    covs = J_ @ R_cw @ Sigma @ R_cw.T @ J_.permute((0,2,1))

    return xys, covs, depths 

    # return xys, depths, radii, conics, compensation, num_tiles_hit, cov3d

### Rasterize ##################################################################

def inv_2d(A):
    A_inv = torch.tensor([
        [A[1,1], -A[0,1]],
        [-A[1,0], A[0,0]],
    ], device='cuda')
    A_inv *= 1/(A[0,0]*A[1,1]-A[0,1]*A[1,0])

    return A_inv

def g(x, m, S):
    ''' x: (h*w, 2) matrix
        m: (2, 1) mean
        S: (2, 2) cov matrix
    '''
    
    x = x.T.view(-1, 1, 2)
    m = m.view(1, 1, 2)

    S_inv = inv_2d(S)
    x_m = x - m

    return torch.exp(-(1/2)*x_m @ S_inv @ x_m.permute(0,2,1))

def rasterize_gaussians(
        xys, 
        depths, 
        covs, 
        conics, 
        num_tiles_hit, 
        colors, 
        opacity, 
        img_height, 
        img_width, 
        block_width, 
        background=None, 
        return_alpha=False
    ):
    x, y = torch.meshgrid(torch.linspace(0,img_width,img_width),torch.linspace(0,img_height,img_height), indexing='xy')
    x = x.reshape(1,-1).to('cuda'); y = y.reshape(1, -1).to('cuda')

    # Sort mu_ by depth
    _, ind = torch.sort(depths)
    xys, covs, colors = xys[ind], covs[ind], colors[ind]

    out_img = torch.zeros(img_height, img_width, 3, device='cuda')
    pixels_xy = torch.cat((x.reshape(1,-1),y.reshape(1,-1)), dim=0)
    cum_alphas = torch.ones(1, img_height, img_width, device='cuda')
    for m, S, c, o in tqdm(zip(xys, covs, colors, opacity), total=len(xys)):

        alpha = g(pixels_xy, m, S).view(1, img_height, img_width) * o
        out_img += (alpha * c.view(3,1,1) * cum_alphas).permute((1,2,0)).flip(dims=(0,))
        cum_alphas *= (1 - alpha)

    return out_img

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    N = 1_000

    mu = (torch.rand((N,3), device=device) - 0.5) * 4.
    scale = torch.rand((N,3), device=device) * 0.2
    quat = torch.rand((N, 4), device=device)
    col = torch.rand((N, 3), device=device)
    opc = torch.rand((N,), device=device)

    # Output Image Width and Height
    W = 1290 
    H =  720

    fov_x = math.pi / 2.0 # Angle of the camera frustum 90°
    focal = 0.5 * float(W) / math.tan(0.5 * fov_x) # Distance to Image Plane

    viewmat = torch.eye(4, device=device)
    viewmat[:3,3] = torch.tensor([0,0,-4])

    (
        mu_,
        cov_,
        z
    ) = project_gaussians(
        means3d=mu,
        scales=scale,
        quats=quat,
        viewmat=viewmat,
        fx=focal,
        fy=focal,
        cx=W/2,
        cy=H/2,
        img_height=H,
        img_width=W
    )

    out_img = rasterize_gaussians(
        xys=mu_,
        depths=z,
        covs=cov_,
        conics=None,
        num_tiles_hit=None,
        colors=col,
        opacity=opc,
        img_height=H,
        img_width=W,
        block_width=None,
        background=None,
        return_alpha=None
    )

    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.matshow(out_img.cpu().detach())

    ellipse_ndim(mu_, cov_, ax2, edgecolor='red')
    for m in mu_:
        ax2.scatter(m[0].cpu(), m[1].cpu())
    # ax2.scatter(mu_[1,0], mu_[1,1])
    ax2.set_aspect('equal', adjustable='box')

    ax2.set_xlim([0,W])
    ax2.set_ylim([0,H])

    plt.show()

if __name__ == '__main__':
    import torch
    import matplotlib.pyplot as plt
    import math
    from utils import ellipse_ndim
    from tqdm import tqdm

    main()