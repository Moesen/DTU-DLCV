# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Generate images using pretrained network pickle."""
from projects.utils import get_project2_root
from projects.project2.src.visualization.SVM_classifier import get_latent_direction, plot_latent_direction


import os
import re
from typing import List, Optional

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
import matplotlib.pyplot as plt

import glob

import legacy

# from projects.project2.stylegan2-ada-pytorch import dnnlib
# from projects.project2.stylegan2-ada-pytorch import legacy


#----------------------------------------------------------------------------

def num_range(s: str) -> List[int]:
    '''Accept either a comma separated list of numbers 'a,b,c' or a range 'a-c' and return as a list of ints.'''

    range_re = re.compile(r'^(\d+)-(\d+)$')
    m = range_re.match(s)
    if m:
        return list(range(int(m.group(1)), int(m.group(2))+1))
    vals = s.split(',')
    return [int(x) for x in vals]

#----------------------------------------------------------------------------

@click.command()
@click.pass_context
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--seeds', type=num_range, help='List of random seeds')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=1, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--noise-mode', help='Noise mode', type=click.Choice(['const', 'random', 'none']), default='const', show_default=True)
@click.option('--projected-w', help='Projection result file', type=str, metavar='FILE')
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
def generate_images(
    ctx: click.Context,
    network_pkl: str,
    seeds: Optional[List[int]],
    truncation_psi: float,
    noise_mode: str,
    outdir: str,
    class_idx: Optional[int],
    projected_w: Optional[str]
):
    """Generate images using pretrained network pickle.
    Examples:
    \b
    # Generate curated MetFaces images without truncation (Fig.10 left)
    python generate.py --outdir=out --trunc=1 --seeds=85,265,297,849 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    \b
    # Generate uncurated MetFaces images with truncation (Fig.12 upper left)
    python generate.py --outdir=out --trunc=0.7 --seeds=600-605 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    \b
    # Generate class conditional CIFAR-10 images (Fig.17 left, Car)
    python generate.py --outdir=out --seeds=0-35 --class=1 \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/cifar10.pkl
    \b
    # Render an image from projected W
    python generate.py --outdir=out --projected_w=projected_w.npz \\
        --network=https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metfaces.pkl
    """
    mag1_max = 10
    mag2_max = 50#20
    mag2glasses_max = 70
    plot_ref = [10,10]
    plot_mag = 50

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore


    PROJECT_ROOT = get_project2_root()
    #ld_path =  PROJECT_ROOT / "data/stylegan2directions/age.npy"
    ld_path =  PROJECT_ROOT / "data/projected_w_markus.npz"
    ldd = np.load(ld_path)
    ld = torch.from_numpy(ldd).to(device)

    seed = seeds
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    #z = torch.randn([1, G.z_dim]).to(device)
    w = G.mapping(z,None) 

    w = w[:,0,:]
    w = w.repeat(18,1)
    w = w.unsqueeze(0)

    #initial image
    img = G.synthesis(w)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil13 = PIL.Image.fromarray(img[0].cpu().numpy().squeeze(), 'RGB')

    #plotting
    fig, axs = plt.subplots(1,4, figsize=(15,5))
    axs[0].imshow( pil13 )

    #older image
    mag1 = list(range(1,4))
    mag1 = [(x / 3)*mag1_max for x in mag1]

    for n,m1 in enumerate(mag1):
        #older image
        proj_w = w + m1*(ld[0,:].repeat(18,1).unsqueeze(0))
        img = G.synthesis(proj_w)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil23 = PIL.Image.fromarray(img[0].cpu().numpy().squeeze(), 'RGB')

        axs[n+1].imshow( pil23 )

        axs[n+1].grid(False)
        axs[n+1].axis('off')

    save_path =  PROJECT_ROOT / "reports/figures2.png"
    plt.savefig(save_path)

    #proj_w = w + mag1*(ld[0,:].repeat(18,1).unsqueeze(0))
    #img = G.synthesis(proj_w)
    #img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    #pil23 = PIL.Image.fromarray(img[0].cpu().numpy().squeeze(), 'RGB')


    #z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)
    #img = G(z, label, truncation_psi=truncation_psi, noise_mode=noise_mode)
    #img = G.synthesis(w.unsqueeze(0), noise_mode=noise_mode)

    labels = ["hair","glasses"]
    data_path =  PROJECT_ROOT / "data" 

    w_paths1 = glob.glob((data_path / (labels[0]+"_w_out_*") / "projected_w.npz").as_posix())
    w_paths2 = glob.glob((data_path / (labels[1]+"_w_out_*") / "projected_w.npz").as_posix())

    ws = []
    w_labels = []

    for wp in w_paths1:
        w = np.load(wp)['w']
        w = w[0,0,:]
        #ws = ws.repeat(18,1)
        #ws = ws.unsqueeze(0)
        ws.append(w)
        w_labels.append( 0 )
    
    for wp in w_paths2:
        w = np.load(wp)['w']
        w = w[0,0,:]
        #ws = ws.repeat(18,1)
        #ws = ws.unsqueeze(0)
        ws.append(w)
        w_labels.append( 1 )
    
    X = np.array(ws)
    y = np.array(w_labels)

    w = get_latent_direction(X, y)
    
    ld1 = torch.from_numpy(w).to(device)


    #####################
    labels = ["hair","bald"]
    data_path =  PROJECT_ROOT / "data" 

    w_paths1 = glob.glob((data_path / (labels[0]+"_w_out_*") / "projected_w.npz").as_posix())
    w_paths2 = glob.glob((data_path / (labels[1]+"_w_out_*") / "projected_w.npz").as_posix())

    ws = []
    w_labels = []

    for wp in w_paths1:
        w = np.load(wp)['w']
        w = w[0,0,:]
        #ws = ws.repeat(18,1)
        #ws = ws.unsqueeze(0)
        ws.append(w)
        w_labels.append( 0 )
    
    for wp in w_paths2:
        w = np.load(wp)['w']
        w = w[0,0,:]
        #ws = ws.repeat(18,1)
        #ws = ws.unsqueeze(0)
        ws.append(w)
        w_labels.append( 1 )
    
    X = np.array(ws)
    y = np.array(w_labels)

    w = get_latent_direction(X, y)
    
    ld = torch.from_numpy(w).to(device)


    #generate a normal image
    #z = torch.randn([1, G.z_dim]).to(device)
    w = G.mapping(z,None) 

    w = w[:,0,:]
    w = w.repeat(18,1)
    w = w.unsqueeze(0)

    #initial image
    img = G.synthesis(w)
    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    pil13 = PIL.Image.fromarray(img[0].cpu().numpy().squeeze(), 'RGB')

    #plotting
    fig, axs = plt.subplots(1,4, figsize=(15,5))
    axs[0].imshow( pil13 )
    axs[0].grid(False)
    axs[0].axis('off')

    mag2 = list(range(1,4))
    mag2 = [(x / 3)*mag2_max for x in mag2]

    mag2g = list(range(1,4))
    mag2g = [(x / 3)*mag2glasses_max for x in mag2g]

    for n,(m2,m2g) in enumerate(zip(mag2,mag2g)):
        #older image
        #proj_w = w + 10*(ld.unsqueeze(0))
        proj_w = w + (m2*ld.repeat(18,1).unsqueeze(0) + m2g*ld1.repeat(18,1).unsqueeze(0))
        img = G.synthesis(proj_w)
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        pil23 = PIL.Image.fromarray(img[0].cpu().numpy().squeeze(), 'RGB')

        axs[n+1].imshow( pil23 )

        axs[n+1].grid(False)
        axs[n+1].axis('off')

    save_path =  PROJECT_ROOT / "reports/figures3.png"
    plt.savefig(save_path)


    plot_latent_direction(X,y,mag=plot_mag,ref=plot_ref,labels=labels)

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

