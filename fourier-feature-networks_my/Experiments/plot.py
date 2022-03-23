# coding=utf-8
"""
Author: yanxinhao
Email: 1914607611xh@i.shu.edu.cn
LastEditTime: 2021-09-22 13:56:44
LastEditors: yanxinhao
FilePath: /NDface/core/utils/plot.py
Date: 2021-09-22 13:56:44
Description: 
"""
import os
import torch
import numpy as np
import plotly.graph_objs as go
import plotly.offline as offline
from skimage import measure
import torchvision
from PIL import Image


def to_cuda(torch_obj):
    if torch.cuda.is_available():
        return torch_obj.cuda()
    else:
        return torch_obj


def get_threed_scatter_trace(points, caption=None, colorscale=None, color=None):

    if type(points) == list:
        trace = [
            go.Scatter3d(
                x=p[0][:, 0],
                y=p[0][:, 1],
                z=p[0][:, 2],
                mode="markers",
                name=p[1],
                marker=dict(
                    size=3,
                    line=dict(width=2,),
                    opacity=0.9,
                    colorscale=colorscale,
                    showscale=True,
                    color=color,
                ),
                text=caption,
            )
            for p in points
        ]

    else:

        trace = [
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                name="projection",
                marker=dict(
                    size=3,
                    line=dict(width=2,),
                    opacity=0.9,
                    colorscale=colorscale,
                    showscale=True,
                    color=color,
                ),
                text=caption,
            )
        ]

    return trace


def plot_surface(
    decoder,
    path,
    iteration,
    shapename,
    resolution,
    mc_value,
    is_uniform_grid,
    verbose,
    save_html,
    save_ply,
    overwrite,
    points=None,
    with_points=False,
    latent=None,
    connected=False,
    cube_length=1.0,
):

    filename = "{0}/{1}_{2}".format(path, iteration, shapename)

    if not os.path.exists(filename) or overwrite:

        if with_points:
            pnts_val = decoder(points)
            pnts_val = pnts_val.cpu()
            points = points.cpu()
            caption = ["decoder : {0}".format(val.item()) for val in pnts_val.squeeze()]
            trace_pnts = get_threed_scatter_trace(points[:, -3:], caption=caption)

        surface = get_surface_trace(
            points,
            decoder,
            latent,
            resolution,
            mc_value,
            is_uniform_grid,
            verbose,
            save_ply,
            connected,
            cube_length=cube_length,
        )
        trace_surface = surface["mesh_trace"]

        layout = go.Layout(
            title=go.layout.Title(text=shapename),
            width=1200,
            height=1200,
            scene=dict(
                xaxis=dict(range=[-2, 2], autorange=False),
                yaxis=dict(range=[-2, 2], autorange=False),
                zaxis=dict(range=[-2, 2], autorange=False),
                aspectratio=dict(x=1, y=1, z=1),
            ),
        )
        if with_points:
            fig1 = go.Figure(data=trace_pnts + trace_surface, layout=layout)
        else:
            fig1 = go.Figure(data=trace_surface, layout=layout)

        if save_html:
            offline.plot(fig1, filename=filename + ".html", auto_open=False)
        if not surface["mesh_export"] is None:
            surface["mesh_export"].export(filename + ".ply", "ply")
        return surface["mesh_export"], surface["volume"]


def get_surface_trace(
    points,
    decoder,
    latent,
    resolution,
    mc_value,
    is_uniform,
    verbose,
    save_ply,
    connected=False,
    cube_length=1.0,
):

    trace = []
    meshexport = None

    if is_uniform:
        grid = get_grid_uniform(resolution, cube_length=cube_length)
    else:
        if not points is None:
            grid = get_grid(points[:, -3:], resolution)
        else:
            grid = get_grid(None, resolution)

    z = []

    for i, pnts in enumerate(torch.split(grid["grid_points"], 20000, dim=0)):
        if verbose:
            print("{0}".format(i / (grid["grid_points"].shape[0] // 20000) * 100))

        if not latent is None:
            pnts = torch.cat([latent.expand(pnts.shape[0], -1), pnts], dim=1)
        z.append(decoder(pnts).detach().cpu().numpy())
        torch.cuda.empty_cache()
    z = np.concatenate(z, axis=0)

    if not (np.min(z) > mc_value or np.max(z) < mc_value):

        import trimesh

        z = z.astype(np.float64)
        volume = z.reshape(
            grid["xyz"][1].shape[0], grid["xyz"][0].shape[0], grid["xyz"][2].shape[0],
        ).transpose([1, 0, 2])
        verts, faces, normals, values = measure.marching_cubes_lewiner(
            volume=z.reshape(
                grid["xyz"][1].shape[0],
                grid["xyz"][0].shape[0],
                grid["xyz"][2].shape[0],
            ).transpose([1, 0, 2]),
            level=mc_value,
            spacing=(
                grid["xyz"][0][2] - grid["xyz"][0][1],
                grid["xyz"][0][2] - grid["xyz"][0][1],
                grid["xyz"][0][2] - grid["xyz"][0][1],
            ),
        )

        verts = verts + np.array(
            [grid["xyz"][0][0], grid["xyz"][1][0], grid["xyz"][2][0]]
        )
        if save_ply:
            meshexport = trimesh.Trimesh(verts, faces, normals, vertex_colors=values)
            if connected:
                connected_comp = meshexport.split(only_watertight=False)
                max_area = 0
                max_comp = None
                for comp in connected_comp:
                    if comp.area > max_area:
                        max_area = comp.area
                        max_comp = comp
                meshexport = max_comp

        def tri_indices(simplices):
            return ([triplet[c] for triplet in simplices] for c in range(3))

        I, J, K = tri_indices(faces)

        trace.append(
            go.Mesh3d(
                x=verts[:, 0],
                y=verts[:, 1],
                z=verts[:, 2],
                i=I,
                j=J,
                k=K,
                name="",
                color="orange",
                opacity=0.5,
            )
        )

    return {"mesh_trace": trace, "mesh_export": meshexport, "volume": volume}


def get_grid(points, resolution):
    eps = 0.1
    input_min = torch.min(points, dim=0)[0].squeeze().cpu().numpy()
    input_max = torch.max(points, dim=0)[0].squeeze().cpu().numpy()
    bounding_box = input_max - input_min
    shortest_axis = np.argmin(bounding_box)
    if shortest_axis == 0:
        x = np.linspace(
            input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution
        )
        length = np.max(x) - np.min(x)
        y = np.arange(
            input_min[1] - eps,
            input_max[1] + length / (x.shape[0] - 1) + eps,
            length / (x.shape[0] - 1),
        )
        z = np.arange(
            input_min[2] - eps,
            input_max[2] + length / (x.shape[0] - 1) + eps,
            length / (x.shape[0] - 1),
        )
    elif shortest_axis == 1:
        y = np.linspace(
            input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution
        )
        length = np.max(y) - np.min(y)
        x = np.arange(
            input_min[0] - eps,
            input_max[0] + length / (y.shape[0] - 1) + eps,
            length / (y.shape[0] - 1),
        )
        z = np.arange(
            input_min[2] - eps,
            input_max[2] + length / (y.shape[0] - 1) + eps,
            length / (y.shape[0] - 1),
        )
    elif shortest_axis == 2:
        z = np.linspace(
            input_min[shortest_axis] - eps, input_max[shortest_axis] + eps, resolution
        )
        length = np.max(z) - np.min(z)
        x = np.arange(
            input_min[0] - eps,
            input_max[0] + length / (z.shape[0] - 1) + eps,
            length / (z.shape[0] - 1),
        )
        y = np.arange(
            input_min[1] - eps,
            input_max[1] + length / (z.shape[0] - 1) + eps,
            length / (z.shape[0] - 1),
        )

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(
        np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float
    ).cuda()
    return {
        "grid_points": grid_points,
        "shortest_axis_length": length,
        "xyz": [x, y, z],
        "shortest_axis_index": shortest_axis,
    }


def get_grid_uniform(resolution, cube_length=1.0):
    x = np.linspace(-cube_length, cube_length, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = to_cuda(
        torch.tensor(
            np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float
        )
    )

    return {
        "grid_points": grid_points,
        "shortest_axis_length": 2.4,
        "xyz": [x, y, z],
        "shortest_axis_index": 0,
    }


def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = (ground_true + 1.0) / 2.0
    rgb_points = (rgb_points + 1.0) / 2.0
    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = (
        torchvision.utils.make_grid(
            output_vs_gt_plot, scale_each=False, normalize=False, nrow=plot_nrow
        )
        .cpu()
        .detach()
        .numpy()
    )
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save("{0}/rendering_{1}.png".format(path, epoch))


def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])
