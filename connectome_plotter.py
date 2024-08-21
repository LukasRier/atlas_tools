# -*- coding: utf-8 -*-
"""
ConnectomePlotter class to create connectome matrices, circle plots and 3D glass brain plots
Created on Fri May 10 14:44:12 2024

@author: ppzlr
"""
import os
import pickle

import numpy as np
import matplotlib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from matplotlib.colors import Normalize
from mpl_toolkits.mplot3d import art3d
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.io import loadmat

from mne_connectivity.viz import plot_connectivity_circle
from mne.viz import circular_layout


class ConnectomePlotter:
    """Connectome plotter class

    Provides plotting functions for connectome matrices, circle, and glass brain plots.
    Needs "atlas_viewer.mat" and "atlas_labels.pkl" files!
    """

    def __init__(
        self,
        connectivity_matrix: np.ndarray,
        colour_range=None,
        atlas: str = None,
        cmap_name: str = "RdBu_r",
        cbar_title: str = "Connectivity",
    ):
        """Initialize the connectome plotter.

        Args:
            connectivity_matrix (np.ndarray): square matrix of connectivity values
            colour_range (int | list[int,int], optional): colour limits. Will be set to +- a given single float or -+ the maximum absolute value in connectivity_matrix if None.
            atlas (str, optional): Atlas name (aal78 or glasser52). Defaults to None and detects based on matrix size (78 =>AAL, 52 => glasser).
            cmap_name (str, optional): numpy colormaps name. Defaults to "RdBu_r".
            cbar_title (str, optional): Title/label for colorbar. Defaults to "Connectivity".

        Raises:
            ValueError: If colour_range is not of len 1 or 2
            ValueError: If matrix is not a 2D square matrix
        """
        self.connectivity_matrix = connectivity_matrix
        np.fill_diagonal(self.connectivity_matrix, np.nan)

        self.colour_range = colour_range
        self.atlas = atlas
        self.cmap_name = cmap_name
        self.cmap = matplotlib.colormaps[self.cmap_name]
        if cbar_title is None:
            self.cbar_title = self.atlas
        else:
            self.cbar_title = cbar_title

        if colour_range is None:
            vlim = np.nanmax(np.abs(self.connectivity_matrix))
            self.colour_range = [-vlim, vlim]
        elif len(colour_range) == 1:
            self.colour_range = [-colour_range[0], colour_range[0]]
        elif len(colour_range) > 2:
            raise ValueError("colour_range must be list of length 1 or 2")

        if self.connectivity_matrix.shape[0] != self.connectivity_matrix.shape[1]:
            raise ValueError("Connectivity matrix should be 2D square matrix")

        atlas_labels_file = os.path.join(os.path.dirname(__file__), "atlas_labels.pkl")
        with open(atlas_labels_file, "rb") as f:
            atlas_labels = pickle.load(f)

        if atlas is None and self.connectivity_matrix.shape[0] == 52:
            self.atlas = "glasser52"
        elif atlas is None and self.connectivity_matrix.shape[0] == 78:
            self.atlas = "aal78"
        elif atlas.lower() == "glasser52":
            assert self.connectivity_matrix.shape[0] == 52
        elif atlas.lower() == "aal78":
            assert self.connectivity_matrix.shape[0] == 78

        self.labels = atlas_labels[self.atlas]["labels"]
        self.node_order = atlas_labels[self.atlas]["order"]

    def plot_matrix(self, figsize=(9, 8), title: str = "", block=False):
        """Plot adjacency matrix with labels.

        Args:
            figsize (tuple, optional): Figure size in inches. Defaults to (9, 8).
            title (str, optional): Figure title. Defaults to "".
            block (bool, optional): Block further execution when running matplotlib.pyplot.show. Defaults to False.

        Returns:
            matplotlib.pyplot.figure: Figure object with matrix plot
        """
        fig, ax = plt.subplots(
            squeeze=False,
            figsize=figsize,
            layout="constrained",
        )

        im = ax[0, 0].imshow(
            self.connectivity_matrix,
            interpolation="nearest",
            norm=Normalize(vmin=self.colour_range[0], vmax=self.colour_range[1]),
            cmap=self.cmap,
        )
        ax[0, 0].set(title=title)

        divider = make_axes_locatable(ax[0, 0])
        cax = divider.append_axes("right", size="3.5%", pad=0.05)
        cax.grid(False)  # avoid mpl warning about auto-removal
        # plt.colorbar(im, cax=cax, format="%.0e")
        plt.colorbar(im, cax=cax, format="%.2f", label=self.cbar_title)

        if self.atlas.lower() == "aal78":
            ticks = np.array([5, 14, 25, 37, 44, 53, 64, 76]) - 1
            label_names_arr = np.array(self.labels)
        elif self.atlas.lower() == "glasser52":
            ticks = np.array([2, 7, 10, 13, 17, 23, 28, 33, 36, 39, 43, 49]) - 1
            label_names_arr = np.array(self.labels)

        ax[0, 0].set_xticks(
            ticks=ticks,
            labels=label_names_arr[ticks],
            rotation=45,
            horizontalalignment="right",
        )
        ax[0, 0].set_yticks(
            ticks=ticks,
            labels=label_names_arr[ticks],
            rotation=45,
            verticalalignment="top",
        )
        plt.show(block=block)
        return fig

    def plot_circle(self, top_n: int = 150, title: str = None, block=False):
        """Wrapper for mne_connectivity.viz.plot_connectivity_circle

        Args:
            top_n (int, optional): Number of strongest connections to plot. Defaults to 150.
            title (str, optional): Colorbar title. Defaults to None.
            block (bool, optional): Block further execution when running matplotlib.pyplot.show. Defaults to False.

        Returns:
            matplotlib.pyplot.figure: Figure object with circle plot
        """
        fig, ax = plt.subplots(
            figsize=(8, 8), facecolor="black", subplot_kw=dict(polar=True)
        )
        node_angles = circular_layout(
            self.labels,
            self.node_order[::-1],
            start_pos=90,
            group_boundaries=[0, len(self.labels) / 2],
        )

        plot_connectivity_circle(
            self.connectivity_matrix,
            self.labels,
            n_lines=top_n,
            colormap=self.cmap_name,
            vmin=self.colour_range[0],
            vmax=self.colour_range[1],
            node_angles=node_angles,
            title=title,
            ax=ax,
            show=False,
        )
        fig.tight_layout()
        plt.show(block=block)
        return fig

    def plot_glassbrain(
        self,
        top_n: int = 150,
        view: str = "top_down",
        title: str = "",
        line_scale: int = 5,
        block=False,
    ):
        """Plot connectome as nodes and edges inside transparent brain volume.
        Node sizes are scaled by number of connections and edge thickness is determined by connectivity strength.

        Args:
            top_n (int, optional): Number of strongest connections to plot. Defaults to 150.
            view (str, optional): Set direction of view for 3D plot. Defaults to "top_down".
            title (str, optional): Colorbar title. Defaults to "".
            line_scale (int, optional): Scale value governing edge thicknesses. Defaults to 5.
            block (bool, optional): Block further execution when running matplotlib.pyplot.show. Defaults to False.

        Returns:
            matplotlib.pyplot.figure: Figure object with glass brain plot
        """
        mat = loadmat(os.path.join(os.path.dirname(__file__), "atlas_viewer.mat"))
        faces = mat["aalviewer"][0][0][1] - 1
        vertices = mat["aalviewer"][0][0][2]

        if self.atlas == "aal78":
            centroids = mat["aalviewer"][0][0][0]
        elif self.atlas == "glasser52":
            centroids = mat["aalviewer"][0][0][3]

        # mne.viz.set_3d_backend("pyvistaqt")

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")

        pc = art3d.Poly3DCollection(vertices[faces], linewidths=0, alpha=0.05)
        face_color = [0.5, 0.5, 0.5]
        face_color = [0, 0, 0]
        pc.set_facecolor(face_color)
        ax.add_collection(pc)

        conn_inds = np.triu_indices(self.connectivity_matrix.shape[0], k=1)

        conn_val = np.array(
            [self.connectivity_matrix[i, j] for i, j in zip(*conn_inds)]
        )
        xs = np.array([(centroids[i][0], centroids[j][0]) for i, j in zip(*conn_inds)])
        ys = np.array([(centroids[i][1], centroids[j][1]) for i, j in zip(*conn_inds)])
        zs = np.array([(centroids[i][2], centroids[j][2]) for i, j in zip(*conn_inds)])

        lws = line_scale * (conn_val / (np.nanmax(np.abs(conn_val)))) ** 2
        conn_val_partitioned = np.partition(np.abs(conn_val), -top_n)
        thresh = conn_val_partitioned[-(top_n)]
        if not np.isnan(thresh):
            include = np.abs(conn_val) >= thresh
        else:
            include = np.abs(conn_val) >= np.nanmin(np.abs(conn_val))
        # include = conn_val >= np.percentile(conn_val, 95)

        lc = _multiline(
            xs[include],
            ys[include],
            zs[include],
            conn_val[include],
            cmap=self.cmap,
            lw=lws[include],
            norm=Normalize(vmin=self.colour_range[0], vmax=self.colour_range[1]),
        )

        include_mat = np.zeros_like(self.connectivity_matrix)
        ct = 0
        for i, j in zip(*conn_inds):
            if include[ct]:
                include_mat[i, j] = 1
                include_mat[j, i] = 1
            ct += 1
        include_vert = np.sum(include_mat, axis=1) > 0
        included_centroids = centroids[include_vert, :]

        node_degree = np.sum(include_mat, axis=1)

        node_sizes = np.interp(
            node_degree[include_vert],
            (
                np.nanmin(node_degree[include_vert]),
                np.nanmax(node_degree[include_vert]),
            ),
            (2, 50),
        )
        ax.scatter(
            *[x for x in included_centroids.T], s=node_sizes, marker="o", c="black"
        )

        axcb = fig.colorbar(lc)
        axcb.set_label(self.cbar_title)
        ax.set_title(title)

        ax.set_xlim3d((vertices[:, 0].min(), vertices[:, 0].max()))
        ax.set_ylim3d((vertices[:, 1].min(), vertices[:, 1].max()))
        ax.set_zlim3d((vertices[:, 2].min(), vertices[:, 2].max()))
        ax._axis3don = False
        ax.set_proj_type("ortho")
        ax.set_aspect("equal", adjustable="box")
        views = {
            "side_r": {"elev": 0, "azim": 0},
            "head_on": {"elev": 0, "azim": 90},
            "top_down": {"elev": 90, "azim": -90},
            "side_l": {"elev": 0, "azim": -180},
        }

        ax.view_init(**views[view])
        fig.tight_layout()

        plt.show(block=block)
        return fig


def _multiline(xs, ys, zs, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    zs : iterable container of z coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y, z]) for x, y, z in zip(xs, ys, zs)]
    lc = Line3DCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    return lc


def get_atlas_labels(atlas: str = "aal78"):
    """get list of atlas labels for 78 region AAL or 52 region Glasser atlas

    Args:
        atlas (str, optional): choice of "aal78" or "glasser52". Defaults to "aal78".

    Raises:
        ValueError: if choice is not "aal78" or "glasser52".

    Returns:
        dict: keys: atlas label strings, values: atlas number integers
    """
    atlas_labels_file = os.path.join(os.path.dirname(__file__), "atlas_labels.pkl")
    with open(atlas_labels_file, "rb") as f:
        atlas_labels = pickle.load(f)

    if atlas.lower() == "glasser52" or atlas.lower() == "aal78":
        names = atlas_labels[atlas.lower()]["labels"]
    else:
        raise ValueError("Only aal78 and glasser52 are supported")

    labels = {n: i + 1 for i, n in enumerate(names)}
    return labels


if __name__ == "__main__":
    # test get_atlas_labels

    aal_labels = get_atlas_labels(atlas="aal78")
    glasser_labels = get_atlas_labels(atlas="glasser52")
    # test plotting
    import itertools
    import pandas as pd

    size = 52
    mu = np.zeros(size)
    sigma = np.eye(size) * 0.2
    # Set within network connectivity
    i, j = list(zip(*itertools.combinations(np.arange(0, 10), 2)))
    sigma[i, j] = 0.7
    i, j = list(zip(*itertools.combinations(np.arange(10, 20), 2)))
    sigma[i, j] = 0.7
    i, j = list(zip(*itertools.combinations(np.arange(20, 30), 2)))
    sigma[i, j] = 0.6
    i, j = list(zip(*itertools.combinations(np.arange(30, 53), 2)))
    # sigma[i, j] = 0.1
    # i, j = list(zip(*itertools.combinations(np.arange(53, 78), 2)))
    # sigma[i, j] = 0.2

    # Set between network connectivity
    sigma[20:][:, :20] = 0.2
    sigma[10:20][:, :10] = 0.1
    # sigma[30:70][:, :10] = 0.7

    sigma = sigma + sigma.T
    np.fill_diagonal(sigma, 1)
    # Generate the random data
    np.random.seed(42)
    data = np.random.multivariate_normal(mu, sigma, 1000)
    # Create connectivity matrix
    cm = pd.DataFrame(data).corr().values
    random_order = np.random.permutation(np.arange(size))
    cm = cm[random_order][:, random_order]

    cm = np.eye(size)
    cm[1, 5] = 0.14
    cm[5, 1] = 0.14
    cm[50, 25] = -0.12
    cm[25, 50] = -0.12
    # cm[15,6] = 0.22
    # cm[6,15] = 0.22
    top_n = 2
    conn_plotter = ConnectomePlotter(cm)
    conn_plotter.plot_glassbrain(top_n=top_n)
    conn_plotter.plot_matrix()
    conn_plotter.plot_circle(top_n=top_n)

    degree = np.nansum(np.abs(cm), axis=1)
    from atlas_plotter import AtlasPlotter

    atlas_plotter = AtlasPlotter(degree, cmap_name=conn_plotter.cmap_name)
    atlas_plotter.plot(block=True)

    mean_ad = loadmat("mean_ad.mat")
    cm = mean_ad["mean_ad"]
    top_n = 100
    conn_plotter = ConnectomePlotter(cm)
    conn_plotter.plot_glassbrain(top_n=top_n)
    conn_plotter.plot_matrix()
    conn_plotter.plot_circle(top_n=top_n)

    degree = np.nansum(np.abs(cm), axis=1)
    from atlas_plotter import AtlasPlotter

    atlas_plotter = AtlasPlotter(
        degree, cmap_name=conn_plotter.cmap_name, colour_range=[0, 10]
    )
    atlas_plotter.plot(block=True)

    degree = np.random.permutation(78)
    atlas_plotter = AtlasPlotter(degree, cmap_name="turbo", colour_range=[0, 78])
    atlas_plotter.plot(block=True)

    degree = np.random.permutation(52)
    atlas_plotter = AtlasPlotter(degree, cmap_name="turbo", colour_range=[0, 52])
    atlas_plotter.plot(block=True)
