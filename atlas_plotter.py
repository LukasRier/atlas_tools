# -*- coding: utf-8 -*-
"""
AtlasPlotter class to create 3D plots of atlas region stats on a cortical surface.
Created on Fri May 10 14:44:12 2024

@author: ppzlr

"""
import os
import pathlib

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from mayavi import mlab
import numpy as np
import numpy.matlib
from PIL import Image
from scipy.io import loadmat


class AtlasPlotter:
    """Atlas plotter class for 78 region AAL and 52 region Glasser atlas
    Needs "surf_info_aal_glasser.mat" in path!
    """

    def __init__(
        self,
        region_stats,
        colour_range=None,
        atlas: str = None,
        cmap_name: str = "turbo",
        cbar_title=None,
    ):
        """Initialize the AtlasPlotter, set optional attributes and generate info.

        Args:
            region_stat (numpy.ndarray): N_regions by 1 array (N_regions = 52 or 78)
            colour_range (arraylike, optional): Two element list of colour limit. Defaults to [min,max] of region_stats.
            atlas (str, optional): atlas name, can be either "glasser52" or "aal78". Defaults to None and gets determined using the size of region_stats.
            cmap_name (str, optional): Matplotlib colormap name. Defaults to "turbo".
            cbar_title (str, optional):  Title for color bar. Gets set to determined atlas name if None. Defaults to None.
        """
        if colour_range is None:
            self.colour_range = [np.nanmin(region_stats), np.nanmax(region_stats)]
            if self.colour_range[0] == self.colour_range[1]:
                self.colour_range[0] = 0

        self.cmap_name = cmap_name

        if atlas is None and region_stats.size == 52:
            self.atlas = "glasser52"
        elif atlas is None and region_stats.size == 78:
            self.atlas = "aal78"
        elif atlas.lower() == "glasser52":
            assert region_stats.size == 52
        elif atlas.lower() == "aal78":
            assert region_stats.size == 78

        self.region_stats = region_stats

        if cbar_title is None:
            self.cbar_title = self.atlas
        else:
            self.cbar_title = cbar_title

        self.vertices = {"L": [], "R": []}
        self.faces = {"L": [], "R": []}
        self.colors = {"L": [], "R": [], "lut": []}
        self.scalars = {"L": [], "R": []}

        self.views = {
            "back": {"elevation": -90, "azimuth": 90},
            "side_right": {"elevation": -90, "azimuth": 180},
            "top_down": {"elevation": 0, "azimuth": 0},
            "side_left": {"elevation": 90, "azimuth": 180},
            "head_on": {"elevation": -90, "azimuth": 270},
            "bottom_up": {"elevation": 180, "azimuth": 0},
        }

        self.get_atlas_info()

    def get_atlas_info(self):
        """Get atlas information needed for surface plots
        Sets:
            self.vertices (dict) : keys "L" and "R" containing [n_points x 3] numpy.ndarray of brain surface vertices for each hemisphere.
            self.faces (dict) : keys "L" and "R" containing [n_faces x 3] numpy.ndarray of brain surface faces for each hemisphere.
            self.colors (dict) : keys "L", "R", "lut" containing [n_points x 3] numpy.ndarray of vertex colours for each hemisphere and the [3x255] numpy.ndarray colourmap lookup table.
            self.scalars (dict) : keys "L" and "R" containing [n_points x 1] numpy.ndarray of brain surface point indices for each hemisphere. Indexes into the vertex-wise lookuptables in colors
            self.views (dict) : contains dicts with keys "azimuth" and "elevation" with angles specifying common viewpoints.
        """

        c_map = plt.get_cmap(self.cmap_name)(np.linspace(0, 1, 256))[:, 0:3]

        atlas_surf_info = loadmat(
            os.path.join(os.path.dirname(__file__), "surf_info_aal_glasser.mat")
        )

        vertices_L = atlas_surf_info["surf_lh"][0][0][1]
        faces_L = atlas_surf_info["surf_lh"][0][0][2] - 1

        vertices_R = atlas_surf_info["surf_rh"][0][0][1]
        faces_R = atlas_surf_info["surf_rh"][0][0][2] - 1

        if self.atlas.lower() == "glasser52":
            vertex_labels_R = atlas_surf_info["surf_rh"][0][0][3]
            vertex_labels_L = atlas_surf_info["surf_lh"][0][0][3]
            labels = atlas_surf_info["glasser_labels"]
        elif self.atlas.lower() == "aal78":
            vertex_labels_R = atlas_surf_info["surf_rh"][0][0][0]
            vertex_labels_L = atlas_surf_info["surf_lh"][0][0][0]
            labels = atlas_surf_info["aal_labels"]
        else:
            raise ValueError('atlas must be "glasser52" or "aal78"')

        n_regions = labels.size
        vertex_colors_r = np.ones((vertices_R.shape[0], 4)) * 0.6
        vertex_colors_r[:, 3] = 1

        vertex_colors_l = np.ones((vertices_L.shape[0], 4)) * 0.6
        vertex_colors_l[:, 3] = 1

        braincolor = np.array([0.6, 0.6, 0.6, 1])
        for j in range(n_regions):
            searchlabel = labels[j]

            colorindex = np.fix(
                (self.region_stats[j] - self.colour_range[0])
                / (self.colour_range[1] - self.colour_range[0])
                * (c_map.shape[0] - 1)
            )
            if not np.isnan(colorindex):
                surfacecolor = c_map[int(colorindex.tolist()), :]
            else:
                surfacecolor = braincolor[0:-1]

            labelind = np.char.startswith(vertex_labels_L, searchlabel.strip())
            if labelind.any():
                vertex_colors_l[labelind, 0:-1] = np.matlib.repmat(
                    surfacecolor, np.sum(labelind), 1
                )

            labelind = np.char.startswith(vertex_labels_R, searchlabel.strip())
            if labelind.any():
                vertex_colors_r[labelind, 0:-1] = np.matlib.repmat(
                    surfacecolor, np.sum(labelind), 1
                )

        n_points_L = vertices_L.shape[0]
        scalars_L = np.arange(n_points_L)  # Set an integer for each pt

        n_points_R = vertices_R.shape[0]
        scalars_R = np.arange(n_points_R)
        # Define color table (including alpha), which must be uint8 and [0,255]
        colors_L = (vertex_colors_l * 255).astype(np.uint8)
        colors_R = (vertex_colors_r * 255).astype(np.uint8)

        # lookup table for actual color map
        lut = np.concatenate((c_map * 255, 255 * np.ones((c_map.shape[0], 1))), axis=1)

        self.vertices = {"L": vertices_L, "R": vertices_R}
        self.faces = {"L": faces_L, "R": faces_R}
        self.colors = {"L": colors_L, "R": colors_R, "lut": lut}
        self.scalars = {"L": scalars_L, "R": scalars_R}
        return (
            self.vertices,
            self.faces,
            self.colors,
            self.scalars,
            self.views,
            self.cbar_title,
            self.colour_range,
        )

    def plot_both(self, figsize=(600, 600)):
        """plot both hemispheres"""
        mlab.figure(bgcolor=(1, 1, 1), size=figsize)
        msh_L = mlab.triangular_mesh(
            self.vertices["L"][:, 0],
            self.vertices["L"][:, 1],
            self.vertices["L"][:, 2],
            self.faces["L"],
            scalars=self.scalars["L"],
        )
        msh_L.module_manager.scalar_lut_manager.lut.table = self.colors["L"]

        msh_R = mlab.triangular_mesh(
            self.vertices["R"][:, 0],
            self.vertices["R"][:, 1],
            self.vertices["R"][:, 2],
            self.faces["R"],
            scalars=self.scalars["R"],
        )
        msh_R.module_manager.scalar_lut_manager.lut.table = self.colors["R"]

        engine = mlab.get_engine()
        scene = engine.scenes[0]
        scene.scene.parallel_projection = True
        poly_data_normals = engine.scenes[0].children[0].children[0]
        poly_data_normals.filter.splitting = False
        poly_data_normals = engine.scenes[0].children[1].children[0]
        poly_data_normals.filter.splitting = False
        mlab.view(**self.views["top_down"], distance=1)

    def plot(self, file: str = None, figsize=(600, 600), explore_mode=True, block=True):
        """Produces 3D plot of a brain surface with coloured in atlas regions.

        If explore_mode is True, an interactive 3D plot will be shown.
        Screenshots of the 3D plot from all angles can be saved by providing a path in 'file'

        Args:
            file (str, optional): path to save figures to. Defaults to None.
                Needs to be of the form /path/to/figurename without file extension.
                if a path is provided, a figure /path/to/figurename.png containing 8
                views of the brain and a colorbar as well as separate files
                /path/to/figurename_<view>.png containing saparate images for each angle
                will be saved.
            figsize (tuple, optional): figure size for plotting. Defaults to (600,600).
            explore_mode (bool, optional): if True opens a mayavi figure with 3D plot to
                allow inspection. Defaults to True.
            block (bool, optional): sets plt.show() 'block' argument

        Returns:
            dict: RGB arrays for each view with keys ["back", "side_right", "top_down",
                                                        "side_left", "head_on", "bottom_up",
                                                        "inside_left", "inside_right"]
        """

        if explore_mode:
            self.plot_both()

            # hides a little image to get the correct colorbar as the meshes are rendered with
            # a vertex-wise lookup table
            image = np.matlib.repmat(
                np.linspace(
                    self.colour_range[0],
                    self.colour_range[1],
                    self.colors["lut"].shape[0],
                ),
                20,
                1,
            )
            im = mlab.imshow(image, extent=(-5, -4, -1, 1, -1, 1))
            im.module_manager.scalar_lut_manager.lut.table = self.colors["lut"]

            # create colorbar and set lablls
            mlab.colorbar(im, title=self.cbar_title, orientation="vertical")
            cb_label_text_property = (
                im.module_manager.scalar_lut_manager.label_text_property
            )
            cb_label_text_property.color = (0.0, 0.0, 0.0)
            cb_label_text_property.bold = False
            cb_label_text_property.italic = False

            cb_title_text_property = (
                im.module_manager.scalar_lut_manager.title_text_property
            )
            cb_title_text_property.italic = False
            cb_title_text_property.color = (0.0, 0.0, 0.0)
            cb_title_text_property.line_offset = -10.0  # shift title up removes overlap
            print("Close figure to continue...")
            mlab.show()

        # arrange multiple views in matplotlib
        mlab.options.offscreen = (
            True  # renders in software without showing the plot for speed
        )
        images = {}
        self.plot_both()

        # cycle through all views and take screenshot for later
        for k, v in self.views.items():
            mlab.view(**self.views[k])
            images[k] = mlab.screenshot(mode="rgb", antialiased=True)

        mlab.close()

        # plots only right hemisphere to get inside view
        mlab.figure(bgcolor=(1, 1, 1), size=figsize)
        msh_R = mlab.triangular_mesh(
            self.vertices["R"][:, 0],
            self.vertices["R"][:, 1],
            self.vertices["R"][:, 2],
            self.faces["R"],
            scalars=self.scalars["R"],
        )
        msh_R.module_manager.scalar_lut_manager.lut.table = self.colors["R"]
        engine = mlab.get_engine()
        scene = engine.scenes[0]
        scene.scene.parallel_projection = True
        poly_data_normals = engine.scenes[0].children[0].children[0]
        poly_data_normals.filter.splitting = False

        mlab.view(**self.views["side_left"])
        images["inside_right"] = mlab.screenshot(mode="rgb", antialiased=True)
        mlab.close()

        # plots only left hemisphere to get inside view
        mlab.figure(bgcolor=(1, 1, 1), size=(400, 600))
        msh_L = mlab.triangular_mesh(
            self.vertices["L"][:, 0],
            self.vertices["L"][:, 1],
            self.vertices["L"][:, 2],
            self.faces["L"],
            scalars=self.scalars["L"],
        )
        msh_L.module_manager.scalar_lut_manager.lut.table = self.colors["L"]

        engine = mlab.get_engine()
        scene = engine.scenes[0]
        scene.scene.parallel_projection = True
        poly_data_normals = engine.scenes[0].children[0].children[0]
        poly_data_normals.filter.splitting = False

        mlab.view(**self.views["side_right"])
        images["inside_left"] = mlab.screenshot(mode="rgb", antialiased=True)
        mlab.close()

        # plot screenshots on matplotlib figure
        fig = plt.figure(figsize=(8, 9), layout="tight")
        ax1 = fig.add_subplot(331)
        ax1.imshow(images["side_left"])
        ax1.set_axis_off()

        ax2 = fig.add_subplot(332)
        ax2.imshow(images["top_down"])
        ax2.set_axis_off()

        ax3 = fig.add_subplot(333)
        ax3.imshow(images["side_right"])
        ax3.set_axis_off()

        ax4 = fig.add_subplot(335)
        ax4.imshow(images["bottom_up"])
        ax4.set_axis_off()

        ax5 = fig.add_subplot(336)
        ax5.imshow(images["inside_left"])
        ax5.set_axis_off()

        ax6 = fig.add_subplot(334)
        ax6.imshow(images["inside_right"])
        ax6.set_axis_off()

        ax7 = fig.add_subplot(337)
        ax7.imshow(images["head_on"])
        ax7.set_axis_off()

        ax8 = fig.add_subplot(338)
        ax8.imshow(images["back"])
        ax8.set_axis_off()

        cbax = fig.add_axes([0.65, 0.2, 0.2, 0.02])
        # cbax = fig.add_subplot(339)
        mappable = ScalarMappable(
            norm=Normalize(vmin=self.colour_range[0], vmax=self.colour_range[1]),
            cmap=self.cmap_name,
        )
        fig.colorbar(
            mappable, orientation="horizontal", cax=cbax, label=self.cbar_title
        )
        if file:
            plt.savefig(file + ".png")
            print(f"saving images to {file}_<view>.png")
            for k, v in images.items():
                im = Image.fromarray(v, mode="RGB")
                im.save(file + "_" + k + ".png", "PNG")
        print("Close figure to continue...")
        plt.draw_all()
        plt.show(block=block)

        return images


if __name__ == "__main__":
    colour_range = [0, 52]
    region_stats = np.arange(1, 53, dtype="float")
    save_path_root = pathlib.Path(".") / "test" / "testplot"
    # colour_range = [0, 78]
    # region_stats = np.arange(1, 79, dtype="float")

    # from numpy.random import permutation

    # region_stats = permutation(52)

    region_stats[:] = np.nan
    reg = 1
    region_stats[reg] = 1
    region_stats[reg + 26] = 1
    cmap_name = "turbo"
    atlas_plt = AtlasPlotter(region_stats)
    atlas_plt.plot()
