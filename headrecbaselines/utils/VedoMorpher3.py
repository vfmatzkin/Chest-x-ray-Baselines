""" Morpher class

This class is used to morph a mesh from one shape to another. It is based on the
Vedo library example https://github.com/marcomusy/vedo/blob/master/examples/advanced/warp5.py
"""

import numpy as np
import vedo
import vg


class Morpher:
    def __init__(self, source, target, distmap=None, params=None):

        self.pm = {  # Default values (they could be declared elsewhere)
            'fit_only_common': True,
            'dm_threshold': 1.0,  # distance map threshold
            'no_angle': 30,  # Angle threshold for normals
            'n_points': 100,  # Number of points to use for the closest point search
            'icp': True,  # Use ICP before to fit the meshes
            'closest_point_fit': True,  # Use closest point fit (using normals)
        }

        self.pm.update(params or {})

        self.source = source
        self.old_source = None
        self.target = target
        self.distmap = distmap
        self.morphed = None
        self.new_points = None

        self.source.compute_normals()
        self.target.compute_normals()

    def far_from_skull(self, coords):
        if not self.distmap:
            print("\rDistance map not provided", end='')
            return False
        t_coords = tuple([float(p) for p in coords])  # To tuple and cast
        point = self.distmap.TransformPhysicalPointToIndex(t_coords)
        if self.distmap[point] > self.pm['dm_threshold']:
            return True

    def closest_point_fit(self):
        self.new_points, nnp = [], 0
        print('\n')
        for i in range(self.source.npoints):
            orig_point = self.source.points()[i]

            if self.far_from_skull(orig_point):
                self.new_points.append(orig_point)
            else:
                # Get the closest points in the target mesh
                cls_pts = self.target.closest_point(orig_point, n=self.pm['n_points'],
                                                    return_point_id=True)

                closest_point = self.closest_sim_norm(i, cls_pts)
                if closest_point is None:
                    print(f"\rPoints without near: {(nnp:=nnp+1)}/"
                          f"{self.source.npoints}. i={i}", end='')
                    self.new_points.append(orig_point)
                else:
                    self.new_points.append(closest_point)
        print('\n')

        return vedo.Mesh((np.array(self.new_points), self.source.faces()))

    def closest_sim_norm(self, i, cls_pts_tgt):
        """
        Given a point and a list of closest points, select the one with the
        most similar normal.

        Parameters
        ----------
        i : int
            Index of the point to move in the source mesh.
        cls_pts_tgt : list
            List of the indexes of the closest points in the target mesh.
        """
        angle = self.pm['no_angle']
        n = self.source.normals()[i]  # Normal of the point
        for p in cls_pts_tgt:
            # Angle between normals
            ang = vg.angle(self.target.normals()[p], n)
            if ang < angle:
                return self.target.points()[p]
        return None

    def write(self):
        save_path = self.pm['save_path']
        if save_path:
            self.morphed.write(save_path)
            print(f"Morphed mesh saved to {save_path}")

    def morph(self):
        icp, cp_fit = self.pm['icp'], self.pm['closest_point_fit']
        if not icp and not cp_fit:
            print("No fitting method selected")
            return None

        self.icp_align() if icp else None  # Adjust source points
        self.morphed = self.source if not cp_fit else self.closest_point_fit()

        self.write()
        
    def icp_align(self):
        n_points = {}  # New points
        s_points = {}  # Static points
        self.old_source = self.source.clone()
        for i, point in enumerate(self.source.points()):
            if self.far_from_skull(point):  # Far points remain static
                s_points[i] = point
            else:
                n_points[i] = point
        self.source = vedo.Mesh(np.array(list(n_points.values())))
        self.source.align_to(self.target, invert=True)
        newpts = self.source.points()
        
        # At this point, I have separatedly the points that were morphed
        # and the ones that were not. I need to merge them back together
        # in the same order as the original mesh (so I don't mess up the
        # faces information).
        nw_pts = dict(zip(n_points.keys(), newpts))  # Associate old indices
        mrph_pts = s_points.copy()  # Static points
        mrph_pts.update(nw_pts)  # Add moved points with indexes
        mrph_pts = dict(sorted(mrph_pts.items()))  # Sort by index
        self.source = vedo.Mesh((np.array(list(mrph_pts.values())),
                                 self.old_source.faces()))
        

def fit_mesh_dmap(src_mesh, tgt_mesh, distmap, params=None,
                  return_morpher=False):
    pm = {
        'fit_only_common': True,  # only fit common points based on distance map
        'save_path': None,  # path to save the morphed mesh
        'dm_threshold': 1.0,  # distance map threshold
        'icp': True,  # Use ICP before to fit the meshes
        'closest_point_fit': True,  # Use closest point fit (using normals)
    }

    pm.update(params or {})
    mr = Morpher(src_mesh, tgt_mesh, distmap, pm)
    mr.morph()
    if return_morpher:
        return mr
    else:
        return mr.target
