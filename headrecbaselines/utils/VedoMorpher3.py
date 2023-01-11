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
            'no_angle' : 30,  # Angle threshold for normals
            'n_points' : 100,  # Number of points to use for the closest point search
        }

        self.pm.update(params or {})

        self.source = source
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

    def compute_new_points(self):
        self.new_points = []
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
                    print(f"No near point found point {i}")
                    self.new_points.append(orig_point)
                else:
                    self.new_points.append(closest_point)
                
            
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
            ang = vg.angle(self.target.normals()[p], n)  # Angle between normals
            if ang < angle:
                return self.target.points()[p]
        return None
        

    def morph(self):
            self.compute_new_points()
            self.morphed = vedo.Mesh((np.array(self.new_points),
                                 self.source.faces()))
            self.write()          
    
    def write(self):
            save_path = self.pm['save_path']
            self.morphed.write(save_path) if save_path else None
            print(f"Morphed mesh saved to {save_path}")
          


def fit_mesh_dmap(src_mesh, tgt_mesh, distmap, params=None):
    pm = {
        'fit_only_common': True,  # only fit common points based on distance map
        'save_path': None,  # path to save the morphed mesh
        'dm_threshold': 1.0,  # distance map threshold
    }

    pm.update(params or {})
    mr = Morpher(src_mesh, tgt_mesh, distmap, pm)
    mr.morph()
    return mr
    