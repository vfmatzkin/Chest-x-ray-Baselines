""" Morpher class

This class is used to morph a mesh from one shape to another. It is based on the
Vedo library example https://github.com/marcomusy/vedo/blob/master/examples/advanced/warp5.py
"""

import scipy.optimize as opt
from vedo import *

settings.use_depth_peeling = True

plt = Plotter(shape=[1, 3], interactive=0, axes=1)


class Morpher:
    def __init__(self, source, target, distmap=None, dm_threshold=1.0):
        self.source = source
        self.target = target
        self.bound = 0.1
        self.method = "SLSQP"  # 'SLSQP', 'L-BFGS-B', 'TNC' ...
        self.tolerance = 0.0001
        self.allowScaling = False
        self.params = []

        self.msource = None
        self.s_size = ([0, 0, 0], 1)  # ave position and ave size
        self.fitResult = None
        self.chi2 = 1.0e10

        self.distmap = distmap
        self.dm_threshold = dm_threshold

    def transform(self, p):
        a1, a2, a3, a4, a5, a6, b1, b2, b3, b4, b5, b6, c1, c2, c3, c4, c5, \
            c6, s = self.params

        pos, sz = self.s_size[0], self.s_size[1]
        x, y, z = (p - pos) / sz * s  # bring to origin, norm and scale
        xx, yy, zz, xy, yz, xz = x * x, y * y, z * z, x * y, y * z, x * z
        xp = x + 2 * a1 * xy + a4 * xx + 2 * a2 * yz + a5 * yy \
             + 2 * a3 * xz + a6 * zz
        yp = +2 * b1 * xy + b4 * xx + y + 2 * b2 * yz + b5 * yy \
             + 2 * b3 * xz + b6 * zz
        zp = +2 * c1 * xy + c4 * xx + 2 * c2 * yz + c5 * yy + z \
             + 2 * c3 * xz + c6 * zz
        p2 = vector(xp, yp, zp)
        p2 = (p2 * sz) + pos  # take back to original size and position
        return p2

    def far_from_skull(self, coords):
        if not self.distmap:
            print("\rDistance map not provided", end='')
            return False
        t_coords = tuple([float(p) for p in coords])  # To tuple and cast
        point = self.distmap.TransformPhysicalPointToIndex(t_coords)
        if self.distmap[point] > self.dm_threshold:
            return True

    def _func(self, pars):
        self.params = pars

        # calculate chi2
        d2sum, n = 0.0, self.source.npoints
        srcpts = self.source.points()
        rng = range(0, n)
        for i in rng:
            p1 = srcpts[i]

            if self.far_from_skull(p1):
                continue  # Exclude points

            p2 = self.transform(p1)
            tp = self.target.closest_point(p2)
            d2sum += mag2(p2 - tp)
        d2sum /= len(rng)

        if d2sum < self.chi2:
            if d2sum < self.chi2 * 0.99:
                print("Emin ->", d2sum)
            self.chi2 = d2sum
        return d2sum

    # ------------------------------------------------------- Fit
    def morph(self):
        def avesize(pts):  # helper fnc
            s, amean = 0, vector(0, 0, 0)
            for p in pts:
                amean = amean + p
            amean /= len(pts)
            for p in pts:
                s += mag(p - amean)
            return amean, s / len(pts)

        print("\n..minimizing with " + self.method)
        self.msource = self.source.clone()

        self.s_size = avesize(self.source.points())
        bnds = [(-self.bound, self.bound)] * 18
        x0 = [0.0] * 18  # initial guess
        x0 += [1.0]  # the optional scale
        if self.allowScaling:
            bnds += [(1.0 - self.bound, 1.0 + self.bound)]
        else:
            bnds += [(1.0, 1.0)]  # fix scale to 1
        res = opt.minimize(self._func, x0,
                           bounds=bnds, method=self.method, tol=self.tolerance)
        self._func(res["x"])
        print("\nFinal fit score", res["fun"])
        self.fitResult = res

        newpts = []
        for p in self.msource.points():
            newp = self.transform(p)
            newpts.append(newp)
        self.msource.points(newpts)

    def draw_shapes(self):

        pos, sz = self.s_size[0], self.s_size[1]

        sphere0 = Sphere(pos, c="gray", r=sz, alpha=0.8, res=16).wireframe()
        sphere1 = sphere0.clone().alpha(0.2).wireframe(False)

        arrs = []
        newpts = []
        for p in sphere0.points():
            newp = self.transform(p)
            newpts.append(newp)
            arrs.append([p, newp])
        sphere1.points(newpts)
        hair = Arrows(arrs, s=0.3, alpha=0.5, c='jet')

        zero = Point(pos, c="black")
        x1, x2, y1, y2, z1, z2 = self.target.bounds()
        tpos = [x1, y2, z1]
        text1 = Text3D("source vs target", tpos, s=sz / 10, c="dg")
        text2 = Text3D("morphed vs target", tpos, s=sz / 10, c="dg")
        text3 = Text3D("deformation", tpos, s=sz / 10, c="dr")

        plt.at(2).show(sphere0, sphere1, zero, text3, hair)
        plt.at(1).show(self.msource, self.target, text2)
        plt.at(0).show(self.source, self.target, text1, zoom=1.2,
                       interactive=True)
        plt.close()


def fit_mesh_dmap(src_mesh, tgt_mesh, distmap, save_path=None,
                  dm_threshold=1, draw_shapes=False):
    """ Fit a source mesh to a target mesh using a distance map.

    :param src_mesh: source mesh
    :param tgt_mesh: target mesh
    :param distmap: distance map
    :param save_path: path to save the morphed mesh
    :param dm_threshold: distance map threshold
    :param draw_shapes: display shapes
    :return:
    """
    mr = Morpher(src_mesh, tgt_mesh, distmap, dm_threshold)
    mr.target.wireframe()
    mr.allowScaling = True
    mr.bound = 0.4  # limits the parameter value

    mr.morph()

    print("Result of parameter fit:\n", mr.params)

    if save_path:
        mr.msource.write(save_path)

    if draw_shapes:
        mr.draw_shapes()
