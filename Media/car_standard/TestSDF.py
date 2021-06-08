

# https://pypi.org/project/mesh-to-sdf/


import mesh_to_sdf
from mesh_to_sdf import sample_sdf_near_surface

import trimesh
import pyrender
import numpy as np

mesh = trimesh.load('D:\\Projects\\physiKA\\physiKA\\Media\\car_standard\\wheel2.obj')

# Signed distance field information.

## wheel3
# nsize = [45, 80, 60]
# pleft = [0.4, 1.3, -1.25]

## wheel1
# nsize = [25, 80, 60]
# pleft = [0.8, -3.3, -1.25]

## wheel2
# nsize = [25, 80, 60]
# pleft = [-1.3, -3.3, -1.25]

nsize = [25, 80, 60]
pleft = [-1.3, -3.3, -1.25]
lh = 0.02
pright = [pleft[0] + lh * nsize[0], pleft[1] + lh * nsize[1], pleft[2] + lh * nsize[2]]

# Compute signed distance field.
points = np.mgrid[pleft[0]:pright[0]:lh, pleft[1]:pright[1]:lh, pleft[2]:pright[2]:lh]
points = np.transpose(points)
vshape = points.shape
print(vshape)
points = np.reshape(points, (vshape[0]*vshape[1]*vshape[2], 3))
print(points.shape)
print(points[5:50, :])
sdf = mesh_to_sdf.mesh_to_sdf(mesh, points, \
    surface_point_method='scan', sign_method='normal', \
        bounding_radius=None, scan_count=100, scan_resolution=400, sample_point_count=10000000, normal_sample_count=11)

print(sdf)
print(sdf.shape)

# Write file.
outfile = "D:\\Projects\\physiKA\\physiKA\\Media\\car_standard\\wheel2.sdf"
with open(outfile,'w') as f:
    f.write(str(nsize[0]) + ' ' + str(nsize[1]) + ' ' + str(nsize[2])+'\n')
    f.write(str(pleft[0]) + ' ' + str(pleft[1]) + ' ' + str(pleft[2])+'\n')
    f.write(str(lh)+'\n')
    for dis in sdf:
        f.write(str(dis)+'\n')
    f.close()
