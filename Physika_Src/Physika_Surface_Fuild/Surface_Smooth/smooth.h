#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_SMOOTH_SMOOTH_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_SMOOTH_SMOOTH_H_

#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/mymesh.h"
#include <vector>
#include <functional>
namespace Physika{
void jacobi_laplace_smooth(MyMesh &mesh, int smooth_times);

void decimater(MyMesh &mesh, double max_err);

MyMesh marching_cubes(MyMesh::Point base, float delta, MyMesh::Point bound, std::function<float(MyMesh::Point const &p)> func);

MyMesh potential_field_mesh(MyMesh &mesh);

void jacobi_laplace_smooth_and_expand(MyMesh &mesh, int smooth_times);
}
#endif
