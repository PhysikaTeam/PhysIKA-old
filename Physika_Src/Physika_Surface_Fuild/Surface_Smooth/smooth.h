#ifndef SMOOTH_H
#define SMOOTH_H

#include "mymesh.h"
#include <vector>
#include <functional>

void jacobi_laplace_smooth(MyMesh &mesh, int smooth_times);

void decimater(MyMesh &mesh, double max_err);

MyMesh marching_cubes(MyMesh::Point base, float delta, MyMesh::Point bound, std::function<float(MyMesh::Point const &p)> func);

MyMesh potential_field_mesh(MyMesh &mesh);

void jacobi_laplace_smooth_and_expand(MyMesh &mesh, int smooth_times);

#endif
