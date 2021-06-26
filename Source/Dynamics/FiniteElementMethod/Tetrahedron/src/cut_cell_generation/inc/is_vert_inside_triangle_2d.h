#include "predicates.h"

#include <cstddef>
#include <vector>
#include <Eigen/Core>


int is_vert_inside_triangle_2d(const double *const a, const double *const b,
                               const double *const c, const double *const p);

int is_vert_inside_triangle_2d(const double *const a, const double *const b,
                               const double *const c, const double *const p,
                               size_t &v1_e, size_t &v2_e);

//2: on triangle vert
int is_vert_above_triangle(const double *const a, const double *const b,
                           const double *const c, const double *const p);

int is_triangle_area_positive(
  size_t axis, const std::vector<Eigen::Vector3d> &tri_v);


int is_triangle_area_positive(double const *pa, double const *pb, double const *pc);
