#include "../inc/manifold_checking_interface.h"

#include <iostream>
#include "../inc/watertight_check.h"
#include "../inc/manifold_check.hpp"
#include "../inc/half_edge.h"
#include <fstream>

using namespace std;

array<bool, 3> manifold_checking(const char* obj_path) {
  exactinit();

  array<bool, 3> model_stat = {true, true, true};

  bool is_watertight = is_mesh_watertight(obj_path);
  if (!is_watertight)
  {
    cerr << "[ERR]: manifold_checking: not watertight model" << endl;
    model_stat[0] = false;
    model_stat[1] = false;
    model_stat[2] = false;
    return model_stat;
  }

  bool is_vert_manifold = vert_manifold(obj_path);
  if (!is_vert_manifold)
  {
    cerr << "[ERR]: manifold_checking: vert nonmanifold" << endl;
    model_stat[0] = false;
  }

  bool is_intersect = is_mesh_self_intersection<double>(obj_path);
  if (is_intersect)
  {
    model_stat[1] = false;
    cerr << "[ERR]: manifold_checking: self-intersection occur" << endl;
  }

  bool is_volume_negative = is_mesh_volume_negative(obj_path);
  if (is_volume_negative) {
    model_stat[2] = false;
    cerr << "[ERR]: manifold_checking: volume is negative" << endl;
  }

  return model_stat;
}
