#include "../inc/write_to_file.h"
#include <map>
#include <fstream>
#include <string>
#include <sstream>
#include <vector>
#include <iostream>
#include "../inc/comp.inc"
using namespace std;
using namespace Eigen;

typedef Eigen::Matrix<size_t, Dynamic, 1> VectorXst;

int write_to_vtk(const vector<MatrixXd> &table_element,
                 const char * const path,
                 double *ptr_table_color)
{
  ofstream f_out(path);
  if (!f_out)
  {
    cerr << "error:: file open" << endl;
    return -1;
  }
  
  map<Vector3d, size_t, VertComp> map_vert;
  vector<Vector3d> table_vert;
  vector<VectorXst> table_face_id;
  size_t cell_sum = 0;
  for (const auto &itr : table_element)
  {
    size_t num = itr.cols();
    VectorXst face(num);
    for (size_t id_v = 0; id_v < num; ++id_v)
    {
      const auto &v = itr.col(id_v);
      auto is_insert = map_vert.emplace(v, map_vert.size());
      bool is_s = is_insert.second;

#ifdef REMOVE_DUPLICATE_VERT_IN_WRITE
      is_s = true;
#endif

      if (is_s == true)
      {
        table_vert.push_back(v);
        face(id_v) = table_vert.size() - 1;
      } 
      else
        face(id_v) = map_vert.at(v);
    }
    table_face_id.push_back(face);
    cell_sum += (num + 1);
  }
  
  f_out << "# vtk DataFile Version 2.0" << endl;
  f_out << "Unstructured Grid Example" << endl;
  f_out << "ASCII" << endl;
  f_out << "DATASET UNSTRUCTURED_GRID" << endl;
  f_out << "POINTS " << table_vert.size() << " double" << endl;

  for (const auto &v : table_vert)
    f_out << v(0) << " " << v(1) << " " << v(2) << endl;

  f_out << "CELLS " << table_face_id.size() << " " << cell_sum << endl;
  for (const auto &itr : table_face_id)
  {
    f_out << itr.size();
    size_t num = itr.size();
    for (size_t id_v = 0; id_v < num; ++id_v)
      f_out << " " << itr(id_v);
    f_out << endl;
  }

  f_out << "CELL_TYPES " << table_face_id.size() << endl;
  for (const auto &itr : table_face_id)
  {
    size_t num = itr.size();
    switch (num)
    {
    case 2:
      f_out << "3" << endl;
      break;
    case 1:
      f_out << "1" << endl;
      break;
    case 3:
      f_out << "5" << endl;
      break;
    default:
      f_out << "7" << endl;
    }
  }

  f_out << "CELL_DATA " << table_face_id.size() << endl;
  f_out << "SCALARS edge double 1" << endl;
  f_out << "LOOKUP_TABLE my_table" << endl;

  size_t num_face = table_face_id.size();
  for (size_t itr = 0; itr < num_face; ++itr)
  {
    if (ptr_table_color != nullptr)
      f_out << ptr_table_color[itr] << endl;
    else
      f_out << fmod(itr * 0.2, 1.2) << endl;
  }

  f_out << "LOOKUP_TABLE my_table 6" << endl;
  f_out << "0.98    1      0.94     1" << endl; // white
  // f_out << "0.118 0.565 1 1" << endl;
  f_out << "0.890 0.090 0.051 1" << endl; // red
  // f_out << "1.0 0.5 0 1" << endl;
  // f_out << "1.0 0.92156 0.8039 1" << endl;
  f_out << "0.4 1 0.3 1" << endl; // blue
  f_out << "0.2666667 0.8 0 1" << endl; // blue
  f_out << "0 0.302 0.902 1" << endl; // blue
  f_out << "0.902 0.75 0 1" << endl; // blue

  
  f_out.close();
  return 0;
}

