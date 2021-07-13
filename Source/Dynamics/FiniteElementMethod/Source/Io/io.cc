/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: io utility
 * @version    : 1.0
 */
#include "io.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include "vtk.h"

using namespace std;
using namespace Eigen;
namespace PhysIKA {
/*
int read_fixed_verts(const char *filename, std::vector<size_t> &fixed) {
  fixed.clear();
  ifstream ifs(filename);
  if ( ifs.fail() ) {
    cerr << "[error] can not open " << filename << endl;
    return __LINE__;
  }
  size_t temp;
  while ( ifs >> temp ) {
    fixed.push_back(temp);
  }
  cout << "[info] fixed verts number: " << fixed.size() << endl;
  ifs.close();
  return 0;
}
*/

int write_MAT(const char* path, const Eigen::MatrixXd& A)
{
    ofstream      ofs(path, ofstream::binary);
    const int64_t rows = A.rows(), cols = A.cols();

    cout << "===write SPM matrix===" << endl;
    cout << "rows " << rows << " cols " << cols << endl;
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));

    const Map<const VectorXd> value(A.data(), A.size());
    for (size_t i = 0; i < value.size(); ++i)
        ofs.write(reinterpret_cast<const char*>(&(A(i))), sizeof(A(i)));

    ofs.close();
    return 0;
};
int write_SPM(const char* path, const Eigen::SparseMatrix<double>& A)
{
    ofstream      ofs(path, ofstream::binary);
    const int64_t rows = A.rows(), cols = A.cols(), nnz = A.nonZeros();

    cout << "===write SPM matrix===" << endl;
    cout << "rows " << rows << " cols " << cols << " nnz " << nnz << endl;
    ofs.write(reinterpret_cast<const char*>(&rows), sizeof(rows));
    ofs.write(reinterpret_cast<const char*>(&cols), sizeof(cols));
    ofs.write(reinterpret_cast<const char*>(&nnz), sizeof(nnz));

    const auto& outer = A.outerIndexPtr();
    for (size_t i = 0; i < cols + 1; ++i)
    {
        const int32_t id = static_cast<int32_t>(outer[i]);
        ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }

    const auto& inner = A.innerIndexPtr();
    for (size_t i = 0; i < nnz; ++i)
    {
        const int32_t id = static_cast<int32_t>(inner[i]);
        ofs.write(reinterpret_cast<const char*>(&id), sizeof(id));
    }

    const auto& value = A.valuePtr();
    for (size_t i = 0; i < nnz; ++i)
        ofs.write(reinterpret_cast<const char*>(&value[i]), sizeof(value[i]));

    ofs.close();
    return 0;
}
int read_fixed_verts_from_csv(const char* filename, std::vector<size_t>& fixed, MatrixXd* pos)
{
    ifstream ifs(filename);
    if (ifs.fail())
    {
        cerr << "[WARN] can not open " << filename << endl;
        return __LINE__;
    }

    vector<double> coords;

    string line;
    getline(ifs, line);
    cout << "# csv title: " << line << endl;

    while (std::getline(ifs, line))
    {
        stringstream linestream(line);
        string       cell;

        std::getline(linestream, cell, ',');
        fixed.push_back(std::stoi(cell));

        while (getline(linestream, cell, ','))
        {
            coords.push_back(std::stod(cell));
        }
    }
    assert(coords.size() == 3 * fixed.size());

    if (pos != nullptr)
    {
        pos->resize(3, fixed.size());
        std::copy(coords.begin(), coords.end(), pos->data());
    }

    return 0;
}

// int tet_mesh_write_to_vtk(const char *path, const Eigen::Ref<Eigen::MatrixXd> nods, const Eigen::Ref<Eigen::MatrixXi> tets, const Eigen::MatrixXd *mtr){
//   assert(tets.rows() == 4);

//   std::ofstream ofs(path);
//   if ( ofs.fail() )
//     return __LINE__;

//   ofs << std::setprecision(15);
//   tet2vtk(ofs, nods.data(), nods.cols(), tets.data(), tets.cols());
//   if ( mtr != nullptr ) {
//     for (int i = 0; i < mtr->rows(); ++i) {
//       const std::string mtr_name = "theta_"+std::to_string(i);
//       const Eigen::MatrixXd curr_mtr = mtr->row(i);
//       if ( i == 0 )
//         ofs << "CELL_DATA " << curr_mtr.size() << "\n";
//       vtk_data(ofs, curr_mtr.data(), curr_mtr.cols(), mtr_name.c_str(), mtr_name.c_str());
//     }
//   }
//   ofs.close();
//   return 0;

// }

/*
int hex_mesh_read_from_vtk(const char *path, matd_t *node, mati_t *hex, matd_t *mtr) {
  ifstream ifs(path);
  if(ifs.fail()) {
    cerr << "[info] " << "can not open file" << path << endl;
    return __LINE__;
  }

  matrix<double> node0;
  matrix<int> hex1;

  string str;
  int point_num = 0,cell_num = 0;

  while(!ifs.eof()){
    ifs >> str;
    if(str == "POINTS"){
      ifs >> point_num >> str;
      node0.resize(3, point_num);
      for(size_t i = 0;i < point_num; ++i){
        for(size_t j = 0;j < 3; ++j)
          ifs >> node0(j, i);
      }
      continue;
    }
    if(str == "CELLS"){
      ifs >> cell_num >> str;
      int point_number_of_cell = 0;
      vector<size_t> hex_temp;
      for(size_t ci = 0; ci < cell_num; ++ci){
        ifs >> point_number_of_cell;
        if(point_number_of_cell != 8){
          for(size_t i = 0; i < point_number_of_cell; ++i)
            ifs >> str;
        }else{
          int p;
          for(size_t i = 0; i < point_number_of_cell; ++i){
            ifs >> p;
            hex_temp.push_back(p);
          }
        }
      }
      hex1.resize(8, hex_temp.size()/8);
      copy(hex_temp.begin(), hex_temp.end(), hex1.begin());
      break;
    }
  }

  vector<size_t> one_hex(hex1.size(1));
  for(size_t hi = 0; hi < hex1.size(2); ++hi){
    copy(hex1(colon(),hi).begin(), hex1(colon(), hi).end(), one_hex.begin());
    hex1(0,hi) = one_hex[6];
    hex1(1,hi) = one_hex[5];
    hex1(2,hi) = one_hex[7];
    hex1(3,hi) = one_hex[4];
    hex1(4,hi) = one_hex[2];
    hex1(5,hi) = one_hex[1];
    hex1(6,hi) = one_hex[3];
    hex1(7,hi) = one_hex[0];
  }

  *node = node0;
  *hex = hex1;

  vector<double> tmp;
  double mtrval;
  if ( mtr != nullptr ) {
    while ( !ifs.eof() ) {
      ifs >> str;
      if ( str == "LOOKUP_TABLE" ) {
        ifs >> str;
        for (size_t i = 0; i < hex->size(2); ++i) {
          ifs >> mtrval;
          tmp.push_back(mtrval);
        }
      }
    }

    if ( tmp.size() > 0 ) {
      ASSERT(tmp.size() % hex->size(2) == 0);
      matd_t tmp_mtr(hex->size(2), tmp.size()/hex->size(2));
      std::copy(tmp.begin(), tmp.end(), tmp_mtr.begin());
      *mtr = trans(tmp_mtr);
    }
  }

  ifs.close();

  return 0;
}

int hex_mesh_write_to_vtk(const char *path, const matd_t &nods,
                          const mati_t &hexs, const matd_t *mtr, const char *type) {
  ASSERT(hexs.size(1) == 8);

  ofstream ofs(path);
  if ( ofs.fail() )
    return __LINE__;

  ofs << setprecision(15);
  hex2vtk(ofs, &nods[0], nods.size()/3, &hexs[0], hexs.size(2));

  if ( mtr != nullptr ) {
    for (int i = 0; i < mtr->size(1); ++i) {
      const string mtr_name = "theta_"+to_string(i);
      const matd_t curr_mtr = (*mtr)(i, colon());
      if ( i == 0 )
        ofs << type << "_DATA " << curr_mtr.size() << "\n";
      vtk_data(ofs, curr_mtr.begin(), curr_mtr.size(), mtr_name.c_str(), mtr_name.c_str());
    }
  }
  ofs.close();
  return 0;
}

*/
int tri_mesh_write_to_vtk(const char* path, const MatrixXd& nods, const MatrixXi& tris, const MatrixXd* mtr)
{
    assert(tris.rows() == 3);
    ofstream ofs(path);
    if (ofs.fail())
        return __LINE__;

    ofs << setprecision(15);

    if (nods.rows() == 2)
    {
        MatrixXd tmp_nods = MatrixXd::Zero(3, nods.cols());
        tmp_nods.row(0)   = nods.row(0);
        tmp_nods.row(1)   = nods.row(1);
        tri2vtk(ofs, tmp_nods.data(), tmp_nods.cols(), tris.data(), tris.cols());
    }
    else if (nods.rows() == 3)
    {
        tri2vtk(ofs, nods.data(), nods.cols(), tris.data(), tris.cols());
    }

    if (mtr != nullptr)
    {
        for (int i = 0; i < mtr->rows(); ++i)
        {
            const string   mtr_name = "theta_" + to_string(i);
            const MatrixXd curr_mtr = (*mtr).row(i);
            if (i == 0)
                ofs << "CELL_DATA " << curr_mtr.size() << "\n";
            vtk_data(ofs, curr_mtr.data(), curr_mtr.size(), mtr_name.c_str(), mtr_name.c_str());
        }
    }
    ofs.close();
    return 0;
}

int point_vector_append2vtk(const bool is_append, const char* path, const MatrixXd& vectors, const size_t num_vecs, const char* vector_name)
{
    assert(vectors.rows() == 3);
    ofstream ofs(path, ios_base::app);
    if (ofs.fail())
        return __LINE__;
    point_data_vector(is_append, ofs, vectors.data(), vectors.cols(), vector_name);
    return 0;
}
int point_scalar_append2vtk(const bool is_append, const char* path, const VectorXd& scalars, const size_t num_sca, const char* scalar_name)
{
    ofstream ofs(path, ios_base::app);
    if (ofs.fail())
        return __LINE__;
    point_data_scalar(is_append, ofs, scalars.data(), scalars.rows(), scalar_name);
    return 0;
}
/*
int quad_mesh_write_to_vtk(const char *path, const matd_t &nods,
                           const mati_t &quad, const matd_t *mtr, const char *type) {
  ASSERT(quad.size(1) == 4);

  ofstream ofs(path);
  if ( ofs.fail() )
    return __LINE__;

  ofs << setprecision(15);

  if ( nods.size(1) == 2 ) {
    matd_t tmp_nods = zeros<double>(3, nods.size(2));
    tmp_nods(colon(0, 1), colon()) = nods;
    quad2vtk(ofs, &tmp_nods[0], tmp_nods.size(2), &quad[0], quad.size(2));
  } else if ( nods.size(1) == 3 ) {
    quad2vtk(ofs, &nods[0], nods.size()/3, &quad[0], quad.size(2));
  }

  if ( mtr != nullptr ) {
    for (int i = 0; i < mtr->size(1); ++i) {
      const string mtr_name = "theta_"+to_string(i);
      const matd_t curr_mtr = (*mtr)(i, colon());
      if ( i == 0 )
        ofs << type << "_DATA " << curr_mtr.size() << "\n";
      vtk_data(ofs, curr_mtr.begin(), curr_mtr.size(), mtr_name.c_str(), mtr_name.c_str());
    }
  }
  ofs.close();
  return 0;
}

int tet_mesh_write_to_vtk(const char *path, const matd_t &nods,
                          const mati_t &tets, const matd_t *mtr) {
  ASSERT(tets.size(1) == 4);

  ofstream ofs(path);
  if ( ofs.fail() )
    return __LINE__;
  ofs << setprecision(15);
  tet2vtk(ofs, &nods[0], nods.size()/3, &tets[0], tets.size(2));
  if ( mtr != nullptr ) {
    for (int i = 0; i < mtr->size(1); ++i) {
      const string mtr_name = "theta_"+to_string(i);
      const matd_t curr_mtr = (*mtr)(i, colon());
      if ( i == 0 )
        ofs << "CELL_DATA " << curr_mtr.size() << "\n";
      vtk_data(ofs, curr_mtr.begin(), curr_mtr.size(), mtr_name.c_str(), mtr_name.c_str());
    }
  }
  ofs.close();
  return 0;
}
*/
int point_write_to_vtk(const char* path, const double* nods, const size_t num_points)
{
    ofstream ofs(path);
    if (ofs.fail())
        return __LINE__;
    // const mati_t cell = colon(0, num_points-1);
    VectorXi cell = VectorXi::Zero(num_points);
    {
        for (size_t i = 0; i < num_points; ++i)
        {
            cell(i) = i;
        }
    }

    point2vtk(ofs, nods, num_points, cell.data(), cell.size());
    ofs.close();
    return 0;
}
/*
int point_write_to_vtk(const char *path, const matd_t &nods) {
  ofstream ofs(path);
  if ( ofs.fail() )
    return __LINE__;

  matd_t nods_to_write = zeros<double>(3, nods.size(2));
  if ( nods.size(1) == 2 )
    nods_to_write(colon(0, 1), colon()) = nods;
  else if ( nods.size(1) == 3 )
    nods_to_write = nods;

  const mati_t cell = colon(0, nods.size(2)-1);
  point2vtk(ofs, &nods_to_write[0], nods_to_write.size(2), &cell[0], cell.size());
  ofs.close();

  return 0;
}

int read_json_file(const char *file, Json::Value &json) {
  Json::Reader reader;
  ifstream ifs(file);
  if ( ifs.fail() ) {
    cerr << "[Error] loading json." << endl;
    return __LINE__;
  }
  if ( !reader.parse(ifs, json) ) {
    cerr << "[Error] parsing json." << endl;
    return __LINE__;
  }
  ifs.close();
  return 0;
}

static inline void parse_vec3(const Json::Value &json, double *vec) {
  vec[0] = json[0].asDouble();
  vec[1] = json[1].asDouble();
  vec[2] = json[2].asDouble();
}

static inline void parse_vec2(const Json::Value &json, double *vec) {
  vec[0] = json[0].asDouble();
  vec[1] = json[1].asDouble();
}

int read_geom_objs_from_json(const char *file, vector<shared_ptr<signed_dist_func>> &objs) {
  Json::Value json;
  if ( read_json_file(file, json) )
    return __LINE__;

  int flag = 0;
  objs.resize(json["objects"].size());
  for (uint i = 0; i < json["objects"].size(); ++i) {
    const Json::Value &curr_obj = json["objects"][i];
    const string name = curr_obj["name"].asString();
    if ( name == "plane" ) {
      double center[3], normal[3];
      parse_vec3(curr_obj["center"], center);
      parse_vec3(curr_obj["normal"], normal);
      objs[i] = make_shared<planeSDF>(center, normal);
      continue;
    }
    if ( name == "sphere" ) {
      double center[3];
      parse_vec3(curr_obj["center"], center);
      double radius = curr_obj["radius"].asDouble();
      objs[i] = make_shared<sphereSDF>(center, radius);
      continue;
    }
    if ( name == "cylinder" ) {
      double center[3], normal[3];
      parse_vec3(curr_obj["center"], center);
      parse_vec3(curr_obj["normal"], normal);
      double radius = curr_obj["radius"].asDouble();
      objs[i] = make_shared<cylinderSDF>(center, normal, radius);
      continue;
    }
    if ( name == "torus" ) {
      double center[3], normal[3];
      parse_vec3(curr_obj["center"], center);
      parse_vec3(curr_obj["normal"], normal);
      double innerR = curr_obj["innerR"].asDouble();
      double outerR = curr_obj["outerR"].asDouble();
      objs[i] = make_shared<torusSDF>(center, normal, innerR, outerR);
      continue;
    }
    if ( name == "line" ) {
      double center[2], normal[2];
      parse_vec2(curr_obj["center"], center);
      parse_vec2(curr_obj["normal"], normal);
      objs[i] = make_shared<lineSDF>(center, normal);
      continue;
    }
    if ( name == "circle" ) {
      double center[2];
      parse_vec2(curr_obj["center"], center);
      double radius = curr_obj["radius"].asDouble();
      objs[i] = make_shared<circleSDF>(center, radius);
      continue;
    }
    cerr << "[WARN] invalid geometry type " << i << endl;
    flag = 1;
  }
  return flag;
}

int write_surf_vec_field(const char *path, const mati_t &surf, const matd_t &nods, const matd_t &vf) {
  ASSERT(nods.size(1) == vf.size(1) && surf.size(2) == vf.size(2));

  const size_t node_dim = nods.size(1);
  const size_t elem_num = surf.size(2);

  matd_t bc = zeros<double>(node_dim, elem_num);
  for (size_t i = 0; i < elem_num; ++i)
    bc(colon(), i) = nods(colon(), surf(colon(), i))*ones<double>(surf.size(1), 1)/surf.size(1);

  matd_t vec_nods(node_dim, 2*elem_num);
  vec_nods(colon(), colon(0, elem_num-1)) = bc;
  vec_nods(colon(), colon(elem_num, 2*elem_num-1)) = bc+vf;

  mati_t vec_cell(2, elem_num);
  vec_cell(0, colon()) = colon(0, elem_num-1);
  vec_cell(1, colon()) = colon(elem_num, 2*elem_num-1);

  ofstream ofs(path);
  if ( ofs.fail() ) {
    cerr << "[Error] can not open " << path << endl;
    return __LINE__;
  }
  line2vtk(ofs, &vec_nods[0], vec_nods.size(2), &vec_cell[0], vec_cell.size(2));
  ofs.close();

  return 0;
}
*/
}  // namespace PhysIKA
