#include "../inc/mesh.h"
#include <iostream>
#include <fstream>
#include <string>
#include <cstdio>
#include <map>
#include <set>
#include <sstream>
#include <unordered_set>
#include <numeric>
#include <algorithm>
#include <Eigen/Geometry>
#include "../inc/vert.h"
#include "../inc/write_to_file.h"
#include "predicates.h"

using namespace std;
using namespace Eigen;

struct IdEqualVert
{
    bool operator()(const Vert* ptr_v1, const Vert* ptr_v2) const
    {
        size_t id_v1 = ptr_v1->get_vert_id();
        size_t id_v2 = ptr_v2->get_vert_id();
        return ptr_v1->mesh_->is_same_vert(id_v1, id_v2);
    }
};

struct HashFuncVert
{
    size_t operator()(const Vert* ptr_v1) const
    {
        if (ptr_v1->is_grid_vert())
        {
            size_t g[3];
            for (size_t i = 0; i < 3; ++i)
            {
                g[i] = ptr_v1->get_grid_line(i);
                assert(g[i] != numeric_limits<size_t>::max());
            }

            return ((std::hash<size_t>()(g[0])
                     ^ (std::hash<size_t>()(g[1]) << 1))
                    >> 1)
                   ^ (std::hash<size_t>()(g[2]) << 1);
        }
        else if (ptr_v1->is_triangle_vert())
        {
            return std::hash<size_t>()(ptr_v1->get_vert_id());
        }
        else
        {
            size_t a1;
            size_t a2;
            size_t g1;
            size_t g2;
            size_t id_tri;
            ptr_v1->get_line_vert_info(a1, a2, g1, g2, id_tri);
            if (id_tri != numeric_limits<size_t>::max())
            {
                return ((std::hash<size_t>()(id_tri)
                         ^ (std::hash<size_t>()(g1) << 1))
                        >> 1)
                       ^ (std::hash<size_t>()(g2) << 1);
            }
            else
            {
                size_t id_v1;
                size_t id_v2;
                size_t axis;
                size_t id_grid_line;
                ptr_v1->get_edge_vert_info(id_v1, id_v2, axis, id_grid_line);
                return ((std::hash<size_t>()(id_v1)
                         ^ (std::hash<size_t>()(id_v2) << 1))
                        >> 1)
                       ^ (std::hash<size_t>()(id_grid_line) << 1);
            }
        }
    }
};

Mesh::Mesh()
{
    exactinit();
}

Mesh::Mesh(const char* const model_dir)
{
    exactinit();
    read_from_file(model_dir);
}

Mesh::~Mesh()
{
    for (auto& itr : table_vert_ptr_)
        delete itr;
}

int Mesh::read_from_file(const char* const model_dir)
{
    table_vert_ptr_.clear();
    table_tri_.clear();
    string   str_model_dir(model_dir);
    ifstream f_in_model(model_dir);
    if (str_model_dir.substr(str_model_dir.rfind(".") + 1) != "obj"
        || !f_in_model)
    {
        cerr << "only obj file supported" << endl;
        cerr << "error in file open!" << model_dir << endl;
        exit(1);
    }

    size_t                          id_p    = 0;
    bool                            use_mtl = false;
    vector<PatchType>               pt;
    vector<size_t>                  pi;
    map<Vector3d, size_t, VertComp> map_vert;
    map<size_t, size_t>             map_old_to_new_id;

    string str_each_line_of_model;
    while (getline(f_in_model, str_each_line_of_model))
    {

        if (str_each_line_of_model.size() <= 3)
            continue;
        else if (str_each_line_of_model[0] == 'v'
                 && str_each_line_of_model[1] == ' ')
        {
            Vector3d v(3, 1);
            sscanf(str_each_line_of_model.c_str(),
                   "%*s %lf %lf %lf",
                   &v(0),
                   &v(1),
                   &v(2));
            if (!map_vert.count(v))
            {
                Vert*  ptr_vert = new Vert(v);
                size_t id_vert  = add_vert(ptr_vert);
                ptr_vert->set_id(id_vert);
                map_vert.emplace(v, map_vert.size());
                map_old_to_new_id.emplace(
                    map_old_to_new_id.size() + 1, map_vert.size() - 1);
            }
            else
            {
                map_old_to_new_id.emplace(
                    map_old_to_new_id.size() + 1, map_vert.at(v));
            }
            continue;
        }
        else if (str_each_line_of_model[0] == 'f'
                 && str_each_line_of_model[1] == ' ')
        {
            Vector3st id_face_ver;
            if (str_each_line_of_model.find("/") == string::npos)
                sscanf(str_each_line_of_model.c_str(),
                       "%*s %zu %zu %zu",
                       &id_face_ver(0),
                       &id_face_ver(1),
                       &id_face_ver(2));
            else
                sscanf(str_each_line_of_model.c_str(),
                       "%*s %zu%*s %zu%*s %zu%*s",
                       &id_face_ver(0),
                       &id_face_ver(1),
                       &id_face_ver(2));

            for (size_t axis = 0; axis < 3; ++axis)
            {
                assert(map_old_to_new_id.count(id_face_ver(axis)));
                id_face_ver(axis) = map_old_to_new_id.at(id_face_ver(axis));
            }

            table_tri_.push_back(id_face_ver);
            continue;
        }
    }

    assert(pt.size() == pi.size());
    if (use_mtl == true)
    {
        size_t num_tri = table_tri_.size();
        size_t num_p   = num_tri - table_tri_type_.size();
        table_tri_type_.insert(table_tri_type_.end(), num_p, pt.back());
        table_tri_group_id_.insert(table_tri_group_id_.end(), num_p, pi.back());
    }
    if (table_tri_type_.empty())
    {
        table_tri_type_     = vector<PatchType>(table_tri_.size(), wall);
        table_tri_group_id_ = vector<size_t>(table_tri_.size(), 0);
    }

    size_t num_tri = table_tri_.size();
    table_triangle_.resize(num_tri);
    for (size_t itr = 0; itr < num_tri; ++itr)
    {
        table_triangle_[itr].set_id(itr);
    }

    f_in_model.close();
    return 0;
}

MatrixXd Mesh::get_aabb(const std::vector<size_t>& table_id) const
{
    MatrixXd aabb(3, 2);
    for (size_t axis = 0; axis < 3; ++axis)
    {
        aabb(axis, 0) = numeric_limits<double>::max();
        aabb(axis, 1) = -numeric_limits<double>::max();
    }

    for_each(begin(table_id), end(table_id), [&aabb, this](auto itr) {
        const Vector3d& v = get_vert(itr).get_vert_coord();
        for (size_t axis = 0; axis < 3; ++axis)
        {
            aabb(axis, 0) = min(aabb(axis, 0), v(axis));
            aabb(axis, 1) = max(aabb(axis, 1), v(axis));
        }
    });

    return aabb;
}

size_t Mesh::get_tri_num() const
{
    return table_tri_.size();
}

size_t Mesh::get_vert_num() const
{
    return table_vert_ptr_.size();
}

int Mesh::set_cut_num(size_t num_span)
{
    num_span_ = num_span;
    return 0;
}

int Mesh::set_cut_line()
{
    vector<size_t> table_id(table_vert_ptr_.size());
    iota(table_id.begin(), table_id.end(), 0);
    aabb_ = get_aabb(table_id);

    grid_line_    = MatrixXd(3, num_span_ - 1);
    Vector3d span = (aabb_.col(1) - aabb_.col(0)) / num_span_;
    for (size_t itr = 1; itr < num_span_; ++itr)
    {
        grid_line_.col(itr - 1) = aabb_.col(0) + itr * 1.0 * span;
    }

    return 0;
}

const vector<Vector2d> Mesh::get_projection_tri(
    size_t id_tri,
    size_t axis) const
{
    vector<Vector3d> v_tri = get_tri(id_tri);
    vector<Vector2d> v_tri_2d(3);
    for (size_t v = 0; v < 3; ++v)
    {
        for (size_t p = 0; p < 2; ++p)
        {
            size_t axis_p  = (axis + p + 1) % 3;
            v_tri_2d[v](p) = v_tri[v](axis_p);
        }
    }

    return v_tri_2d;
}

const vector<Vector3d> Mesh::get_tri(size_t id_tri) const
{
    vector<Vector3d> table_tri_v(3);
    const Vector3st& tri = table_tri_.at(id_tri);
    for (size_t itr = 0; itr < 3; ++itr)
    {
        const Vector3d v = get_vert(tri(itr)).get_vert_coord();
        table_tri_v[itr] = v;
    }

    return table_tri_v;
}

double Mesh::get_grid_line(size_t axis, size_t id_grid) const
{
    return grid_line_(axis, id_grid);
}

const Vert& Mesh::get_vert(size_t id_v) const
{
    return *(table_vert_ptr_.at(id_v));
}

int Mesh::write_triangle_to_file(size_t id_tri, const char* const path) const
{
    vector<Vector3d> tri_v = get_tri(id_tri);
    Matrix3d         tri;
    for (size_t itr = 0; itr < 3; ++itr)
    {
        tri.col(itr) = tri_v[itr];
    }

    write_to_vtk(vector<MatrixXd>(1, tri), path);

    return 0;
}

const Vert* Mesh::get_vert_ptr(size_t id_v) const
{
    return table_vert_ptr_.at(id_v);
}

Vert* Mesh::get_vert_ptr_to_add_info(size_t id_v)
{
    return table_vert_ptr_.at(id_v);
}

const Vector3st& Mesh::get_tri_vert_id(size_t id_tri) const
{
    return table_tri_.at(id_tri);
}

const MatrixXd& Mesh::get_grid_line() const
{
    return grid_line_;
}

const MatrixXd& Mesh::get_aabb() const
{
    return aabb_;
}

Vector3d Mesh::get_grid_vert_coordinate(size_t a1, size_t a2, size_t a3) const
{
    Vector3d p;
    p(0) = grid_line_(0, a1);
    p(1) = grid_line_(1, a2);
    p(2) = grid_line_(2, a3);

    return p;
}

bool Mesh::is_same_vert(size_t v1, size_t v2)
{
    if (v1 == v2)
        return true;

    const Vert* ptr = get_vert_ptr(v1);

    if (ptr->is_same_vert_with(v2))
        return true;
    else
        return false;
}

int Mesh::write_cell_to_file(const char* const path)
{
    ofstream f_out(path);
    if (!f_out)
    {
        cerr << "error:: file open" << endl;
        return -1;
    }

    map<size_t, size_t>                                           old_to_new_v_id;
    unordered_map<const Vert*, size_t, HashFuncVert, IdEqualVert> map_vert;
    vector<Vert*>                                                 table_new_v;
    size_t                                                        id_v = 0;
    for (const auto& ptr_v : table_vert_ptr_)
    {
        if (!map_vert.count(ptr_v))
        {
            table_new_v.push_back(ptr_v);
            map_vert.emplace(ptr_v, table_new_v.size() - 1);
            old_to_new_v_id.emplace(id_v, table_new_v.size() - 1);
            ++id_v;
        }
        else
        {
            old_to_new_v_id.emplace(id_v, map_vert.at(ptr_v));
            ++id_v;
        }
    }

    f_out << "# vtk DataFile Version 2.0" << endl;
    f_out << "Unstructured Grid Example" << endl;
    f_out << "ASCII" << endl;
    f_out << "DATASET UNSTRUCTURED_GRID" << endl;
    f_out << "POINTS " << table_new_v.size() << " double" << endl;
    for (const auto& ptr_v : table_new_v)
    {
        Vector3d coord_v = ptr_v->get_vert_coord();
        f_out << coord_v(0) << " " << coord_v(1) << " " << coord_v(2) << endl;
    }

    f_out << "CELLS " << polygon_cell_.size() << " ";
    size_t pos = f_out.tellp();
    f_out << "                            " << endl;
    size_t                   total_v   = 0;
    size_t                   max_patch = 0;
    Vector3d                 corner    = aabb_.col(0) - Vector3d::Constant(1);
    vector<array<double, 4>> color;

    unordered_map<Vector3st, double, HashFucLatticeId, IdEqualLattice> color_id_;

    for (size_t i = 0; i < 2; ++i)
        for (size_t j = 0; j < 2; ++j)
            for (size_t k = 0; k < 2; ++k)
            {
                array<double, 4> color_one;
                Vector3st        color_idx;
                color_idx << i, j, k;
                for (size_t c = 0; c < 3; ++c)
                {
                    if (color_idx[c] == 0)
                        color_one[c] = 104 / 255.0;
                    else
                        color_one[c] = 204 / 255.0;
                }
                color_one[3] = 1.0;
                color_id_.emplace(color_idx, 0.0 + color_id_.size() / 7.0);
                color.push_back(color_one);
            }

    for (const auto& cell : polygon_cell_)
    {
        double area = 0;
        for (const auto& p : cell.second)
        {
            assert(p.size() >= 3);
            size_t num_v = p.size();
            for (size_t i = 1; i < num_v - 1; ++i)
            {
                vector<size_t>   tri_v = { p[0], p[i], p[i + 1] };
                vector<Vector3d> tri_v_coord;
                for (auto v : tri_v)
                {
                    tri_v_coord.push_back(get_vert_ptr(v)->get_vert_coord());
                }
                Vector3d ab = tri_v_coord[1] - tri_v_coord[0];
                Vector3d ac = tri_v_coord[2] - tri_v_coord[0];
                Vector3d ad = corner - tri_v_coord[0];
                area -= ab.cross(ac).dot(ad) / 6.0;
            }
        }
        assert(area > -1e-12);

        const auto& inner_p = cell.second;
        size_t      line_v  = 1;
        for_each(begin(inner_p), end(inner_p), [&line_v](const auto& p) {
            line_v += (p.size() + 1);
        });
        f_out << line_v << " " << inner_p.size();
        if (patch_cell_.count(cell.first))
        {
            if (patch_cell_.at(cell.first).size() > max_patch)
                max_patch = patch_cell_.at(cell.first).size();
        }

        for (const auto& p : inner_p)
        {
            f_out << " " << p.size();
            for (auto v : p)
                f_out << " " << old_to_new_v_id.at(v);
        }
        f_out << endl;
        total_v += line_v + 1;
    }

    f_out << "CELL_TYPES " << polygon_cell_.size() << endl;
    size_t num_cell = polygon_cell_.size();
    for (size_t i = 0; i < num_cell; ++i)
        f_out << "42" << endl;

    f_out << "FIELD PatchType 1" << endl;
    f_out << "type " << max_patch * 2 << " " << polygon_cell_.size() << " int" << endl;
    for (const auto& cell : polygon_cell_)
    {
        if (patch_cell_.count(cell.first))
        {
            const auto& table_patch = patch_cell_.at(cell.first);
            assert(table_patch.size() <= max_patch);
            for (const auto& p : table_patch)
                f_out << p.first << " " << p.second << " ";
            for (size_t i = table_patch.size(); i < max_patch; ++i)
                f_out << "0 0 ";
            f_out << endl;
        }
        else
        {
            for (size_t i = 0; i < max_patch; ++i)
                f_out << "0 0 ";
            f_out << endl;
        }
    }

    f_out << "CELL_DATA " << polygon_cell_.size() << endl;
    f_out << "SCALARS edge double 1" << endl;
    f_out << "LOOKUP_TABLE default" << endl;
    for (const auto& cell : polygon_cell_)
    {
        Vector3st lattice = cell.first;
        for (size_t i = 0; i < 3; ++i)
        {
            lattice(i) = lattice(i) % 2;
        }
        assert(color_id_.count(lattice));
        f_out << color_id_.at(lattice) << endl;
    }
    // for (size_t i = 0; i < num_cell; ++i)
    // {
    //   double c = min(1.0, fmod(i * 0.09091, 1.09092));
    //   f_out << c << endl;
    // }

    f_out << "LOOKUP_TABLE my_table " << color.size() << endl;
    for (const auto& c : color)
    {
        f_out << c[0] << " " << c[1] << " "
              << c[2] << " " << c[3] << endl;
    }

    // f_out << "0.98    1      0.94     1" << endl; // white
    // f_out << "0.890 0.090 0.051 1" << endl; // red
    // f_out << "0.4 1 0.3 1" << endl; // blue
    // f_out << "0.2666667 0.8 0 1" << endl; // blue
    // f_out << "0 0.302 0.902 1" << endl; // blue
    // f_out << "0.902 0.75 0 1" << endl; // blue
    // f_out << "0.60 0.00 0.40 1" << endl;
    // f_out << "0.50 0.50 0.00 1" << endl;
    // f_out << "0.0 1 0.604 1" << endl;
    // f_out << "0.518 0.702 0.373 1" << endl;
    // f_out << "0.863 0.078 0.235 1" << endl;
    // f_out << "0.729 0.773 0.1 1" << endl;

    f_out.seekp(pos);
    f_out << total_v;
    f_out.close();

    return 0;
}
