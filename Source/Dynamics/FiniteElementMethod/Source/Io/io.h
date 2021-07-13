/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: io utility
 * @version    : 1.0
 */
#ifndef IO_H

#define IO_H

#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "vtk.h"
// using mati_t=zjucad::matrix::matrix<size_t>;
// using matd_t=zjucad::matrix::matrix<double>;

namespace PhysIKA {

int read_fixed_verts_from_csv(const char* filename, std::vector<size_t>& fixed, Eigen::MatrixXd* pos = nullptr);

int write_MAT(const char* path, const Eigen::MatrixXd& A);
int write_SPM(const char* path, const Eigen::SparseMatrix<double>& A);

int tri_mesh_write_to_vtk(const char* path, const Eigen::MatrixXd& nods, const Eigen::MatrixXi& tris, const Eigen::MatrixXd* mtr = nullptr);
// int quad_mesh_write_to_vtk(const char *path, const matd_t &nods, const mati_t &quad,
//                            const matd_t *mtr=nullptr, const char *type="CELL");
int point_write_to_vtk(const char* path, const double* nods, const size_t num_points);
int point_vector_append2vtk(const bool is_append, const char* path, const Eigen::MatrixXd& vectors, const size_t num_vecs, const char* vector_name);
int point_scalar_append2vtk(const bool is_append, const char* path, const Eigen::VectorXd& scalars, const size_t num_sca, const char* scalar_name);

template <typename T, size_t num_vert, size_t dim = 3>
int mesh_read_from_vtk(const char* filename, Eigen::Matrix<T, -1, -1>& nods)
{
    std::ifstream ifs(filename);
    if (ifs.fail())
    {
        std::cerr << "[info] "
                  << "can not open file" << filename << std::endl;
        return __LINE__;
    }

    std::string str;
    int         point_num = 0, cell_num = 0;

    while (!ifs.eof())
    {
        ifs >> str;
        if (str == "POINTS")
        {
            ifs >> point_num >> str;
            nods = Eigen::Matrix<T, -1, -1>(dim, point_num);
            T item;
            for (size_t i = 0; i < point_num; ++i)
            {
                for (size_t j = 0; j < dim; ++j)
                {
                    ifs >> nods(j, i);
                }
            }
            continue;
        }
    }
    ifs.close();

    return 0;
}

template <typename T, size_t num_vert, size_t dim = 3>
int mesh_read_from_vtk(const char* filename, Eigen::Matrix<T, -1, -1>& nods, Eigen::MatrixXi& cells, T* mtr = nullptr)
{
    std::ifstream ifs(filename);
    if (ifs.fail())
    {
        std::cerr << "[info] "
                  << "can not open file" << filename << std::endl;
        return __LINE__;
    }

    std::string str;
    int         point_num = 0, cell_num = 0;

    while (!ifs.eof())
    {
        ifs >> str;
        if (str == "POINTS")
        {
            ifs >> point_num >> str;
            nods = Eigen::Matrix<T, -1, -1>(dim, point_num);
            T item;
            for (size_t i = 0; i < point_num; ++i)
            {
                for (size_t j = 0; j < dim; ++j)
                {
                    ifs >> nods(j, i);
                }
            }
            continue;
        }
        if (str == "CELLS")
        {
            ifs >> cell_num >> str;
            int point_number_of_cell = 0;
            cells                    = Eigen::Matrix<int, -1, -1>(num_vert, cell_num);
            size_t true_cell_num     = 0;
            for (size_t ci = 0; ci < cell_num; ++ci)
            {
                ifs >> point_number_of_cell;
                if (point_number_of_cell != num_vert)
                {
                    for (size_t i = 0; i < point_number_of_cell; ++i)
                        ifs >> str;
                }
                else
                {
                    int p;
                    for (size_t i = 0; i < point_number_of_cell; ++i)
                    {
                        ifs >> p;
                        cells(i, true_cell_num) = p;
                    }

                    ++true_cell_num;
                }
            }
            break;
        }
    }

    std::vector<T> tmp;
    T              mtrval;
    if (mtr != nullptr)
    {

        while (!ifs.eof())
        {
            ifs >> str;
            if (str == "LOOKUP_TABLE")
            {
                ifs >> str;
                for (size_t i = 0; i < cells.cols(); ++i)
                {
                    ifs >> mtrval;
                    tmp.push_back(mtrval);
                }
            }
        }

        if (tmp.size() > 0)
        {
            assert(tmp.size() % cells.cols() == 0);
            Eigen::Map<Eigen::Matrix<T, -1, -1>> mtr_mat(mtr, cells.cols(), tmp.size() / cells.cols());
            std::copy(tmp.begin(), tmp.end(), mtr_mat.data());
            mtr_mat.transposeInPlace();
        }
    }

    ifs.close();

    return 0;
}
template <typename FLOAT, size_t num_vert>
int mesh_write_to_vtk(const char* path, const Eigen::Ref<Eigen::Matrix<FLOAT, -1, -1>> nods, const Eigen::Ref<Eigen::MatrixXi> cells, const Eigen::Matrix<FLOAT, -1, -1>* mtr = nullptr, size_t dim = 3)
{
    assert(cells.rows() == num_vert);

    std::ofstream ofs(path);
    if (ofs.fail())
        return __LINE__;

    ofs << std::setprecision(15);
    if (dim == 3 && num_vert == 4)
        tet2vtk(ofs, nods.data(), nods.cols(), cells.data(), cells.cols());
    else if (dim == 3 && num_vert == 8)
        hex2vtk(ofs, nods.data(), nods.cols(), cells.data(), cells.cols());
    else if (dim == 2 && num_vert == 4)
        quad2vtk(ofs, nods.data(), nods.cols(), cells.data(), cells.cols());
    else if (dim == 2 && num_vert == 3)
        tri2vtk(ofs, nods.data(), nods.cols(), cells.data(), cells.cols());

    if (mtr != nullptr)
    {
        for (int i = 0; i < mtr->rows(); ++i)
        {
            const std::string                 mtr_name = "theta_" + std::to_string(i);
            const Eigen::Matrix<FLOAT, 1, -1> curr_mtr = mtr->row(i);
            if (i == 0)
                ofs << "CELL_DATA " << curr_mtr.size() << "\n";
            vtk_data(ofs, curr_mtr.data(), curr_mtr.cols(), mtr_name.c_str(), mtr_name.c_str());
        }
    }
    ofs.close();
    return 0;
}

}  // namespace PhysIKA

#endif
