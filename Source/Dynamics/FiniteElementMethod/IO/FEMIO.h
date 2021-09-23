/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: io utility
 * @version    : 1.0
 */
#pragma once

#include <fstream>
#include <iomanip>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <iostream>
#include "FEMIOvtk.h"
// using mati_t=zjucad::matrix::matrix<size_t>;
// using matd_t=zjucad::matrix::matrix<double>;

namespace PhysIKA {

/**
 * @brief Read the fixed vertexs data from a csv file
 * 
 * @param filename 
 * @param fixed 
 * @param pos 
 * @return int 
 */
int read_fixed_verts_from_csv(const char* filename, std::vector<size_t>& fixed, Eigen::MatrixXd* pos = nullptr);

/**
 * @brief Write the MAT data to a file
 * 
 * @param path 
 * @param A 
 * @return int 
 */
int write_MAT(const char* path, const Eigen::MatrixXd& A);

/**
 * @brief Write the SPM data to a file
 * 
 * @param path 
 * @param A 
 * @return int 
 */
int write_SPM(const char* path, const Eigen::SparseMatrix<double, Eigen::RowMajor>& A);

/**
 * @brief Write the SPM data to a file
 * 
 * @param path 
 * @param A 
 * @return int 
 */
int write_SPM(const char* path, const Eigen::SparseMatrix<float, Eigen::RowMajor>& A);

/**
 * @brief Write the triangle mesh data to a file in vtk format
 * 
 * @param path 
 * @param nods 
 * @param tris 
 * @param mtr 
 * @return int 
 */
int tri_mesh_write_to_vtk(const char* path, const Eigen::MatrixXd& nods, const Eigen::MatrixXi& tris, const Eigen::MatrixXd* mtr = nullptr);
// int quad_mesh_write_to_vtk(const char *path, const matd_t &nods, const mati_t &quad,
//                            const matd_t *mtr=nullptr, const char *type="CELL");

/**
 * @brief Write the points to a file in vtk format
 * 
 * @param path 
 * @param nods 
 * @param num_points 
 * @return int 
 */
int point_write_to_vtk(const char* path, const double* nods, const size_t num_points);

/**
 * @brief Append the points to a file in vtk format
 * 
 * @param is_append 
 * @param path 
 * @param vectors 
 * @param num_vecs 
 * @param vector_name 
 * @return int 
 */
int point_vector_append2vtk(const bool is_append, const char* path, const Eigen::MatrixXd& vectors, const size_t num_vecs, const char* vector_name);

/**
 * @brief Write the scalar data of points to a file in vtk format
 * 
 * @param is_append 
 * @param path 
 * @param scalars 
 * @param num_sca 
 * @param scalar_name 
 * @return int 
 */
int point_scalar_append2vtk(const bool is_append, const char* path, const Eigen::VectorXd& scalars, const size_t num_sca, const char* scalar_name);

/**
 * @brief Read the mesh data from a vtk file
 * 
 * @tparam T 
 * @tparam num_vert 
 * @tparam dim 
 * @param filename 
 * @param nods 
 * @return int 
 */
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

/**
 * @brief Read the mesh data from a vtk file
 * 
 * @tparam T 
 * @tparam num_vert 
 * @tparam dim 
 * @param filename 
 * @param nods 
 * @param cells 
 * @param mtr 
 * @return int 
 */
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

/**
 * @brief Write the mesh data to a vtk file
 * 
 * @tparam FLOAT 
 * @tparam num_vert 
 * @param path 
 * @param nods 
 * @param cells 
 * @param mtr 
 * @param dim 
 * @return int 
 */
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
