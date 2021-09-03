/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: io utility
 * @version    : 1.0
 */
#include "FEMIo.h"

#include <iostream>
#include <fstream>
#include <iomanip>
#include "FEMIoVtk.h"

using namespace std;
using namespace Eigen;
namespace PhysIKA {

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
int write_SPM(const char* path, const Eigen::SparseMatrix<double, Eigen::RowMajor>& A)
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

int write_SPM(const char* path, const Eigen::SparseMatrix<float, Eigen::RowMajor>& A)
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

}  // namespace PhysIKA