#include "geometry.h"
#include <Eigen/LU>
#include <Eigen/Geometry>
using namespace Eigen;
namespace marvel {
double clo_surf_vol(const MatrixXd& nods, const MatrixXi& surf)
{
    //TODO:check if the surface is closed and manifold
    double volume = 0;
    for (size_t i = 0; i < surf.cols(); ++i)
    {
        Matrix3d tet;
        for (size_t j = 0; j < 3; ++j)
        {
            tet.row(j) = nods.col(surf(j, i));
        }
        //TODO:check
        volume += tet.determinant();
    }

    return volume;
}

int build_bdbox(const MatrixXd& nods, MatrixXd& bdbox)
{
    //simple bounding box
    //bounding box is a dimension * 2 matrix, whose first column is minimal value and second column is maximal value.
    bdbox = nods.col(0) * MatrixXd::Ones(1, 2);
    for (size_t i = 0; i < nods.cols(); ++i)
    {
        for (size_t j = 0; j < nods.rows(); ++j)
        {
            if (bdbox(j, 0) > nods(j, i))
                bdbox(j, 0) = nods(j, i);
            if (bdbox(j, 1) < nods(j, i))
                bdbox(j, 1) = nods(j, i);
        }
    }
    return 0;
}

}  // namespace marvel
