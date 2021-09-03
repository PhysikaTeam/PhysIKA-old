#pragma once

#include <Eigen/Core>

namespace marvel {

double clo_surf_vol(const Eigen::MatrixXd& nods, const Eigen::MatrixXi& surf);
int    build_bdbox(const Eigen::MatrixXd& nods, Eigen::MatrixXd& bdbox);

}  // namespace marvel
