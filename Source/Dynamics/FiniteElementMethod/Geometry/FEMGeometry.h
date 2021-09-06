#pragma once

#include <Eigen/Core>

namespace PhysikaFEM {

/**
 * @brief Get the surface data
 * 
 * @param nods 
 * @param surf 
 * @return double 
 */
double clo_surf_vol(const Eigen::MatrixXd& nods, const Eigen::MatrixXi& surf);

/**
 * @brief build the bounding box
 * 
 * @param nods 
 * @param bdbox 
 * @return int 
 */
int build_bdbox(const Eigen::MatrixXd& nods, Eigen::MatrixXd& bdbox);

}  // namespace PhysikaFEM
