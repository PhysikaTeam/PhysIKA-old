#pragma once

#include <vector>
#include <Eigen/Core>

int write_to_vtk(const std::vector<Eigen::MatrixXd>& table_element,
                 const char* const                   path            = "demo.vtk",
                 double*                             ptr_table_color = nullptr);
