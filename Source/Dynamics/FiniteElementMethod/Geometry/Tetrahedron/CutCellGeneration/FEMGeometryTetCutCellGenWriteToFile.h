#pragma once

#include <vector>
#include <Eigen/Core>

/**
 * @brief Write the data to a vtk file
 * 
 * @param table_element 
 * @param path 
 * @param ptr_table_color 
 * @return int 
 */
int write_to_vtk(const std::vector<Eigen::MatrixXd>& table_element,
                 const char* const                   path            = "demo.vtk",
                 double*                             ptr_table_color = nullptr);
