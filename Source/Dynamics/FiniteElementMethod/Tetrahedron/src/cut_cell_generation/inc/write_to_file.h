#ifndef FWRITE_TO_FILE_JJ_H
#define FWRITE_TO_FILE_JJ_H

#include <vector>
#include <Eigen/Core>

int write_to_vtk(const std::vector<Eigen::MatrixXd>& table_element,
                 const char* const                   path            = "demo.vtk",
                 double*                             ptr_table_color = nullptr);

#endif  // FWRITE_TO_FILE_JJ_H
