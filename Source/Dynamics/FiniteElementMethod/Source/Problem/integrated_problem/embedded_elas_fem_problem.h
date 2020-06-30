#ifndef PhysIKA_GEN_EMBEDDED_ELAS_PROBLEM
#define PhysIKA_GEN_EMBEDDED_ELAS_PROBLEM
#include <boost/property_tree/ptree.hpp>

#include "Common/framework.h"
#include "Problem/constraint/constraints.h"
#include "Problem/energy/basic_energy.h"
#include "Model/fem/elas_energy.h"
#include "Geometry/embedded_interpolate.h"

namespace PhysIKA{

template<typename T>
class embedded_elas_problem_builder : public embedded_problem_builder<T, 3>{
 public:
  embedded_elas_problem_builder(const T*x, const boost::property_tree::ptree& pt); 
  std::shared_ptr<Problem<T, 3>> build_problem() const;

  int update_problem(const T* x, const T* v = nullptr);
  
  Eigen::Matrix<T, -1, -1> get_nods()const {return REST_;}
  Eigen::MatrixXi get_cells()const {return cells_;}
  std::shared_ptr<constraint_4_coll<T>> get_collider() const {return collider_;}
  const Eigen::SparseMatrix<T> &get_coarse_to_fine_coefficient() const { return coarse_to_fine_coef_; }
  const Eigen::SparseMatrix<T> &get_fine_to_coarse_coefficient() const { return fine_to_coarse_coef_; }

  std::shared_ptr<elas_intf<T, 3>> get_elas_energy()const{
    return elas_intf_;
  }

  std::shared_ptr<embedded_interpolate<T>> get_embedded_interpolate() { return embedded_interp_;}

private:
  Eigen::Matrix<T, -1, -1> REST_;
  Eigen::MatrixXi cells_;
  int fine_verts_num_;
  std::shared_ptr<embedded_interpolate<T>> embedded_interp_;

  
  Eigen::SparseMatrix<T> coarse_to_fine_coef_; // coarse * coef = fine
  Eigen::SparseMatrix<T> fine_to_coarse_coef_; // fine * coef = coarse

  std::shared_ptr<constraint_4_coll<T>> collider_;
  std::shared_ptr<momentum<T, 3>> kinetic_;
  std::vector<std::shared_ptr<Functional<T, 3>>> ebf_;
  std::vector<std::shared_ptr<Constraint<T>>> cbf_;
  std::shared_ptr<elas_intf<T, 3>> elas_intf_;

  const boost::property_tree::ptree pt_;
  

  
};






}
#endif
