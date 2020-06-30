#ifndef _OBJECT_
#define _OBJECT_

#include "Common/def.h"
#include "head.h"
#include "myvector.h"
#include "vertex.h"
#include "edge.h"
#include "simplex.h"
#include "matrix_form_cuda_jacobi_solver.h"
#include <iostream>

//It is the most important class
template<typename T,size_t dim>
class mass_spring_obj : public PhysIKA::Functional<T, dim>
{
 public:
  // function of Functional
  virtual size_t Nx() const;
  virtual int Val(const T *x,
                  std::shared_ptr<PhysIKA::dat_str_core<T,dim>>& data) const;
  virtual int Gra(const T *x,
                  std::shared_ptr<PhysIKA::dat_str_core<T,dim>>& data) const;
  virtual int Hes(const T *x,
                  std::shared_ptr<PhysIKA::dat_str_core<T,dim>>& data) const;

  int save_x_regardless_of_const(const T* x) const;



  mass_spring_obj();
  mass_spring_obj(std::string input_dir,T dt,T density,int line_search,T weight_line_search,T stiffness,std::string newton_fastMS);
  ~mass_spring_obj();


  T time_all;
  T norm_Jacobian_cal;

  Eigen::SimplicialLLT<Eigen::SparseMatrix<double > > linearSolver;
  wtyatzoo::matrix_form_cuda_jacobi_solver<double > *my_matrix_form_cuda_jacobi_solver;

  std::vector<std::pair<T,T > > time_norm_pair;

  // basic element
  std::vector<vertex<T,dim> > myvertexs; //need to save in vtk

  // which simplex 
  std::vector<simplex<T> > mysimplexs; //need to save in vtk
  //mass spring system's base
  std::vector<edge<T,dim> > myedges; //need to save in vtk

  int dim_simplex;
  size_t num_all_dof; // dof for all
  size_t num_cal_dof; // dof for calculating
  
  size_t num_vertex;
  size_t num_edges;
  size_t num_simplexs;
  
  size_t num_fixed;

  size_t* mapIndexToLocInMartix;
  size_t* mapLocInMatrixToIndex;
  
  bool converge;

  T density;
  T stiffness;
  //dt decides whether it is static or dynamic simulation 
  T dt;

  // for chebyshev semi-iterative method 
  // {
  T rho;
  T omega;
  T gamma;
  size_t start_iteration_num; // the begin of the acceleration 
  // }

  std::string newton_fastMS; // newton fastMS_original fastMS_ChebyshevSIM
  int pre_succeed; //for fast mass spring method,precompute the decomposition of the matrix
  //if is newton, with a line search strategy
  int line_search;
  T weight_line_search;
  
  size_t iteration_num;
  size_t max_iteration_num;

  // trick for no penetration of a sphere
  T intensity;
  
  int prepare();

  int getEdges();
  int checkFixedOrFree();
  int init_Energy_now_ForEdge();
  int calMassForVertex();
  int dynamicSimulator(); // if d1dt equals zero,it is a static simulator also.
  int calJacobianAndHessianForEdge(std::vector<Eigen::Triplet<double> > &tripletsForHessian,Eigen::VectorXd &Jacobian);
  bool checkInversion(Eigen::VectorXd &dx);
  bool checkInversion(Eigen::VectorXd &dx,Eigen::VectorXd &Jacobian);
  T calEnergyDif();
  T calElasticEnergy();

  std::vector<Eigen::Triplet<double>> tripletsForHessianspa;
  // Eigen::SparseMatrix<double,Eigen::RowMajor> Hessianspa;
  Eigen::VectorXd Jacobian_cal;
  T elasticE;
};

template class mass_spring_obj<double,3>;
template class mass_spring_obj<float,3>;
#endif
