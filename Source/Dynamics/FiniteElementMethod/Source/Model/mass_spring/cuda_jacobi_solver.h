#ifndef _CUDA_JACOBI_SOLVER_
#define _CUDA_JACOBI_SOLVER_

#include "head.h"
#include "triplet.h"
#define USE_CUDA
namespace wtyatzoo {

#ifdef USE_CUDA
  template<typename T>
  class cuda_jacobi_solver
  {
  public:
    cuda_jacobi_solver(T**A,const T*dig_A_coefficient,const size_t N); // assemble from dense matrix
    cuda_jacobi_solver(const std::vector<triplet<T> > &mytriplets_A,const int N); // assemble from input coo vector
    cuda_jacobi_solver(const T* val_A,const int* col_index_A,const int* row_offset_A,const int num_non_zero_A,const int N);
    cuda_jacobi_solver(){}
    ~cuda_jacobi_solver();
    int apply(const T b[],T x[],const T tol,const size_t max_iteration_num,const bool given);
    // given=0: without given initial value, so the initial values are forced to be all zero! given=1: with given initial value to iterate 
  
  private:
    // array
    T *dig_A_coefficient_device;
    T *val_device;
    size_t *col_index_device;
    size_t *row_offset_device;
    T *b_device;

    T *x_device_iteration; //2*N*sizeof(double)
    T *residual_device;

    // single variable
    size_t N;
    size_t num_non_zero;
    T *residual_sum_device;
    T residual_sum_host;
  };

  template class cuda_jacobi_solver<double>;
  template class cuda_jacobi_solver<float>;
#endif
  
}
#endif
