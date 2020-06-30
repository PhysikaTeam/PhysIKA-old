#ifndef _MATRIX_FORM_CUDA_JACOBI_SOLVER_
#define _MATRIX_FORM_CUDA_JACOBI_SOLVER_

#include "head.h"
#include "triplet.h"
#define USE_CUDA
namespace wtyatzoo {

#ifdef USE_CUDA
  template<typename T>
  class matrix_form_cuda_jacobi_solver
  {
  public:
    matrix_form_cuda_jacobi_solver(T**A,const T*dig_A_coefficient,const int N); // assemble from dense matrix
    matrix_form_cuda_jacobi_solver(const std::vector<triplet<T> > &mytriplets_A,const int N); // assemble from input coo vector
    matrix_form_cuda_jacobi_solver(const T* val_A,const int* col_index_A,const int* row_offset_A,const int num_non_zero_A,const int N);
    matrix_form_cuda_jacobi_solver(){}
    ~matrix_form_cuda_jacobi_solver();
    int apply(const T b[],T x[],const T tol,const int max_iteration_num,const bool given);
    // given=0: without given initial value, so the initial values are forced to be all zero! given=1: with given initial value to iterate 


    //for matrix form jacobi solver, the iteration scheme is formulated as x(k)=T*x(k-1)+c where T=(D-1)*(L+U) c=(D-1)*b
  private:
    cublasHandle_t my_cublas_handle;
    cusparseHandle_t my_cusparse_handle;
    cusparseMatDescr_t my_descr;
    // array
    T *val_T_device; 
    int *col_index_T_device;
    int *row_offset_T_device;

    T *b_device;
    T *dig_A_coefficient_inverse_device;
    T *dig_inverse_b_device;
    
    T *x_device_iteration; //2*N*sizeof(double)
    T *residual_device;

    // single variable
    int N;
    int num_non_zero_T;
    T residual_sum_host;

    
  };
  template class matrix_form_cuda_jacobi_solver<double>;
  template class matrix_form_cuda_jacobi_solver<float>;
#endif
  
}
#endif
