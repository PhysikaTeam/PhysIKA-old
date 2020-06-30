#ifndef _JACOBI_SOLVER_
#define _JACOBI_SOLVER_

#include "head.h"
#include "triplet.h"
namespace wtyatzoo {

  template<typename T>
  class jacobi_solver
  {
  public:
    jacobi_solver(T**A,const T*dig_A_coefficient,const size_t N); //assemble from dense matrix
    jacobi_solver(const std::vector<triplet<T> > &mytriplets_A,const int N); // assemble from input coo vector
    jacobi_solver(const T* val_A,const int* col_index_A,const int* row_offset_A,const int num_non_zero_A,const int N);
    jacobi_solver(){}
    ~jacobi_solver();
    int apply(const T b[],T x[],const T tol,const size_t max_iteration_num,const bool given);
    // given=0: without given initial value, so the initial values are forced to be all zero! given=1: with given initial value to iterate 
  public:
     int num_threads;
  private:
    // array
    T *dig_A_coefficient;
    T *val;
    size_t *col_index;
    size_t *row_offset;
    T *b;

    T *x_iteration; //2*N*sizeof(T)

    // single variable
    size_t N;
    size_t num_non_zero;
    T residual_sum;

    void init(const bool given);
    void jacobi_solve_csr(int new_iteration);
    void sum_tol();
  };

  template class jacobi_solver<double>;
  template class jacobi_solver<float>;
  
}
#endif

