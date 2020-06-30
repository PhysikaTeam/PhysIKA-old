#include "head.h"
#include "configure.h"
#include "helper_cuda.h"
#include "triplet.h"
#include "jacobi_solver.h"
using namespace std;

namespace wtyatzoo
{
  template<typename T>
  void jacobi_solver<T>::jacobi_solve_csr(int new_iteration)
  {
    int old_iteration=!new_iteration;

    // just thread level parallel the first for loop
    omp_set_num_threads(this->num_threads);  
#pragma omp parallel for 
    for(size_t i=0;i<N;++i)
      {
	//printf("i = %u, I am Thread %d\n", i, omp_get_thread_num());
	int j;
	T sum=0;
	for(j=row_offset[i];j<row_offset[i+1];j++)
	  {
	    if(col_index[j]!=i)
	      {
		T x_here=x_iteration[old_iteration*N+col_index[j]];
		sum-=val[j]*x_here;
	      }
	  }
	sum+= b[i];
	x_iteration[new_iteration*N+i]=sum/dig_A_coefficient[i];     
      }
  }

  template<typename T>
  void jacobi_solver<T>::init(const bool given)
  {
    // just thread level parallel the first for loop
    if(given==0)
      {
	omp_set_num_threads(this->num_threads);  
#pragma omp parallel for
	for(size_t i=0;i<N*2;++i)
	  {
	    x_iteration[i]=0;
	  }
      }
    else if(given==1)
      {
	omp_set_num_threads(this->num_threads);  
#pragma omp parallel for
	for(size_t i=0;i<N;++i)
	  {
	    x_iteration[i+N]=x_iteration[i];
	  }
      }
    
  }

  template<typename T>
  void jacobi_solver<T>::sum_tol()
  {
    residual_sum=0.0;
    // just thread level parallel the first for loop 
    omp_set_num_threads(this->num_threads);  
#pragma omp parallel for
    for(size_t i=0;i<N;++i)
      {
	residual_sum+=((x_iteration[i]-x_iteration[N+i])*(x_iteration[i]-x_iteration[N+i]));
      }
  }

  template<typename T>
  jacobi_solver<T>::jacobi_solver(const T* val_A,const int* col_index_A,const int* row_offset_A,const int num_non_zero_A,const int N)
  {
    num_threads=6;
    size_t i,j;
    this->num_non_zero=num_non_zero_A;
    this->N=N;
    printf("num_non_zero :%d\n N :%d\n",num_non_zero,N);

    this->row_offset=(size_t*)malloc(sizeof(size_t)*(N+1));
    this->col_index=(size_t*)malloc(sizeof(size_t)*num_non_zero);
    this->val=(T*)malloc(sizeof(T)*num_non_zero);
    this->dig_A_coefficient=(T*)malloc(sizeof(T)*N);

    for(i=0;i<N;++i)
      {
	for(j=row_offset_A[i];j<row_offset_A[i+1];++j)
	  {
	    if(col_index_A[j]==i)
	      {
		dig_A_coefficient[i]=val_A[j];
	      }
	  }
      }
    memcpy(row_offset,row_offset_A,sizeof(size_t)*(N+1));
    memcpy(col_index,col_index_A,sizeof(size_t)*(num_non_zero));
    memcpy(val,val_A,sizeof(T)*num_non_zero);

  }

  template<typename T>
  jacobi_solver<T>::jacobi_solver(const std::vector<triplet<T> > &mytriplets_A,const int N)// assemble from input coo vector
  {
    num_threads=6;
    size_t i,j;
    this->num_non_zero=mytriplets_A.size();
    this->N=N;
    printf("num_non_zero :%d\n N :%d\n",num_non_zero,N);
    
    this->row_offset=(size_t*)malloc(sizeof(size_t)*(N+1));
    this->col_index=(size_t*)malloc(sizeof(size_t)*num_non_zero);
    this->val=(T*)malloc(sizeof(T)*num_non_zero);
    this->dig_A_coefficient=(T*)malloc(sizeof(T)*N);
    
    size_t offset_now=0;
    size_t row_now=0;
    row_offset[0]=0;
    for(i=0;i<num_non_zero;++i)
      {
	if(mytriplets_A[i].row==row_now)
	  {
	    row_offset[row_now]=offset_now;
	    row_now++; offset_now++;
	  }
	else
	  {
	    offset_now++;
	  }

	if(mytriplets_A[i].col==mytriplets_A[i].row)
	  {
	    dig_A_coefficient[mytriplets_A[i].col]=mytriplets_A[i].val;
	  }
	col_index[i]=mytriplets_A[i].col;
	val[i]=mytriplets_A[i].val;
      }
    row_offset[N]=num_non_zero;
  }
  
  template<typename T>
  jacobi_solver<T>::jacobi_solver( T**A,const T*dig_A_coefficient,const size_t N)
  {
    num_threads=6;
    //    printf("%lf\n",*(*(A+0)+0));
    // assemble the sparse matrix structure on GPU based on the dese matrix input
    size_t i,j;

    vector<triplet<T> > mytriplets;
    while(!mytriplets.empty())
      {
	mytriplets.pop_back();
      }
    T EPS=1e-10;
    for(i=0;i<N;++i)
      {
	for(j=0;j<N;++j)
	  {
	    //	    printf("%lf\n",A[i][j]);
	    if(fabs(A[i][j])>=EPS)
	      {
		mytriplets.push_back(triplet<T>(i,j,A[i][j]));
	      }
	  }
      }
    this->num_non_zero=mytriplets.size();
    this->N=N;
    printf("num_non_zero :%d\n N :%d\n",num_non_zero,N);
    
    this->row_offset=(size_t*)malloc(sizeof(size_t)*(N+1));
    this->col_index=(size_t*)malloc(sizeof(size_t)*num_non_zero);
    this->val=(T*)malloc(sizeof(T)*num_non_zero);
    this->dig_A_coefficient=(T*)malloc(sizeof(T)*N);
    
    size_t offset_now=0;
    size_t row_now=0;
    row_offset[0]=0;
    for(i=0;i<num_non_zero;++i)
      {
	if(mytriplets[i].row==row_now)
	  {
	    row_offset[row_now]=offset_now;
	    row_now++; offset_now++;
	  }
	else
	  {
	    offset_now++;
	  }
	col_index[i]=mytriplets[i].col;
	val[i]=mytriplets[i].val;
      }
    row_offset[N]=num_non_zero;
    memcpy(this->dig_A_coefficient,dig_A_coefficient,sizeof(T)*N);
  }

  template<typename T>
  jacobi_solver<T>::~jacobi_solver()
  {
    free(dig_A_coefficient);
    free(val);
    free(col_index);
    free(row_offset);
    free(b);
    free(x_iteration);    
  }

  template<typename T>
  int jacobi_solver<T>::apply(const T b[],T x[],const T tol,const size_t max_iteration_num,const bool given)
  {
    size_t iteration_num_now=0;
    this->b=(T*)malloc(sizeof(T)*N);
    this->x_iteration=(T*)malloc(sizeof(T)*2*N);     
    memcpy(this->b,b,sizeof(T)*N);
    memcpy(this->x_iteration,x,sizeof(T)*N); // in case that the given == 1
    init(given);
    int new_iteration=1;
    residual_sum=0;
    for(iteration_num_now=0;iteration_num_now<max_iteration_num;iteration_num_now++)
      {
        jacobi_solve_csr(new_iteration);
        sum_tol();
	if(residual_sum<tol)
	  {
	    break;
	  }
	new_iteration=!new_iteration;
      }
    printf("iteration_num_now ::%d\n",iteration_num_now);
    memcpy(x,x_iteration, N*sizeof(T));
    return 0;
  }
}
