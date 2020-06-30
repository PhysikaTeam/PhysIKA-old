#include "head.h"
#include "configure.h"
#include "helper_cuda.h"
#include "triplet.h"
#include "cuda_jacobi_solver.h"
using namespace std;

template<typename T>
__global__ void jacobi_solve_csr(const T*dig_A_coefficient_device, const T*val_device, const size_t *col_index_device,const size_t*row_offset_device,const T* b_device,T* x_device_iteration,int new_iteration,size_t N,size_t num_non_zero)
{
  int i = blockIdx.x*blockDim.x+threadIdx.x;
  int old_iteration=!new_iteration;
  if (i<N)
    {
      int j;
      T sum=0;
      for(j=row_offset_device[i];j<row_offset_device[i+1];j++)
	{
	  if(col_index_device[j]!=i)
	    {
	      T x_here=x_device_iteration[old_iteration*N+col_index_device[j]];
	      sum-=val_device[j]*x_here;
	    }
	}
      sum+= b_device[i];
      x_device_iteration[new_iteration*N+i]=sum/dig_A_coefficient_device[i];     
    }
}

template<typename T>
__global__ void init(T* x_device_iteration,size_t N_2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N_2)
    {
      x_device_iteration[i]=0;
    }
}

template<typename T>
__global__ void get_tol_csr(T* x_device_iteration,T* residual_device,size_t N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N)
    {
      T x=x_device_iteration[i];
      T y=x_device_iteration[N+i];
      residual_device[i]=(x-y)*(x-y);
    }
}

template<typename T>
__global__ void sum_tol(const T* residual_device,T *residual_sum_device,size_t N)
{
  *residual_sum_device=0.0;
  for(size_t i=0;i<N;++i)
    {
      *residual_sum_device+=residual_device[i];
    }
}

namespace wtyatzoo
{
  template<typename T>
  cuda_jacobi_solver<T>::cuda_jacobi_solver(const T* val_A,const int* col_index_A,const int* row_offset_A,const int num_non_zero_A,const int N)
  {
    size_t i,j;
    this->num_non_zero=num_non_zero_A;
    this->N=N;
    printf("num_non_zero :%d\n N :%d\n",num_non_zero,N);
    
    T* dig_A_coefficient=(T*)malloc(sizeof(T)*N);

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

    checkCudaErrors(cudaMalloc((void**)&val_device,sizeof(T)*num_non_zero));
    checkCudaErrors(cudaMalloc((void**)&col_index_device,sizeof(size_t)*num_non_zero));
    checkCudaErrors(cudaMalloc((void**)&row_offset_device,sizeof(size_t)*(N+1)));
    checkCudaErrors(cudaMalloc((void**)&dig_A_coefficient_device,sizeof(T)*N));
    
    checkCudaErrors(cudaMemcpy(val_device,val_A,sizeof(T)*num_non_zero,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(col_index_device,col_index_A,sizeof(size_t)*num_non_zero,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(row_offset_device,row_offset_A,sizeof(size_t)*(N+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dig_A_coefficient_device,dig_A_coefficient,sizeof(T)*N,cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void**)&residual_sum_device,sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&b_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&x_device_iteration,sizeof(T)*2*N));
    checkCudaErrors(cudaMalloc((void**)&residual_device,sizeof(T)*N));
    // cpu's used memory
    free(dig_A_coefficient);
  }
  template<typename T>
  cuda_jacobi_solver<T>::cuda_jacobi_solver(const std::vector<triplet<T> > &mytriplets_A,const int N)
  {
    size_t i,j;
    this->num_non_zero=mytriplets_A.size();
    this->N=N;
    printf("num_non_zero :%d\n N :%d\n",num_non_zero,N);
    
    size_t* row_offset=(size_t*)malloc(sizeof(size_t)*(N+1));
    size_t* col_index=(size_t*)malloc(sizeof(size_t)*num_non_zero);
    T* val=(T*)malloc(sizeof(T)*num_non_zero);

    T* dig_A_coefficient=(T*)malloc(sizeof(T)*N);
    
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

    checkCudaErrors(cudaMalloc((void**)&val_device,sizeof(T)*num_non_zero));
    checkCudaErrors(cudaMalloc((void**)&col_index_device,sizeof(size_t)*num_non_zero));
    checkCudaErrors(cudaMalloc((void**)&row_offset_device,sizeof(size_t)*(N+1)));
    checkCudaErrors(cudaMalloc((void**)&dig_A_coefficient_device,sizeof(T)*N));
    
    checkCudaErrors(cudaMemcpy(val_device,val,sizeof(T)*num_non_zero,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(col_index_device,col_index,sizeof(size_t)*num_non_zero,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(row_offset_device,row_offset,sizeof(size_t)*(N+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dig_A_coefficient_device,dig_A_coefficient,sizeof(T)*N,cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void**)&residual_sum_device,sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&b_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&x_device_iteration,sizeof(T)*2*N));
    checkCudaErrors(cudaMalloc((void**)&residual_device,sizeof(T)*N));
    // cpu's used memory
    free(row_offset);
    free(col_index);
    free(val);
    free(dig_A_coefficient);
  }
  
  template<typename T>
  cuda_jacobi_solver<T>::cuda_jacobi_solver( T**A,const T*dig_A_coefficient,const size_t N)
  {
    //printf("%lf\n",*(*(A+0)+0));
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
    
    size_t* row_offset=(size_t*)malloc(sizeof(size_t)*(N+1));
    size_t* col_index=(size_t*)malloc(sizeof(size_t)*num_non_zero);
    T* val=(T*)malloc(sizeof(T)*num_non_zero);
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

    checkCudaErrors(cudaMalloc((void**)&val_device,sizeof(T)*num_non_zero));
    checkCudaErrors(cudaMalloc((void**)&col_index_device,sizeof(size_t)*num_non_zero));
    checkCudaErrors(cudaMalloc((void**)&row_offset_device,sizeof(size_t)*(N+1)));
    checkCudaErrors(cudaMalloc((void**)&dig_A_coefficient_device,sizeof(T)*N));
    
    checkCudaErrors(cudaMemcpy(val_device,val,sizeof(T)*num_non_zero,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(col_index_device,col_index,sizeof(size_t)*num_non_zero,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(row_offset_device,row_offset,sizeof(size_t)*(N+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dig_A_coefficient_device,dig_A_coefficient,sizeof(T)*N,cudaMemcpyHostToDevice));


    checkCudaErrors(cudaMalloc((void**)&residual_sum_device,sizeof(T)));
    checkCudaErrors(cudaMalloc((void**)&b_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&x_device_iteration,sizeof(T)*2*N));
    checkCudaErrors(cudaMalloc((void**)&residual_device,sizeof(T)*N));
    // cpu's used memory
    free(row_offset);
    free(col_index);
    free(val);
  }

  template<typename T>
  cuda_jacobi_solver<T>::~cuda_jacobi_solver()
  {
    checkCudaErrors(cudaFree(dig_A_coefficient_device));
    checkCudaErrors(cudaFree(val_device));
    checkCudaErrors(cudaFree(col_index_device));
    checkCudaErrors(cudaFree(row_offset_device));

    checkCudaErrors(cudaFree(residual_sum_device));
    checkCudaErrors(cudaFree(b_device));

    checkCudaErrors(cudaFree(x_device_iteration));
    
    checkCudaErrors(cudaFree(residual_device));
    
  }

  template<typename T>
  int cuda_jacobi_solver<T>::apply(const T b[],T x[],const T tol,const size_t max_iteration_num,const bool given)
  {
    
    int block_size=1024;
    int num_block=(int)ceil(N/(T)block_size);

    size_t iteration_num_now=0;
    
    checkCudaErrors(cudaMemcpy(b_device,b,sizeof(T)*N,cudaMemcpyHostToDevice));

    if(given==0)
      {
	init<<<num_block*2,block_size>>>(x_device_iteration,2*N);
      }
    else if(given==1)
      {
	checkCudaErrors(cudaMemcpy(x_device_iteration,x,sizeof(T)*N,cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(x_device_iteration+N,x,sizeof(T)*N,cudaMemcpyHostToDevice));
      }
    

    int new_iteration=1;
    residual_sum_host=0;
    for(iteration_num_now=0;iteration_num_now<max_iteration_num;iteration_num_now++)
      {
        jacobi_solve_csr<<<num_block,block_size>>>(dig_A_coefficient_device,val_device,col_index_device,row_offset_device,b_device,x_device_iteration,new_iteration,N,num_non_zero);
	get_tol_csr<<<num_block,block_size>>>(x_device_iteration,residual_device,N);
        sum_tol<<<1,1>>>(residual_device,residual_sum_device,N);
	
	checkCudaErrors(cudaMemcpy(&residual_sum_host,residual_sum_device,sizeof(T),cudaMemcpyDeviceToHost));
	if(residual_sum_host<tol)
	  {
	    break;
	  }
	new_iteration=!new_iteration;
      }
    printf("iteration_num_now ::%d\n",iteration_num_now);
    checkCudaErrors(cudaMemcpy(x,x_device_iteration, N*sizeof(T), cudaMemcpyDeviceToHost));
    return 0;
  }
}
