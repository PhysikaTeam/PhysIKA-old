#include "head.h"
#include "configure.h"
#include "helper_cuda.h"
#include "triplet.h"
#include "matrix_form_cuda_jacobi_solver.h"
using namespace std;

template<typename T>
__global__ void set_Dinverseb(const T* dig_A_coefficient_inverse_device,const T* b_device,T* dig_inverse_b_device,const int  N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N)
  {
    dig_inverse_b_device[i] = dig_A_coefficient_inverse_device[i]*b_device[i];
  }
}

template<typename T>
__global__ void init(T* x_device_iteration,int  N_2)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N_2)
    {
      x_device_iteration[i]=0;
    }
}

template<typename T>
__global__ void get_tol_csr(T* x_device_iteration,T* residual_device,int  N)
{
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<N)
    {
      residual_device[i]=x_device_iteration[i]-x_device_iteration[i+N];
    }
}

namespace wtyatzoo
{
  template<typename T>
  matrix_form_cuda_jacobi_solver<T>::matrix_form_cuda_jacobi_solver(const T* val_A,const int* col_index_A,const int* row_offset_A,const int num_non_zero_A,const int N)
  {
    int i,j;
    //need to free
    T* dig_A_coefficient_inverse=(T*)malloc(sizeof(T)*N);

    this->num_non_zero_T=num_non_zero_A-N;
    this->N=N;
    //    printf("num_non_zero_T :%d \n N :%d\n num_non_zero_A :%d\n",num_non_zero_T,N,num_non_zero_A);

    // need to free!
    int * row_offset_T=(int *)malloc(sizeof(int )*(N+1));
    int * col_index_T=(int *)malloc(sizeof(int )*num_non_zero_T);
    T* val_T=(T*)malloc(sizeof(T)*num_non_zero_T);
    
    int  offset_now=0;
    int  row_now=0;
    row_offset_T[0]=0;

    for(i=0;i<N;++i)
      {
	for(j=row_offset_A[i];j<row_offset_A[i+1];++j)
	  {
	    if(col_index_A[j]==i)
	      {
		dig_A_coefficient_inverse[i]=1.0/val_A[j];
	      }
	    else if(col_index_A[j]!=i)
	      {
		val_T[offset_now]=val_A[j];
		col_index_T[offset_now]=col_index_A[j];
		offset_now++;
	      }
	  }
	row_offset_T[i+1]=offset_now;
      }
    for(i=0;i<N;++i)
      {
	for(j=row_offset_T[i];j<row_offset_T[i+1];++j)
	  {
	    val_T[j]=val_T[j]*dig_A_coefficient_inverse[i]*-1.0;
	  }
      }
    

    checkCudaErrors(cudaMalloc((void**)&val_T_device,sizeof(T)*num_non_zero_T));
    checkCudaErrors(cudaMalloc((void**)&col_index_T_device,sizeof(int )*num_non_zero_T));
    checkCudaErrors(cudaMalloc((void**)&row_offset_T_device,sizeof(int )*(N+1)));
    checkCudaErrors(cudaMalloc((void**)&dig_A_coefficient_inverse_device,sizeof(T)*N));
    
    checkCudaErrors(cudaMemcpy(val_T_device,val_T,sizeof(T)*num_non_zero_T,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(col_index_T_device,col_index_T,sizeof(int )*num_non_zero_T,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(row_offset_T_device,row_offset_T,sizeof(int )*(N+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dig_A_coefficient_inverse_device,dig_A_coefficient_inverse,sizeof(T)*N,cudaMemcpyHostToDevice));

    //cpu's used memory
    {
      free(dig_A_coefficient_inverse);
      free(row_offset_T);
      free(col_index_T);
      free(val_T);
    }
    
    my_cublas_handle=0;
    checkCudaErrors(cublasCreate(&my_cublas_handle));
    my_cusparse_handle=0;
    checkCudaErrors(cusparseCreate(& my_cusparse_handle));

    my_descr=0;
    checkCudaErrors(cusparseCreateMatDescr(& my_descr));
    cusparseSetMatType(my_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(my_descr,CUSPARSE_INDEX_BASE_ZERO);


    checkCudaErrors(cudaMalloc((void**)&b_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&x_device_iteration,sizeof(T)*2*N));
    checkCudaErrors(cudaMalloc((void**)&residual_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&dig_inverse_b_device,sizeof(T)*N));
  }

  
  // assume that the vector's i & j pair is without repetition and in order, so before this function, the user need to make the data be satisfied with the assumption.
  template<typename T>
  matrix_form_cuda_jacobi_solver<T>::matrix_form_cuda_jacobi_solver(const std::vector<triplet<T> > &mytriplets_A,const int N)
  {
    int i,j;
    vector<triplet<T> > mytriplets;
    while(!mytriplets.empty())
      {
	mytriplets.pop_back();
      }
    //need to free
    T* dig_A_coefficient_inverse=(T*)malloc(sizeof(T)*N);

    int num_non_zero_A=mytriplets_A.size();
    for(i=0;i<num_non_zero_A;++i)
      {
	triplet<T> tri_now=mytriplets_A[i];
	if(tri_now.col!=tri_now.row)
	  {
	    mytriplets.push_back(tri_now);
	  }
	else if(tri_now.col==tri_now.row)
	  {
	    dig_A_coefficient_inverse[tri_now.col]=1.0/tri_now.val;
	  }
      }

    this->num_non_zero_T=mytriplets.size();
    this->N=N;
    //  printf("num_non_zero_T :%d \n N :%d\n num_non_zero_A :%d\n",num_non_zero_T,N,num_non_zero_A);
    for(i=0;i<num_non_zero_T;++i)
      {
	int row_now=mytriplets[i].row;
	mytriplets[i].val=mytriplets[i].val*dig_A_coefficient_inverse[row_now]*-1;
      }

    // need to free!
    int * row_offset_T=(int *)malloc(sizeof(int )*(N+1));
    int * col_index_T=(int *)malloc(sizeof(int )*num_non_zero_T);
    T* val_T=(T*)malloc(sizeof(T)*num_non_zero_T);
    
    int  offset_now=0;
    int  row_now=0;
    row_offset_T[0]=0;
    for(i=0;i<num_non_zero_T;++i)
      {
	if(mytriplets[i].row==row_now)
	  {
	    row_offset_T[row_now]=offset_now;
	    row_now++; offset_now++;
	  }
	else
	  {
	    offset_now++;
	  }
	col_index_T[i]=mytriplets[i].col;
	val_T[i]=mytriplets[i].val;
      }
    row_offset_T[N]=num_non_zero_T;

    checkCudaErrors(cudaMalloc((void**)&val_T_device,sizeof(T)*num_non_zero_T));
    checkCudaErrors(cudaMalloc((void**)&col_index_T_device,sizeof(int )*num_non_zero_T));
    checkCudaErrors(cudaMalloc((void**)&row_offset_T_device,sizeof(int )*(N+1)));
    checkCudaErrors(cudaMalloc((void**)&dig_A_coefficient_inverse_device,sizeof(T)*N));
    
    checkCudaErrors(cudaMemcpy(val_T_device,val_T,sizeof(T)*num_non_zero_T,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(col_index_T_device,col_index_T,sizeof(int )*num_non_zero_T,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(row_offset_T_device,row_offset_T,sizeof(int )*(N+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dig_A_coefficient_inverse_device,dig_A_coefficient_inverse,sizeof(T)*N,cudaMemcpyHostToDevice));

    //cpu's used memory
    {
      free(dig_A_coefficient_inverse);
      free(row_offset_T);
      free(col_index_T);
      free(val_T);
    }
    
    my_cublas_handle=0;
    checkCudaErrors(cublasCreate(&my_cublas_handle));
    my_cusparse_handle=0;
    checkCudaErrors(cusparseCreate(& my_cusparse_handle));

    my_descr=0;
    checkCudaErrors(cusparseCreateMatDescr(& my_descr));
    cusparseSetMatType(my_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(my_descr,CUSPARSE_INDEX_BASE_ZERO);


    checkCudaErrors(cudaMalloc((void**)&b_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&x_device_iteration,sizeof(T)*2*N));
    checkCudaErrors(cudaMalloc((void**)&residual_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&dig_inverse_b_device,sizeof(T)*N));
  }


  template<typename T>
  matrix_form_cuda_jacobi_solver<T>::matrix_form_cuda_jacobi_solver( T**A,const T*dig_A_coefficient,const int  N)
  {
    // assemble the sparse matrix structure on GPU based on the dese matrix input
    int  i,j;

    vector<triplet<T> > mytriplets;
    while(!mytriplets.empty())
      {
	mytriplets.pop_back();
      }
    //need to free
    T* dig_A_coefficient_inverse=(T*)malloc(sizeof(T)*N);

    T EPS=1e-10;
    for(i=0;i<N;++i)
      {
        dig_A_coefficient_inverse[i] = 1.0/dig_A_coefficient[i];
	for(j=0;j<N;++j)
	  {
//	    printf("%lf\n",A[i][j]);
	    if(fabs(A[i][j])>=EPS&&(i!=j))
	      {
		mytriplets.push_back(triplet<T>(i,j,-1*A[i][j]*dig_A_coefficient_inverse[i]));
	      }
	  }
      }
    this->num_non_zero_T=mytriplets.size();
    this->N=N;
    //    printf("num_non_zero_T :%d\n N :%d\n",num_non_zero_T,N);


    // need to free!
    int * row_offset_T=(int *)malloc(sizeof(int )*(N+1));
    int * col_index_T=(int *)malloc(sizeof(int )*num_non_zero_T);
    T* val_T=(T*)malloc(sizeof(T)*num_non_zero_T);
    
    int  offset_now=0;
    int  row_now=0;
    row_offset_T[0]=0;
    for(i=0;i<num_non_zero_T;++i)
      {
	if(mytriplets[i].row==row_now)
	  {
	    row_offset_T[row_now]=offset_now;
	    row_now++; offset_now++;
	  }
	else
	  {
	    offset_now++;
	  }
	col_index_T[i]=mytriplets[i].col;
	val_T[i]=mytriplets[i].val;
      }
    row_offset_T[N]=num_non_zero_T;

    checkCudaErrors(cudaMalloc((void**)&val_T_device,sizeof(T)*num_non_zero_T));
    checkCudaErrors(cudaMalloc((void**)&col_index_T_device,sizeof(int )*num_non_zero_T));
    checkCudaErrors(cudaMalloc((void**)&row_offset_T_device,sizeof(int )*(N+1)));
    checkCudaErrors(cudaMalloc((void**)&dig_A_coefficient_inverse_device,sizeof(T)*N));
    
    checkCudaErrors(cudaMemcpy(val_T_device,val_T,sizeof(T)*num_non_zero_T,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(col_index_T_device,col_index_T,sizeof(int )*num_non_zero_T,cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(row_offset_T_device,row_offset_T,sizeof(int )*(N+1),cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(dig_A_coefficient_inverse_device,dig_A_coefficient_inverse,sizeof(T)*N,cudaMemcpyHostToDevice));

    //cpu's used memory
    {
      free(dig_A_coefficient_inverse);
      free(row_offset_T);
      free(col_index_T);
      free(val_T);
    }
    
    my_cublas_handle=0;
    checkCudaErrors(cublasCreate(&my_cublas_handle));
    my_cusparse_handle=0;
    checkCudaErrors(cusparseCreate(& my_cusparse_handle));

    my_descr=0;
    checkCudaErrors(cusparseCreateMatDescr(& my_descr));
    cusparseSetMatType(my_descr,CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(my_descr,CUSPARSE_INDEX_BASE_ZERO);


    checkCudaErrors(cudaMalloc((void**)&b_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&x_device_iteration,sizeof(T)*2*N));
    checkCudaErrors(cudaMalloc((void**)&residual_device,sizeof(T)*N));
    checkCudaErrors(cudaMalloc((void**)&dig_inverse_b_device,sizeof(T)*N));
  }

  template<typename T>
  matrix_form_cuda_jacobi_solver<T>::~matrix_form_cuda_jacobi_solver()
  {
    //  printf("GG\n");
    checkCudaErrors(cudaFree(dig_A_coefficient_inverse_device));
    checkCudaErrors(cudaFree(val_T_device));
    checkCudaErrors(cudaFree(col_index_T_device));
    checkCudaErrors(cudaFree(row_offset_T_device));
    checkCudaErrors(cudaFree(b_device));
    checkCudaErrors(cudaFree(dig_inverse_b_device));
    checkCudaErrors(cudaFree(x_device_iteration));
    checkCudaErrors(cudaFree(residual_device));

    // cudaFree(dig_A_coefficient_inverse_device);
    // cudaFree(val_T_device);
    // cudaFree(col_index_T_device);
    // cudaFree(row_offset_T_device);
    // cudaFree(b_device);
    // cudaFree(dig_inverse_b_device);
    // cudaFree(x_device_iteration);
    // cudaFree(residual_device);
    
  }

  template<typename T>
  int matrix_form_cuda_jacobi_solver<T>::apply(const T b[],T x[],const T tol,const int  max_iteration_num,const bool given)
  {
    int block_size=1024;
    int num_block=(int)ceil(N/(T)block_size);

    int  iteration_num_now=0;

    //  printf("apply\n");
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
    
    set_Dinverseb<<< num_block,block_size >>>(dig_A_coefficient_inverse_device,b_device,dig_inverse_b_device,N); //no cuBlas API support
    int new_iteration=1,old_iteration;
    residual_sum_host=0;
    const T alpha=1.0, beta=0;

    T test_f_d;
    const char* p_char=typeid(test_f_d).name();
    if((*p_char)=='d')
      {
	for(iteration_num_now=0;iteration_num_now<max_iteration_num;iteration_num_now++)
	  {
	    old_iteration=!new_iteration; 	
	    cusparseDcsrmv(my_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,N,N, num_non_zero_T,(const double*)&alpha,my_descr,(const double*)val_T_device,row_offset_T_device,col_index_T_device,(const double*)x_device_iteration+(old_iteration*N),(const double*)&beta,(double*)x_device_iteration+(new_iteration*N));
	    cublasDaxpy(my_cublas_handle,N,(const double*)&alpha,(const double*)dig_inverse_b_device,1,(double*)x_device_iteration+(new_iteration*N),1);
	    get_tol_csr<<<num_block,block_size>>>(x_device_iteration,residual_device,N);
	    cublasDnrm2(my_cublas_handle,N,(const double*)residual_device,1,(double*)&residual_sum_host);

	    //  printf("residual_sum_host :: %lf\n",residual_sum_host);
	    if(pow(residual_sum_host,2)<tol)
	      {
		break;
	      }
	    new_iteration=!new_iteration;
	  }
      }
    else if((*p_char)=='f')
      {
	for(iteration_num_now=0;iteration_num_now<max_iteration_num;iteration_num_now++)
	  {
	    old_iteration=!new_iteration; 	
	    cusparseScsrmv(my_cusparse_handle,CUSPARSE_OPERATION_NON_TRANSPOSE,N,N, num_non_zero_T,(const float*)&alpha,my_descr,(const float*)val_T_device,row_offset_T_device,col_index_T_device,(const float*)x_device_iteration+(old_iteration*N),(const float*)&beta,(float*)x_device_iteration+(new_iteration*N));
	    cublasSaxpy(my_cublas_handle,N,(const float*)&alpha,(const float*)dig_inverse_b_device,1,(float*)x_device_iteration+(new_iteration*N),1);
	    get_tol_csr<<<num_block,block_size>>>(x_device_iteration,residual_device,N);
	    cublasSnrm2(my_cublas_handle,N,(const float*)residual_device,1,(float*)&residual_sum_host);

	    //	printf("residual_sum_host :: %lf\n",residual_sum_host);
	    if(pow(residual_sum_host,2)<tol)
	      {
		break;
	      }
	    new_iteration=!new_iteration;
	  }
      }
    
    printf("iteration_num_now ::%d\n",iteration_num_now);
    checkCudaErrors(cudaMemcpy(x,x_device_iteration, N*sizeof(T), cudaMemcpyDeviceToHost));
    return 0;
  }
}
