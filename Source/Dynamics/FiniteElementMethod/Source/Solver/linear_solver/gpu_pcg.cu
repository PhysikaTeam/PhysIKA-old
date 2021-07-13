/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: GPU-based precondition conjugate gradient method
 * @version    : 1.0
 */
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusparse.h"
#include <fstream>
#include <assert.h>
#include <cmath>
#include <vector>
#include <cublas_v2.h>
#include <thrust/sort.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/Eigen>
#include "gpu_pcg.cuh"

#include "Common/error.h"
namespace PhysIKA {
//#define RECORD_TIME
#define RECORD_RESIDUAL
template <typename T>
__global__ void initialvalue(int N, T* A, T* B, T* Minverse, T* r, T* z, T* p, int* IA, int* JA)
{
    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int tid     = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

    while (tid < N)
    {
        int jtmp = IA[tid + 1] - IA[tid];
        for (int j = 0; j < jtmp; j++)
        {
            if (JA[j + IA[tid]] == tid)
            {
                Minverse[tid] = 1.0 / A[j + IA[tid]];
            }
        }
        r[tid] = B[tid];
        z[tid] = Minverse[tid] * r[tid];
        p[tid] = z[tid];
        tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
    }
}

template <typename T>
__global__ void VectorAMUtiplyP(int N, T* A, T* p, T* ap, int* IA, int* JA)
{

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int tid     = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    while (tid < N)
    {
        T   temp = 0;
        int jtemp;
        jtemp = IA[tid + 1] - IA[tid];
        for (int j = 0; j < jtemp; j++)
        {
            temp += A[j + IA[tid]] * p[JA[j + IA[tid]]];
        }
        ap[tid] = temp;
        tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
    }
}

template <typename T>
__global__ void inerate_ak(T* zr, T* pap, T* ak)
{
    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        *ak = (*zr) / (*pap);
    }
}

template <typename T>
__global__ void inerate_x(int N, T* p, T* ak, T* x)
{

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int tid     = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    while (tid < N)
    {
        x[tid] = x[tid] + (*ak) * p[tid];
        tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
    }
}

template <typename T>
__global__ void inerate_r(int N, T* ak, T* ap, T* r)
{

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int tid     = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    while (tid < N)
    {
        r[tid] = r[tid] - (*ak) * ap[tid];
        tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
    }
}

template <typename T>
__global__ void inerate_z(int N, T* Minverse, T* r, T* z)
{

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int tid     = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    while (tid < N)
    {
        z[tid] = Minverse[tid] * r[tid];
        tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
    }
}

template <typename T>
__global__ void inerate_p(int N, T* zrnew, T* zr, T* z, T* p)
{

    int blockId = blockIdx.y * gridDim.x + blockIdx.x;
    int tid     = blockId * (blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;
    while (tid < N)
    {
        p[tid] = z[tid] + ((*zrnew) / (*zr)) * p[tid];
        tid += (gridDim.x * blockDim.x) * (gridDim.y * blockDim.y);
    }
}

template <typename T>
__global__ void decouple_pos(thrust::pair<int, int>* pos, int* pos_x, int* pos_y, T* value)
{
    int index    = threadIdx.x;
    pos_x[index] = pos[index].first;
    pos_y[index] = pos[index].second;
}

template <typename T>
void GPU_PCG<T>::readIAandJA(int size_Matrix, int size_nozeronumber, int* IAtemp, int* JAtemp, constuctor_type TYPE)
{
    cudaMalloc(( void** )&IA, sizeof(int) * (size_Matrix + 1));
    cudaMalloc(( void** )&JA, sizeof(int) * size_nozeronumber);
    cudaMemcpy(IA, IAtemp, sizeof(int) * (size_Matrix + 1), ( cudaMemcpyKind )TYPE);
    cudaMemcpy(JA, JAtemp, sizeof(int) * size_nozeronumber, ( cudaMemcpyKind )TYPE);
}

template <typename T>
GPU_PCG<T>::GPU_PCG(int Ntemp, int NNZtemp, thrust::pair<int, int>* coo, T* hes_val, T* right_hand, constuctor_type TYPE)
{

    cudaMalloc(( void** )&csr, sizeof(int) * (Ntemp + 1));
    cudaMalloc(( void** )&pos_y, sizeof(int) * NNZtemp);
    cudaMalloc(( void** )&pos_x, sizeof(int) * NNZtemp);
    thrust::device_ptr<T>                      dev_data_ptr(hes_val);
    thrust::device_ptr<thrust::pair<int, int>> dev_keys_ptr(coo);
    thrust::sort_by_key(dev_keys_ptr, dev_keys_ptr + NNZtemp, dev_data_ptr);
    decouple_pos<<<1, NNZtemp>>>(coo, pos_x, pos_y, hes_val);
    cusparseHandle_t handle;
    cusparseStatus_t status = cusparseCreate(&handle);
    cusparseXcoo2csr(handle, pos_x, NNZtemp, N, csr, CUSPARSE_INDEX_BASE_ZERO);
    initialize(Ntemp, NNZtemp, hes_val, right_hand, csr, pos_y, TYPE);
}

#define init_array_on_device(var, num, type)        \
    cudaMalloc(( void** )&var, sizeof(type) * num); \
    cudaMemset(var, 0, num * sizeof(type));

template <typename T>
void GPU_PCG<T>::initialize(int Ntemp, int NNZtemp, T* Atemp, T* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE)
{  //IAtemp 最后一项必须为总非零点数目
    readIAandJA(Ntemp, NNZtemp, IAtemp, JAtemp, TYPE);
    N   = Ntemp;
    NNZ = NNZtemp;
    assert(Ntemp <= N_MAX);
    assert(NNZtemp <= NNZ_MAX);
#ifdef RECORD_TIME
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    block = dim3(32, 32);
    grid  = dim3((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    zr    = new T;
    pap   = new T;
    ak    = new T;
    x     = new T[N];
    zrnew = new T;

    init_array_on_device(A, NNZ, T);
    init_array_on_device(B, N, T);
    init_array_on_device(dev_Minverse, N, T);
    init_array_on_device(dev_r, N, T);
    init_array_on_device(dev_z, N, T);
    init_array_on_device(dev_p, N, T);
    init_array_on_device(dev_zr, N, T);
    init_array_on_device(dev_ap, N, T);
    init_array_on_device(dev_pap, N, T);
    init_array_on_device(dev_ak, N, T);
    init_array_on_device(dev_x, N, T);
    init_array_on_device(dev_zrnew, N, T);
    // cudaMalloc((void**)& A, sizeof(T) * NNZ);
    // cudaMalloc((void**)& B, sizeof(T) * N);
    // cudaMalloc((void**)& dev_Minverse, sizeof(T) * N);
    // cudaMalloc((void**)& dev_r, sizeof(T) * N);
    // cudaMalloc((void**)& dev_z, sizeof(T) * N);
    // cudaMalloc((void**)& dev_p, sizeof(T) * N);
    // cudaMalloc((void**)& dev_zr, sizeof(T));
    // cudaMalloc((void**)& dev_ap, sizeof(T) * N);
    // cudaMalloc((void**)& dev_pap, sizeof(T));
    // cudaMalloc((void**)& dev_ak, sizeof(T));
    // cudaMalloc((void**)& dev_x, sizeof(T) * N);
    // cudaMalloc((void**)& dev_zrnew, sizeof(T));

    cudaMemcpy(A, Atemp, sizeof(T) * NNZtemp, ( cudaMemcpyKind )TYPE);
    cudaMemcpy(B, Btemp, sizeof(T) * N, ( cudaMemcpyKind )TYPE);

    initialvalue<<<grid, block>>>(N, A, B, dev_Minverse, dev_r, dev_z, dev_p, IA, JA);

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS ERROR" << std::endl;
        getchar();
    }
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
}

template <typename T>
GPU_PCG<T>::GPU_PCG(int Ntemp, int NNZtemp, T* Atemp, T* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE)
{  //IAtemp 最后一项必须为总非零点数目
    initialize(Ntemp, NNZtemp, Atemp, Btemp, IAtemp, JAtemp, TYPE);
}
template <typename T>
GPU_PCG<T>::GPU_PCG(int Ntemp, int NNZtemp, const T* Atemp, const T* Btemp, const int* IAtemp, const int* JAtemp, constuctor_type TYPE)
{  //IAtemp 最后一项必须为总非零点数目
    initialize(Ntemp, NNZtemp, ( T* )Atemp, ( T* )Btemp, ( int* )IAtemp, ( int* )JAtemp, TYPE);
}

template <typename T>
void GPU_PCG<T>::update_hes(T* Atemp, T* Btemp, constuctor_type TYPE)
{
#ifdef RECORD_TIME
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
#endif
    dim3 block(32, 32);
    dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    zr    = new T;
    pap   = new T;
    ak    = new T;
    x     = new T[N];
    zrnew = new T;

    cudaMalloc(( void** )&A, sizeof(T) * NNZ);
    cudaMalloc(( void** )&B, sizeof(T) * N);
    cudaMalloc(( void** )&dev_Minverse, sizeof(T) * N);
    cudaMalloc(( void** )&dev_r, sizeof(T) * N);
    cudaMalloc(( void** )&dev_z, sizeof(T) * N);
    cudaMalloc(( void** )&dev_p, sizeof(T) * N);
    cudaMalloc(( void** )&dev_zr, sizeof(T));
    cudaMalloc(( void** )&dev_ap, sizeof(T) * N);
    cudaMalloc(( void** )&dev_pap, sizeof(T));
    cudaMalloc(( void** )&dev_ak, sizeof(T));
    cudaMalloc(( void** )&dev_x, sizeof(T) * N);
    cudaMalloc(( void** )&dev_zrnew, sizeof(T));

    cudaMemcpy(A, Atemp, sizeof(T) * NNZ, ( cudaMemcpyKind )TYPE);
    cudaMemcpy(B, Btemp, sizeof(T) * N, ( cudaMemcpyKind )TYPE);

    initialvalue<<<grid, block>>>(N, A, B, dev_Minverse, dev_r, dev_z, dev_p, IA, JA);

    status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cout << "CUBLAS ERROR" << std::endl;
        getchar();
    }
}
template <typename T>
GPU_PCG<T>::~GPU_PCG()
{
    cudaFree(A);
    cudaFree(B);
    cudaFree(dev_Minverse);
    cudaFree(dev_r);
    cudaFree(dev_z);
    cudaFree(dev_p);
    cudaFree(dev_zr);
    cudaFree(dev_ap);
    cudaFree(dev_pap);
    cudaFree(dev_ak);
    cudaFree(dev_x);
    cudaFree(dev_zrnew);
}

// template<typename T>
// T* PCG<T>::solve_pcg(constuctor_type TYPE) {

// }

template <>
int GPU_PCG<double>::solve_pcg(constuctor_type TYPE, double* x)
{
    for (int i = 0; i < N; i++)
    {
        cublasDdot(handle, N, dev_z, 1, dev_r, 1, dev_zr);
        VectorAMUtiplyP<double><<<grid, block>>>(N, A, dev_p, dev_ap, IA, JA);
        cublasDdot(handle, N, dev_ap, 1, dev_p, 1, dev_pap);
        inerate_ak<double><<<grid, block>>>(dev_zr, dev_pap, dev_ak);
        inerate_x<double><<<grid, block>>>(N, dev_p, dev_ak, dev_x);
        inerate_r<double><<<grid, block>>>(N, dev_ak, dev_ap, dev_r);
        inerate_z<double><<<grid, block>>>(N, dev_Minverse, dev_r, dev_z);
        cublasDdot(handle, N, dev_z, 1, dev_r, 1, dev_zrnew);
        cudaMemcpy(zrnew, dev_zrnew, sizeof(double), cudaMemcpyDeviceToHost);
        if (sqrt(*zrnew) < 1.0e-8)
            break;
        inerate_p<double><<<grid, block>>>(N, dev_zrnew, dev_zr, dev_z, dev_p);
    }
#ifdef RECORD_TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << time;
#endif
    // if (TYPE==constuctor_type::DeviceArray)
    //   return 0;
    // double* x = new double[N];
    cudaMemcpy(x, dev_x, sizeof(double) * N, cudaMemcpyDeviceToHost);
    return 0;
}

template <>
int GPU_PCG<float>::solve_pcg(constuctor_type TYPE, float* x)
{
    for (int i = 0; i < N; i++)
    {
        cublasSdot(handle, N, dev_z, 1, dev_r, 1, dev_zr);
        VectorAMUtiplyP<float><<<grid, block>>>(N, A, dev_p, dev_ap, IA, JA);
        cublasSdot(handle, N, dev_ap, 1, dev_p, 1, dev_pap);
        inerate_ak<float><<<grid, block>>>(dev_zr, dev_pap, dev_ak);
        inerate_x<float><<<grid, block>>>(N, dev_p, dev_ak, dev_x);
        inerate_r<float><<<grid, block>>>(N, dev_ak, dev_ap, dev_r);
        inerate_z<float><<<grid, block>>>(N, dev_Minverse, dev_r, dev_z);
        cublasSdot(handle, N, dev_z, 1, dev_r, 1, dev_zrnew);
        cudaMemcpy(zrnew, dev_zrnew, sizeof(float), cudaMemcpyDeviceToHost);
        if (sqrt(*zrnew) < 1.0e-8)
            break;
        inerate_p<float><<<grid, block>>>(N, dev_zrnew, dev_zr, dev_z, dev_p);
    }
#ifdef RECORD_TIME
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float time;
    cudaEventElapsedTime(&time, start, stop);
    std::cout << time;
#endif
    // if (TYPE==constuctor_type::DeviceArray)
    //   return dev_x;

    cudaMemcpy(x, dev_x, sizeof(float) * N, cudaMemcpyDeviceToHost);
    return 0;
}

__global__ void print(int* a)
{
    if (threadIdx.x == 0)
        printf("%d", *a);
}

template <typename T>
int CUDA_PCG<T>::solve(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A, const T* b, const Eigen::SparseMatrix<T, Eigen::RowMajor>& J, const T* c, T* solution) const
{
    using namespace Eigen;
    using namespace std;
    using VEC = Eigen::Matrix<T, -1, 1>;

    //  cout<<A*A<<endl;
    printf("\nStarted CUDA_PCG.\n");
    T*          value = const_cast<T*>(A.valuePtr());
    int*        inner = const_cast<int*>(A.innerIndexPtr());
    int*        outer = const_cast<int*>(A.outerIndexPtr());
    T*          B     = const_cast<T*>(b);
    int         NNZ   = A.nonZeros();
    int         N     = A.rows();
    GPU_PCG<T>* pcg;
    pcg = new GPU_PCG<T>(N, NNZ, value, B, outer, inner, constuctor_type::HostArray);
    IF_ERR(return, pcg->solve_pcg(constuctor_type::HostArray, solution));

#ifdef RECORD_RESIDUAL
    const Map<const VEC> rhs(b, N);
    const Map<const VEC> X(solution, N);
    cout << "Norm of residual : " << (A * X - rhs).norm() << endl;
#endif

    printf("\nFinished CUDA_PCG.\n");

    return 0;
}
template class CUDA_PCG<double>;
template class CUDA_PCG<float>;
}  // namespace PhysIKA
