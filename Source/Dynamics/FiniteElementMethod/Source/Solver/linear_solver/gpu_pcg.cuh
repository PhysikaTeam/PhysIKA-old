/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: GPU-based precondition conjugate gradient method
 * @version    : 1.0
 */
#ifndef GPU_PCG_CUH_PhysIKA
#define GPU_PCG_CUH_PhysIKA
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

#include "linear_solver.h"
#define N_MAX 400000
#define NNZ_MAX 15248000
namespace PhysIKA {
/**
 * constructor type enumeration.
 *
 */
enum class constuctor_type
{
    HostArray   = cudaMemcpyHostToDevice,
    DeviceArray = cudaMemcpyDeviceToDevice
};

// #ifndef SPM
// template<typename T>
// using SPM = Eigen::SparseMatrix<T, Eigen::RowMajor>;
// #endif
/**
 * GPU-based preconditioner conjugate gradient class.
 *
 */
template <typename T>
class GPU_PCG
{
private:
    T*             A;
    T*             B;
    int            N, NNZ, *csr;
    cudaEvent_t    start, stop;
    T *            dev_Minverse, *dev_r, *dev_z, *dev_p;
    T *            zr, *dev_zr, *dev_ap, *pap, *dev_pap, *ak, *dev_ak, *x, *dev_x, *zrnew, *dev_zrnew;
    int *          pos_x, *pos_y;
    dim3           block;
    dim3           grid;
    cublasStatus_t status;
    cublasHandle_t handle;
    int*           IA;
    int*           JA;
    void           initialize(int Ntemp, int NNZtemp, T* Atemp, T* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE);

public:
    GPU_PCG() = default;
    GPU_PCG(int Ntemp, int NNZtemp, thrust::pair<int, int>* coo, T* hes_val, T* right_hand, constuctor_type TYPE);
    GPU_PCG(int Ntemp, int NNZtemp, T* Atemp, T* Btemp, int* IAtemp, int* JAtemp, constuctor_type TYPE);
    GPU_PCG(int Ntemp, int NNZtemp, const T* Atemp, const T* Btemp, const int* IAtemp, const int* JAtemp, constuctor_type TYPE);
    // T* solve_pcg(constuctor_type TYPE);
    int  solve_pcg(constuctor_type TYPE, T* x);
    void update_hes(T* Atemp, T* Btemp, constuctor_type TYPE);
    void readIAandJA(int size_Matrix, int size_nozeronumber, int* IAtemp, int* JAtemp, constuctor_type TYPE);
    ~GPU_PCG();
};

/**
 * cuda-impl conjugate gradient method class.
 *
 */
template <typename T>
class CUDA_PCG : public unconstrainted_linear_solver<T>
{
private:
    bool hes_is_constant_;

public:
    using VEC = Eigen::Matrix<T, -1, 1>;
    CUDA_PCG(const bool hes_is_constant)
        : hes_is_constant_(hes_is_constant) {}
    int solve(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A, const T* b, const Eigen::SparseMatrix<T, Eigen::RowMajor>& J, const T* c, T* solution) const;
};
}  // namespace PhysIKA

#endif
