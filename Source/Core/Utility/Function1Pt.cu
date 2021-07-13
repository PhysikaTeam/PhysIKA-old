#include "Function1Pt.h"
#include "Core/Utility.h"
namespace PhysIKA {
namespace Function1Pt {
template <typename T1, typename T2>
__global__ void KerLength(T1* lhs, T2* rhs, int num)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= num)
        return;

    lhs[pId] = length(rhs[pId]);
}

template <typename T1, typename T2>
void Length(DeviceArray<T1>& lhs, DeviceArray<T2>& rhs)
{
    assert(lhs.size() == rhs.size());
    unsigned pDim = cudaGridSize(rhs.size(), BLOCK_SIZE);
    KerLength<<<pDim, BLOCK_SIZE>>>(lhs.begin(), rhs.begin(), lhs.size());
}

template void Length(DeviceArray<float>&, DeviceArray<float3>&);
}  // namespace Function1Pt
}  // namespace PhysIKA