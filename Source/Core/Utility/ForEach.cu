#include "ForEach.h"
#include <cuda_runtime.h>

namespace PhysIKA {
template <typename Operation>
__global__ void kernel_foreach(Operation o)
{
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    o(i);
}

void ForEach(size_t size, ...)
{
}
}  // namespace PhysIKA
