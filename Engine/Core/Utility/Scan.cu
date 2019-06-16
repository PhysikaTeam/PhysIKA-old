#include "Scan.h"
#include "cuda_runtime.h"
#include "Core/Utility.h"

Scan::Scan()
{
	cudaMalloc(&m_sum, sizeof(int));
}

Scan::~Scan()
{
	cudaFree(m_sum);
}



void Scan::allocateBuffer(int size)
{
	m_size = size;
	cudaMalloc(&m_buffer, size * sizeof(int));
}

Scan* Scan::create(int buf_size, bool inOrder)
{
	Scan* scan = new Scan();
	if (inOrder)
		scan->allocateBuffer(buf_size);

	return scan;
}


__global__
void S_ExclusiveScan(int *g_odata, int *g_idata, int n, int *SUM,
	int add_last) {
	// shared memory init
	__shared__ int temp[SCAN_BLOCK_SIZE << 1];

	// local variables for the later usage to improve the performance
	int thid = threadIdx.x;
	int blockId = blockDim.x * blockIdx.x << 1;
	int offset = 0;
	int last = 0;

	// load the elements from global memory into the shared memory
	if (blockId + (thid << 1) < n)
		temp[thid << 1] = g_idata[blockId + (thid << 1)];
	if (blockId + (thid << 1) + 1 < n)
		temp[(thid << 1) + 1] = g_idata[blockId + (thid << 1) + 1];

	// save the last element for later to improve the performance
// 	if (add_last && thid == SCAN_BLOCK_SIZE - 1)
// 		last = temp[(thid << 1) + 1];

	// build sum in place up the tree (reduction phase)
	for (int d = SCAN_BLOCK_SIZE; d > 0; d >>= 1) {
		__syncthreads();
		if (thid < d) {
			int ai = (((thid << 1) + 1) << offset) - 1;
			int bi = (((thid << 1) + 2) << offset) - 1;
			temp[bi] += temp[ai];
		}
		offset++;
	}

	// clear the last element
	if (thid == 0)
		temp[(SCAN_BLOCK_SIZE << 1) - 1] = 0;

	// traverse down tree & build scan (distribution phase)
	for (int d = 1; d < (SCAN_BLOCK_SIZE << 1); d <<= 1) {
		offset--;
		__syncthreads();
		if (thid < d) {
			int ai = (((thid << 1) + 1) << offset) - 1;
			int bi = (((thid << 1) + 2) << offset) - 1;
			int t = temp[ai];
			temp[ai] = temp[bi];
			temp[bi] += t;
		}
	}

	__syncthreads();
// 	// extract the sum (merged to improve the performance)
// 	if (add_last && thid == SCAN_BLOCK_SIZE - 1)
// 		SUM[blockIdx.x] = temp[(thid << 1) + 1] + last;

	// update the output vector by loading shared memory into the global memory
	if (blockId + (thid << 1) < n)
		g_odata[blockId + (thid << 1)] = temp[thid << 1];
	if (blockId + (thid << 1) + 1 < n)
		g_odata[blockId + (thid << 1) + 1] = temp[(thid << 1) + 1];
}


int Scan::ExclusiveScan(int* dst, int* src, int size)
{
	int dim = 1 + ((size - 1) / (SCAN_BLOCK_SIZE * 2));

	S_ExclusiveScan << <dim, SCAN_BLOCK_SIZE >> > (dst, src, size, NULL, 0);
	cuSynchronize();
// 	int ret;
// 	cudaMemcpy(&ret, m_sum, sizeof(int), cudaMemcpyDeviceToHost);

	return 0;
}

int Scan::ExclusiveScan(int* src, int size)
{
	cudaMemcpy(m_buffer, src, size * sizeof(int), cudaMemcpyDeviceToDevice);

	return ExclusiveScan(src, m_buffer, size);
}
