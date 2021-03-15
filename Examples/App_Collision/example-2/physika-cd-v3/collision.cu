//atomicAdd(XX, 1) -> atomicInc !!!!!!!
#define OUTPUT_TXT

// CUDA Runtime
#include <cuda_runtime.h>

#include <cuda_profiler_api.h>
#include <assert.h>

#include <string>
using namespace std;

typedef unsigned int uint;

#define REAL_infinity 1.0e30

// Utilities and system includes
#include <helper_functions.h>  // helper for shared functions common to CUDA SDK samples
#include <helper_cuda.h>       // helper for CUDA error checking

#include "vec3.cuh"
#include "tools.cuh"
#include "box.cuh"
#include "tri3f.cuh"
#include "bvh.cuh"
#include "pair.cuh"
#include "tri-contact.cuh"
#include "mesh.cuh"

#include <math.h>
#include <stdarg.h>

#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include "forceline.h"


//=======================================================

cudaDeviceProp deviceProp;
extern void initPairsGPU();

void initGPU()
{
	static int devID = 0;

	if (devID == 0) {
		cudaGetDevice(&devID);
		checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));

		initPairsGPU();
	}
}

//=======================================================

g_mesh theCloth;
g_bvh theBVH[2];
g_front theFront[2];
g_pair thePairs[2]; // potentially colliding pairs
g_pair retPairs; //results
g_pairCCD retPairsCCD;

//=======================================================

void initPairsGPU()
{
	//pairs[0].init(MAX_PAIR_NUM); // MAX_PAIR_NUM);
	thePairs[1].init(MAX_PAIR_NUM);
	retPairs.init(MAX_PAIR_NUM);

	retPairsCCD.init(MAX_PAIR_NUM / 10);
}

void pushMesh2GPU(int  numFace, int numVert, void *faces, void *nodes)
{
	theCloth.init();

	theCloth.numFace = numFace;
	theCloth.numVert = numVert;

	cudaMalloc((void **)&theCloth._df, numFace*sizeof(tri3f));
	cudaMalloc((void **)&theCloth._dfBx, numFace*sizeof(g_box));
	cudaMalloc((void **)&theCloth._dx, numVert*sizeof(REAL3));
	cudaMalloc((void **)&theCloth._dx0, numVert*sizeof(REAL3));

	cudaMemcpy(theCloth._df, faces, sizeof(tri3f)*numFace, cudaMemcpyHostToDevice);
	cudaMemcpy(theCloth._dx, nodes, sizeof(REAL3)*numVert, cudaMemcpyHostToDevice);
	cudaMemcpy(theCloth._dx0, theCloth._dx, sizeof(REAL3)*numVert, cudaMemcpyDeviceToDevice);

	theCloth.computeWSdata(0, false);
}

void updateMesh2GPU(void *nodes,void *prenodes,REAL thickness)
{
	//cudaMemcpy(theCloth._dx0, theCloth._dx, sizeof(REAL3)*theCloth.numVert, cudaMemcpyDeviceToDevice);
	cudaMemcpy(theCloth._dx, nodes, sizeof(REAL3)*theCloth.numVert, cudaMemcpyHostToDevice);
	cudaMemcpy(theCloth._dx0, prenodes, sizeof(REAL3)*theCloth.numVert, cudaMemcpyHostToDevice);
	theCloth.computeWSdata(thickness, true);

	REAL3* dx = new REAL3[theCloth.numVert];
	REAL3* dx0 = new REAL3[theCloth.numVert];

	cudaMemcpy(dx, theCloth._dx, sizeof(REAL3)*theCloth.numVert, cudaMemcpyDeviceToHost);
	cudaMemcpy(dx0, theCloth._dx0, sizeof(REAL3)*theCloth.numVert, cudaMemcpyDeviceToHost);

	vector<REAL3> tem;
	for (int i = 0; i < theCloth.numVert; i++)
	{
		tem.push_back(dx[i]);
		tem.push_back(dx0[i]);
	}

	int ss = 0;
	ss++;
}

//=======================================================

void pushBVHIdx(int max_level, unsigned int *level_idx, bool isCloth)
{
	theBVH[isCloth]._max_level = max_level;
	theBVH[isCloth]._level_idx = new uint[max_level];
	memcpy(theBVH[isCloth]._level_idx, level_idx, sizeof(uint)*max_level);
}

void pushBVH(unsigned int length, int *ids, bool isCloth)
{
	theBVH[isCloth]._num = length;
	checkCudaErrors(cudaMalloc((void**)&theBVH[isCloth]._bvh, length*sizeof(int) * 2));
	checkCudaErrors(cudaMemcpy(theBVH[isCloth]._bvh, ids, length*sizeof(int) * 2, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMalloc((void**)&theBVH[isCloth]._bxs, length*sizeof(g_box)));
	checkCudaErrors(cudaMemset(theBVH[isCloth]._bxs, 0, length*sizeof(g_box)));
	theBVH[isCloth].hBxs = NULL;

	theBVH[isCloth]._triBxs = isCloth ? theCloth._dfBx : NULL;
	theBVH[isCloth]._triCones = NULL;
}

void pushBVHLeaf(unsigned int length, int *idf, bool isCloth)
{
	checkCudaErrors(cudaMalloc((void**)&theBVH[isCloth]._bvh_leaf, length*sizeof(int)));
	checkCudaErrors(cudaMemcpy(theBVH[isCloth]._bvh_leaf, idf, length*sizeof(int), cudaMemcpyHostToDevice));
}

//======================================================


void refitBVH_Serial(bool isCloth, int length)
{
	refit_serial_kernel << <1, 1, 0 >> >
		(theBVH[isCloth]._bvh, theBVH[isCloth]._bxs, theBVH[isCloth]._triBxs,
		theBVH[isCloth]._cones, theBVH[isCloth]._triCones,
		length == 0 ? theBVH[isCloth]._num : length);

	getLastCudaError("refit_serial_kernel");
	cudaThreadSynchronize();
}

void refitBVH_Parallel(bool isCloth, int st, int length)
{
	BLK_PAR(length);

	refit_kernel << < B, T >> >
		(theBVH[isCloth]._bvh, theBVH[isCloth]._bxs, theBVH[isCloth]._triBxs,
		theBVH[isCloth]._cones, theBVH[isCloth]._triCones,
		st, length);

	getLastCudaError("refit_kernel");
	cudaThreadSynchronize();
}

void refitBVH(bool isCloth)
{
	// before refit, need to get _tri_boxes !!!!
	// copying !!!
	for (int i = theBVH[isCloth]._max_level - 1; i >= 0; i--) {
		int st = theBVH[isCloth]._level_idx[i];
		int ed = (i != theBVH[isCloth]._max_level - 1) ?
			theBVH[isCloth]._level_idx[i + 1] - 1 : theBVH[isCloth]._num - 1;

		int length = ed - st + 1;
		if (i < 5) {
			refitBVH_Serial(isCloth, length + st);
			break;
		}
		else
		{
			refitBVH_Parallel(isCloth, st, length);
		}
	}
}

//===============================================

void pushFront(bool self, int num, unsigned int *data)
{
	g_front *f = &theFront[self];

	f->init();
	f->push(num, (uint4 *)data);
}

//===============================================
// show memory usage of GPU
void  reportMemory(char *tag)
{
	//return;

#ifdef OUTPUT_TXT
	size_t free_byte;
	size_t total_byte;
	cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);

	if (cudaSuccess != cuda_status) {
		printf("Error: cudaMemGetInfo fails, %s \n", cudaGetErrorString(cuda_status));
		exit(1);
	}

	REAL free_db = (REAL)free_byte;
	REAL total_db = (REAL)total_byte;
	REAL used_db = total_db - free_db;
	printf("%s: GPU memory usage: used = %f, free = %f MB, total = %f MB\n",
		tag, used_db / 1024.0 / 1024.0, free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
#endif
}

//===============================================

#define STACK_SIZE 50
#define EMPTY (nIdx == 0)

#define PUSH_PAIR(nd1, nd2)  {\
	nStack[nIdx].x = nd1;\
	nStack[nIdx].y = nd2;\
	nIdx++;\
}

#define POP_PAIR(nd1, nd2) {\
	nIdx--;\
	nd1 = nStack[nIdx].x;\
	nd2 = nStack[nIdx].y;\
}

#define NEXT(n1, n2) 	POP_PAIR(n1, n2)


inline __device__ void pushToFront(int a, int b, uint4 *front, uint *idx, uint ptr)
{
	//	(*idx)++;
	if (*idx < MAX_FRONT_NUM)
	{
		uint offset = atomicAdd(idx, 1);
		front[offset] = make_uint4(a, b, 0, ptr);
	}
}

inline __device__ void sproutingAdaptive(int left, int right,
	int *bvhA, g_box *bxsA, int *bvhB, g_box *bxsB,
	uint4 *front, uint *frontIdx,
	uint2 *pairs, uint *pairIdx, bool update, uint ptr)
{
	uint2 nStack[STACK_SIZE];
	uint nIdx = 0;

	for (int i = 0; i<4; i++)
	{
		if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
			pushToFront(left, right, front, frontIdx, ptr);
		}
		else {
			if (!overlaps(left, right, bxsA, bxsB)) {
				pushToFront(left, right, front, frontIdx, ptr);
			}
			else {
				if (isLeaf(left, bvhA)) {
					PUSH_PAIR(left, getLeftChild(right, bvhB));
					PUSH_PAIR(left, getRightChild(right, bvhB));
				}
				else {
					PUSH_PAIR(getLeftChild(left, bvhA), right);
					PUSH_PAIR(getRightChild(left, bvhA), right);
				}
			}
		}

		if (EMPTY)
			return;

		NEXT(left, right);
	}

	while (!EMPTY) {
		NEXT(left, right);
		pushToFront(left, right, front, frontIdx, ptr);
	}
}

inline __device__ void sprouting(int left, int right,
	int *bvhA, g_box *bxsA, int *bvhB, g_box *bxsB,
	uint4 *front, uint *frontIdx,
	int2 *pairs, uint *pairIdx, bool update, uint ptr, tri3f *Atris)
{
	uint2 nStack[STACK_SIZE];
	uint nIdx = 0;

	while (1)
	{
		if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
			if (update)
				pushToFront(left, right, front, frontIdx, ptr);

			if (overlaps(left, right, bxsA, bxsB))
			{
				if(!covertex(getTriID(left, bvhA), getTriID(right, bvhB), Atris))
					addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
			}
				
		}
		else {
			if (!overlaps(left, right, bxsA, bxsB)) {
				if (update)
					pushToFront(left, right, front, frontIdx, ptr);

			}
			else {
				if (isLeaf(left, bvhA)) {
					PUSH_PAIR(left, getLeftChild(right, bvhB));
					PUSH_PAIR(left, getRightChild(right, bvhB));
				}
				else {
					PUSH_PAIR(getLeftChild(left, bvhA), right);
					PUSH_PAIR(getRightChild(left, bvhA), right);
				}
			}
		}

		if (EMPTY)
			return;

		NEXT(left, right);
	}
}

__device__ void doPropogate(
	uint4 *front, uint *frontIdx, int num,
	int *bvhA, g_box *bxsA, int bvhAnum,
	int *bvhB, g_box *bxsB, int bvhBnum,
	int2 *pairs, uint *pairIdx, bool update, tri3f *Atris, int idx, bool *flags)
{
	uint4 node = front[idx];
	if (node.z != 0) {
#if defined(_DEBUG) || defined(OUTPUT_TXT)
		atomicAdd(frontIdx + 1, 1);
#endif
		return;
	}

#ifdef USE_NC
	if (flags != NULL && flags[node.w] == 0) {
#if defined(_DEBUG) || defined(OUTPUT_TXT)
		atomicAdd(frontIdx + 2, 1);
#endif
		return;
	}
#endif

	uint left = node.x;
	uint right = node.y;

	if (isLeaf(left, bvhA) && isLeaf(right, bvhB)) {
		bool tem = overlaps(left, right, bxsA, bxsB);
		if (bxsA[left]._min.z > bxsB[right]._max.z)
			tem = false;
		if (bxsA[left]._min.z == bxsB[right]._max.z)
			tem = false;
		if (overlaps(left, right, bxsA, bxsB))
			if (bvhA != bvhB)
				addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
			else { // for self ccd, we need to remove adjacent triangles, they will be processed seperatedly with orphan set
				if (!covertex(getTriID(left, bvhA), getTriID(right, bvhB), Atris))
					addPair(getTriID(left, bvhA), getTriID(right, bvhB), pairs, pairIdx);
			}
			return;
	}

	if (!overlaps(left, right, bxsA, bxsB))
		return;

	if (update)
		front[idx].z = 1;

	int ptr = node.w;
	if (isLeaf(left, bvhA)) {
		sprouting(left, getLeftChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
		sprouting(left, getRightChild(right, bvhB), bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
	}
	else {
		sprouting(getLeftChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
		sprouting(getRightChild(left, bvhA), right, bvhA, bxsA, bvhB, bxsB, front, frontIdx, pairs, pairIdx, update, ptr, Atris);
	}
}


__global__ void kernelPropogate(uint4 *front, uint *frontIdx, int num,
	int *bvhA, g_box *bxsA, int bvhAnum,
	int *bvhB, g_box *bxsB, int bvhBnum,uint* tt,
	int2 *pairs, uint *pairIdx, bool update, tri3f *Atris, int stride, bool *flags)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i<stride; i++) {
		int j = idx*stride + i;
		if (j >= num)
			return;
		atomicAdd(tt, 1);
		doPropogate(front, frontIdx, num,
			bvhA, bxsA, bvhAnum, bvhB, bxsB, bvhBnum, pairs, pairIdx, update, Atris, j, flags);
	}
}

int g_front::propogate(bool &update, bool self, bool ccd)
{
	uint dummy[1];
	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef OUTPUT_TXT
	printf("Before propogate, length = %d\n", dummy[0]);
#endif

#if defined(_DEBUG) || defined(OUTPUT_TXT)
	uint dummy2[5] = { 0, 0, 0, 0, 0 };
	cutilSafeCall(cudaMemcpy(_dIdx + 1, dummy2, 5 * sizeof(int), cudaMemcpyHostToDevice));
#endif

	if (dummy[0] != 0) {
		g_bvh *pb1 = &theBVH[1];
		g_bvh *pb2 = (self) ? &theBVH[1] : &theBVH[0];
		tri3f *faces = (self ? theCloth._df: NULL);

		int stride = 4;
#ifdef FIX_BT_NUM
		BLK_PAR2(dummy[0], stride);
#else
		BLK_PAR3(dummy[0], stride, getBlkSize((void *)kernelPropogate));
#endif
		uint* tt;
		cutilSafeCall(cudaMalloc((void**)&tt, 1 * sizeof(uint)));
		uint ttt = 0;
		cutilSafeCall(cudaMemcpy(tt, &ttt, 1 * sizeof(uint), cudaMemcpyHostToDevice));
		

		kernelPropogate << < B, T >> >
			(_dFront, _dIdx, dummy[0],
			pb1->_bvh, pb1->_bxs, pb1->_num,
			pb2->_bvh, pb2->_bxs, pb2->_num,tt,
			thePairs[self]._dPairs, thePairs[self]._dIdx, update, faces, stride, self ? theBVH[1]._ctFlags : NULL);
		//thePairs[self]._dPairs, thePairs[self]._dIdx, update, faces, stride, (self && !ccd) ? theBVH[1]._ctFlags : NULL);
		//reportMemory("propogate");

		//cutilSafeCall(cudaMemcpy(&ttt, tt, 1 * sizeof(uint), cudaMemcpyDeviceToHost));

		cudaThreadSynchronize();
		getLastCudaError("kernelPropogate");
	}

	cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
#ifdef OUTPUT_TXT
	printf("After propogate, length = %d\n", dummy[0]);
#endif

#if defined(_DEBUG) || defined(OUTPUT_TXT)
	cutilSafeCall(cudaMemcpy(dummy2, _dIdx + 1, 5 * sizeof(int), cudaMemcpyDeviceToHost));
	//printf("Invalid = %d, NC culled = %d\n", dummy2[0], dummy2[1]);
#endif

	if (update && dummy[0] > SAFE_FRONT_NUM) {
		printf("Too long front, stop updating ...\n");
		update = false;
	}

	if (dummy[0] > MAX_FRONT_NUM) {
		printf("Too long front, exiting ...\n");
		exit(0);
	}
	return dummy[0];
}

//===============================================

__global__ void
kernel_face_ws(tri3f *face, REAL3 *x, REAL3 *ox, g_box *bxs, bool ccd, REAL thickness, int num)
{
	LEN_CHK(num);

	int id0 = face[idx].id0();
	int id1 = face[idx].id1();
	int id2 = face[idx].id2();

	REAL3 ox0 = ox[id0];
	REAL3 ox1 = ox[id1];
	REAL3 ox2 = ox[id2];
	REAL3 x0 = x[id0];
	REAL3 x1 = x[id1];
	REAL3 x2 = x[id2];

	bxs[idx].set(ox0, ox1);
	bxs[idx].add(ox2);

	if (ccd) {
		bxs[idx].add(x0);
		bxs[idx].add(x1);
		bxs[idx].add(x2);
	}
	//else
	bxs[idx].enlarge(thickness);
}

void g_mesh::computeWSdata(REAL thickness, bool ccd)
{
	if (numFace == 0)
		return;

	{
		int num = numFace;
		BLK_PAR(num);
		kernel_face_ws << <B, T >> > (
			_df, _dx, _dx0, _dfBx, ccd, thickness, num);
		getLastCudaError("kernel_face_ws");
	}
}

//===============================================
 __device__ REAL triProduct(REAL3 &a, REAL3 &b, REAL3 &c)
{
	return dot(cross(a, b), c);
}

 __device__ REAL3 xvpos(REAL3 x, REAL3 v, REAL t)
{
	return x + v*t;
}

 __device__ int sgn(REAL x) { return x<0 ? -1 : 1; }

 FORCEINLINE __device__ int solve_quadratic(REAL a, REAL b, REAL c, REAL x[2]) {
	// http://en.wikipedia.org/wiki/Quadratic_formula#Floating_point_implementation
	REAL d = b*b - 4 * a*c;
	if (d < 0) {
		x[0] = -b / (2 * a);
		return 0;
	}
	REAL q = -(b + sgn(b)*sqrt(d)) / 2;
	int i = 0;
	if (abs(a) > 1e-12*abs(q))
		x[i++] = q / a;
	if (abs(q) > 1e-12*abs(c))
		x[i++] = c / q;
	if (i == 2 && x[0] > x[1])
		fswap(x[0], x[1]);
	return i;
}

 FORCEINLINE __device__ REAL newtons_method(REAL a, REAL b, REAL c, REAL d, REAL x0,
	int init_dir) {
	if (init_dir != 0) {
		// quadratic approximation around x0, assuming y' = 0
		REAL y0 = d + x0*(c + x0*(b + x0*a)),
			ddy0 = 2 * b + x0*(6 * a);
		x0 += init_dir*sqrt(abs(2 * y0 / ddy0));
	}
	for (int iter = 0; iter < 100; iter++) {
		REAL y = d + x0*(c + x0*(b + x0*a));
		REAL dy = c + x0*(2 * b + x0 * 3 * a);
		if (dy == 0)
			return x0;
		REAL x1 = x0 - y / dy;
		if (abs(x0 - x1) < 1e-6)
			return x0;
		x0 = x1;
	}
	return x0;
}


// solves a x^3 + b x^2 + c x + d == 0
FORCEINLINE __device__
int solve_cubic(REAL a, REAL b, REAL c, REAL d, REAL x[3])
{
	REAL xc[2];
	int ncrit = solve_quadratic(3 * a, 2 * b, c, xc);
	if (ncrit == 0) {
		x[0] = newtons_method(a, b, c, d, xc[0], 0);
		return 1;
	}
	else if (ncrit == 1) {// cubic is actually quadratic
		return solve_quadratic(b, c, d, x);
	}
	else {
		REAL yc[2] = { d + xc[0] * (c + xc[0] * (b + xc[0] * a)),
			d + xc[1] * (c + xc[1] * (b + xc[1] * a)) };
		int i = 0;
		if (yc[0] * a >= 0)
			x[i++] = newtons_method(a, b, c, d, xc[0], -1);
		if (yc[0] * yc[1] <= 0) {
			int closer = abs(yc[0])<abs(yc[1]) ? 0 : 1;
			x[i++] = newtons_method(a, b, c, d, xc[closer], closer == 0 ? 1 : -1);
		}
		if (yc[1] * a <= 0)
			x[i++] = newtons_method(a, b, c, d, xc[1], 1);
		return i;
	}
}

__device__ REAL signed_ee_distance(const REAL3 &x0, const REAL3 &x1,
	const REAL3 &y0, const REAL3 &y1,
	REAL3 *n, REAL *w);

__device__ REAL signed_vf_distance
(const REAL3 &x,
	const REAL3 &y0, const REAL3 &y1, const REAL3 &y2,
	REAL3 *n, REAL *w);

 __device__ bool collision_test(
	const REAL3 &x0, const REAL3 &x1, const REAL3 &x2, const REAL3 &x3,
	const REAL3 &v0, const REAL3 &v1, const REAL3 &v2, const REAL3 &v3,REAL &time,const int isVF)//0:vf  1:ee
{
	REAL a0 = stp(x1, x2, x3),
		a1 = stp(v1, x2, x3) + stp(x1, v2, x3) + stp(x1, x2, v3),
		a2 = stp(x1, v2, v3) + stp(v1, x2, v3) + stp(v1, v2, x3),
		a3 = stp(v1, v2, v3);

	if (a1 == 0 && a2 == 0 && a3 == 0)
		return false;


	REAL t[4];
	int nsol = solve_cubic(a3, a2, a1, a0, t);
	//t[nsol] = 1; // also check at end of timestep

	bool t_flag = false;
	for (int i = 0; i < nsol; i++) {
		if (t[i] < 0 || t[i] > 1)
			continue;
		
		REAL3 tx0 = xvpos(x0, v0, t[i]), tx1 = xvpos(x1 + x0, v1 + v0, t[i]),
			tx2 = xvpos(x2 + x0, v2 + v0, t[i]), tx3 = xvpos(x3 + x0, v3 + v0, t[i]);

		REAL3 n;
		REAL w[4];
		REAL d;
		bool inside;

		if (isVF == 0) {
			d = signed_vf_distance(tx0, tx1, tx2, tx3, &n, w);
			inside = (fmin(-w[1], fmin(-w[2], -w[3])) >= -1e-6);
		}
		else {// Impact::EE
			d = signed_ee_distance(tx0, tx1, tx2, tx3, &n, w);
			inside = (fmin(fmin(w[0], w[1]), fmin(-w[2], -w[3])) >= -1e-6);
		}

		if (fabs(d) < 1e-6 && inside)
		{
			time = t[i];
			return true;
		}
			

		//t_flag = true;
		//if (t[i] < time)
			//time = t[i];
	}
	return t_flag;
}

__device__ void doImpactVF(
	REAL3 x0, REAL3 x1, REAL3 x2, REAL3 x3,
	REAL3 x00,REAL3 x10,REAL3 x20,REAL3 x30,REAL &time
)
{
	REAL3 p0 = x00;
	REAL3 p1 = x10 - x00;
	REAL3 p2 = x20 - x00;
	REAL3 p3 = x30 - x00;
	REAL3 v0 = x0 - x00;
	REAL3 v1 = x1 - x10 - v0;
	REAL3 v2 = x2 - x20 - v0;
	REAL3 v3 = x3 - x30 - v0;

	/*	if (iii == 69 &&
	vid == 162 && t.id0() == 4 &&
	t.id1() == 44 && t.id2() == 32)
	vid = 162;
	*/
#ifdef  USE_DNF_FILTER
	//bool ret1 = dnf_filter(x00, x10, x20, x30, x0, x1, x2, x3);
#endif
	bool ret = collision_test(p0, p1, p2, p3, v0, v1, v2, v3,time,0);
	if (ret) {

	}
}

inline __device__ REAL3 norm(REAL3 p1, REAL3 p2, REAL3 p3)
{
	return cross(p2 - p1, p3 - p1);
}

inline __device__ bool
dnf_filter(REAL3 &a0, REAL3 &b0, REAL3 &c0, REAL3 &d0,
	REAL3 &a1, REAL3 &b1, REAL3 &c1, REAL3 &d1)
{
	REAL3 n0 = norm(a0, b0, c0);
	REAL3 n1 = norm(a1, b1, c1);
	REAL3 delta = norm(a1 - a0, b1 - b0, c1 - c0);
	REAL3 nX = (n0 + n1 - delta)*REAL(0.5);

	REAL3 pa0 = d0 - a0;
	REAL3 pa1 = d1 - a1;

	REAL A = dot(n0, pa0);
	REAL B = dot(n1, pa1);
	REAL C = dot(nX, pa0);
	REAL D = dot(nX, pa1);
	REAL E = dot(n1, pa0);
	REAL F = dot(n0, pa1);

	if (A > 0 && B > 0 && (REAL(2.0) * C + F) > 0 && (REAL(2.0) * D + E) > 0)
		return false;

	if (A < 0 && B < 0 && (REAL(2.0) * C + F) < 0 && (REAL(2.0) * D + E) < 0)
		return false;

	return true;
}



__device__ void doImpactEE(
	REAL3 x0, REAL3 x1, REAL3 x2, REAL3 x3,
	REAL3 x00, REAL3 x10, REAL3 x20, REAL3 x30, REAL &time)
{

	REAL3 p0 = x00;
	REAL3 p1 = x10 - x00;
	REAL3 p2 = x20 - x00;
	REAL3 p3 = x30 - x00;
	REAL3 v0 = x0 - x00;
	REAL3 v1 = x1 - x10 - v0;
	REAL3 v2 = x2 - x20 - v0;
	REAL3 v3 = x3 - x30 - v0;

#ifdef  USE_DNF_FILTER
	//bool ret1 = dnf_filter(x10, x20, x30, x40, x1, x2, x3, x4);
#endif
	/*	if (e0.x == 41 && e0.y == 624 &&
	e1.x == 599 && e1.y == 383)
	e0.x = 41;
	*/
	//bool ret1 = dnf_filter(p0, p1, p2, p3, v0, v1, v2, v3);

	//if (!ret1)
	//{
	//	return;
	//}

	bool ret = collision_test(p0, p1, p2, p3, v0, v1, v2, v3,time,1);
	if (ret) {
	}
}

__device__ REAL signed_vf_distance
(const REAL3 &x,
	const REAL3 &y0, const REAL3 &y1, const REAL3 &y2,
	REAL3 *n, REAL *w)
{
	REAL3 _n; if (!n) n = &_n;
	REAL _w[4]; if (!w) w = _w;
	*n = cross(normalize(y1 - y0), normalize(y2 - y0));
	if (norm2(*n) < 1e-6)
		return REAL_infinity;
	*n = normalize(*n);
	REAL h = dot(x - y0, *n);
	REAL b0 = stp(y1 - x, y2 - x, *n),
		b1 = stp(y2 - x, y0 - x, *n),
		b2 = stp(y0 - x, y1 - x, *n);
	w[0] = 1;
	w[1] = -b0 / (b0 + b1 + b2);
	w[2] = -b1 / (b0 + b1 + b2);
	w[3] = -b2 / (b0 + b1 + b2);
	return h;
}

__device__ REAL signed_ee_distance(const REAL3 &x0, const REAL3 &x1,
	const REAL3 &y0, const REAL3 &y1,
	REAL3 *n, REAL *w) {
	REAL3 _n; if (!n) n = &_n;
	REAL _w[4]; if (!w) w = _w;
	*n = cross(normalize(x1 - x0), normalize(y1 - y0));
	if (norm2(*n) < 1e-6)
		return REAL_infinity;
	*n = normalize(*n);
	REAL h = dot(x0 - y0, *n);
	REAL a0 = stp(y1 - x1, y0 - x1, *n), a1 = stp(y0 - x0, y1 - x0, *n),
		b0 = stp(x0 - y1, x1 - y1, *n), b1 = stp(x1 - y0, x0 - y0, *n);
	w[0] = a0 / (a0 + a1);
	w[1] = a1 / (a0 + a1);
	w[2] = -b0 / (b0 + b1);
	w[3] = -b1 / (b0 + b1);
	return h;
}



__device__ REAL doProximityVF(
	REAL3 _x4, REAL3 _x1, REAL3 _x2, REAL3 _x3, REAL *_mrt
)
{
	REAL mrt = _mrt[0];

	REAL3 x1 = _x1;
	REAL3 x2 = _x2;
	REAL3 x3 = _x3;
	REAL3 x4 = _x4;

	REAL3 n;
	REAL w[4];
	REAL d = signed_vf_distance(x4, x1, x2, x3, &n, w);
	d = abs(d);

	const REAL dmin = mrt;
	if (d >= dmin)
		return d;

	bool inside = (min(min(-w[1], -w[2]), -w[3]) >= -1e-6);
	if (!inside)
		return -d;

	return d;
}

__device__ REAL doProximityEE(
	REAL3 _e00, REAL3 _e01, REAL3 _e10, REAL3 _e11, REAL *_mrt
)
{
	REAL mrt = _mrt[0];

	REAL3 e00 = _e00;
	REAL3 e01 = _e01;
	REAL3 e10 = _e10;
	REAL3 e11 = _e11;


	/*	if (e0.x == 891 && e0.y == 446 &&
	e1.x == 18 && e1.y == 188)
	e1.x = 18;
	*/
	REAL3 n;
	REAL w[4];
	REAL d = signed_ee_distance(e00, e01, e10, e11, &n, w);
	d = abs(d);
	if (d == 0)
		int sd = d;

	const REAL dmin = mrt;
	if (d >= dmin)
		return d;

	bool inside = min(min(w[0], w[1]), min(-w[2], -w[3])) >= -1e-6;
	if (!inside) return -d;

	return d;

}



__device__ void tri_CCD(
	uint t1, uint t2, uint t3,
	uint tt1, uint tt2, uint tt3,
	REAL3 *cx, REAL3 *cx0, int2 *pairRets, uint *pairIdx, int fid1, int fid2, int *t
)
{
	REAL3 p[3];
	p[0] = cx[t1];
	p[1] = cx[t2];
	p[2] = cx[t3];

	REAL3 pp[3];
	pp[0] = cx0[t1];
	pp[1] = cx0[t2];
	pp[2] = cx0[t3];

	REAL3 q[3];
	q[0] = cx[tt1];
	q[1] = cx[tt2];
	q[2] = cx[tt3];

	REAL3 qq[3];
	qq[0] = cx0[tt1];
	qq[1] = cx0[tt2];
	qq[2] = cx0[tt3];

	REAL time = 2;

	///*
	//VF
	for (int st = 0; st < 3; st++)
	{
		doImpactVF(
			p[st], q[0], q[1], q[2],
			pp[st], qq[0], qq[1], qq[2],
			time
		);
	}

	//VF
	for (int st = 0; st < 3; st++)
	{
		doImpactVF(q[st], p[0], p[1], p[2],
			qq[st], pp[0], pp[1], pp[2],
			time
		);
	}

	//EE
	uint idx0[3];
	uint idx1[3];

	idx0[0] = t1;
	idx0[1] = t2;
	idx0[2] = t3;

	idx1[0] = tt1;
	idx1[1] = tt2;
	idx1[2] = tt3;

	for (int st = 0; st < 3; st++)
	{
		int ind0 = st;
		int ind1 = (st + 1) % 3;
		for (int ss = 0; ss < 3; ss++)
		{
			int ind2 = st;
			int ind3 = (st + 1) % 3;
			doImpactEE(
				cx[idx0[ind0]], cx[idx0[ind1]], cx[idx1[ind2]], cx[idx1[ind3]],
				cx0[idx0[ind0]], cx0[idx0[ind1]], cx0[idx1[ind2]], cx0[idx1[ind3]],
				time
			);
		}
	}

	if (time >= 0 && time <= 1)
	{
		//int index = addPair(fid1, fid2, pairRets, pairIdx);
		//if (index != -1)
		//{
			t[0]=1;
		//}
	}
}

__device__ void tri_DCD(
	uint t1, uint t2, uint t3,
	uint tt1, uint tt2, uint tt3,
	REAL3 *cx, REAL3 *cx0, int2 *pairRets, int4 *dv, int *VF_EE, REAL* dist, int *CCDres, int *t, uint *pairIdx, int fid1, int fid2, REAL *thickness
)
{
	REAL3 p[3];
	p[0] = cx[t1];
	p[1] = cx[t2];
	p[2] = cx[t3];

	REAL3 pp[3];
	pp[0] = cx0[t1];
	pp[1] = cx0[t2];
	pp[2] = cx0[t3];

	REAL3 q[3];
	q[0] = cx[tt1];
	q[1] = cx[tt2];
	q[2] = cx[tt3];

	REAL3 qq[3];
	qq[0] = cx0[tt1];
	qq[1] = cx0[tt2];
	qq[2] = cx0[tt3];

	if (tri_contact(p[0], p[1], p[2], q[0], q[1], q[2]))
	
		if (fid1 > fid2)
			addPairDCD(fid2, fid1, 0, 0, 0, 0, 0, 0, NULL,
				pairRets, NULL, NULL, NULL, CCDres , pairIdx);
		else
			addPairDCD(fid1, fid2, 0, 0, 0, 0, 0, 0, NULL,
				pairRets, NULL, NULL, NULL, CCDres, pairIdx);
}

__device__ void tri_DCD2(
	uint t1, uint t2, uint t3,
	uint tt1, uint tt2, uint tt3,
	REAL3 *cx, REAL3 *cx0, int2 *pairRets, int4 *dv, int *VF_EE, REAL* dist,int *CCDres,int *t, uint *pairIdx, int fid1, int fid2, REAL *thickness
)
{
	REAL3 p[3];
	p[0] = cx[t1];
	p[1] = cx[t2];
	p[2] = cx[t3];

	REAL3 pp[3];
	pp[0] = cx0[t1];
	pp[1] = cx0[t2];
	pp[2] = cx0[t3];

	REAL3 q[3];
	q[0] = cx[tt1];
	q[1] = cx[tt2];
	q[2] = cx[tt3];

	REAL3 qq[3];
	qq[0] = cx0[tt1];
	qq[1] = cx0[tt2];
	qq[2] = cx0[tt3];

	//REAL time = 2;

	uint idx0[3];
	uint idx1[3];

	idx0[0] = t1;
	idx0[1] = t2;
	idx0[2] = t3;

	idx1[0] = tt1;
	idx1[1] = tt2;
	idx1[2] = tt3;

	REAL time = 2;
	t[0] = 0;

	///*
	//VF
	for (int st = 0; st < 3; st++)
	{
		t[0] = 0;
		time = 2;

		doImpactVF(
			p[st], q[0], q[1], q[2],
			pp[st], qq[0], qq[1], qq[2],
			time
		);

		REAL d = doProximityVF(
			p[st], q[0], q[1], q[2],
			thickness
		);

		if ((time >= 0 && time <= 1))
		{
			t[0] = 1;
		}

		if ((time >= 0 && time <= 1)||(d<thickness[0] && d>=0))
		{
			
			addPairDCD(fid1, fid2, 0, idx0[st], idx1[0], idx1[1], idx1[2], d, t,
				pairRets, dv, VF_EE, dist, CCDres, pairIdx);
		}

		

		if (d != -1)
		{
			//addPairDCD(fid1, fid2, 0, idx0[st], idx1[0], idx1[1], idx1[2], d,t,
			//	pairRets, dv, VF_EE, dist, CCDres,pairIdx);
		}
	}

	//VF
	for (int st = 0; st < 3; st++)
	{
		t[0] = 0;
		time = 2;

		if (st == 0)
		{
			REAL d = doProximityVF(
				q[st], p[0], p[1], p[2],
				thickness
			);
		}

		doImpactVF(q[st], p[0], p[1], p[2],
			qq[st], pp[0], pp[1], pp[2],
			time
		);

		REAL d = doProximityVF(
			q[st], p[0], p[1], p[2],
			thickness
		);

		if ((time >= 0 && time <= 1))
		{
			t[0] = 1;
		}

		if ((time >= 0 && time <= 1) || (d < thickness[0] && d >= 0))
		{
			addPairDCD(fid1, fid2, 0, idx1[st], idx0[0], idx0[1], idx0[2], d, t,
				pairRets, dv, VF_EE, dist, CCDres, pairIdx);
		}

		

		if (d != -1)
		{
			//addPairDCD(fid1, fid2, 0, idx1[st], idx0[0], idx0[1], idx0[2], d,t,
			//	pairRets, dv, VF_EE, dist, CCDres, pairIdx);
		}
	}

	//EE

	for (int st = 0; st < 3; st++)
	{
		int ind0 = st;
		int ind1 = (st + 1) % 3;
		for (int ss = 0; ss < 3; ss++)
		{
			int ind2 = ss;
			int ind3 = (ss + 1) % 3;

			t[0] = 0;
			time = 2;

			doImpactEE(
				cx[idx0[ind0]], cx[idx0[ind1]], cx[idx1[ind2]], cx[idx1[ind3]],
				cx0[idx0[ind0]], cx0[idx0[ind1]], cx0[idx1[ind2]], cx0[idx1[ind3]],
				time
			);

			REAL d = doProximityEE(
				cx[idx0[ind0]], cx[idx0[ind1]], cx[idx1[ind2]], cx[idx1[ind3]],
				thickness
			);

			if ((time >= 0 && time <= 1))
			{
				t[0] = 1;
			}

			if ((time >= 0 && time <= 1) || (d < thickness[0] && d >= 0))
			{

				addPairDCD(fid1, fid2, 1, idx0[ind0], idx0[ind1], idx1[ind2], idx1[ind3], d, t,
					pairRets, dv, VF_EE, dist, CCDres, pairIdx);
			}


			
			if (d != -1)
			{
				//addPairDCD(fid1, fid2, 1, idx0[ind0], idx0[ind1], idx1[ind2], idx1[ind3], d,t,
				//	pairRets, dv, VF_EE, dist, CCDres, pairIdx);
			}
		}


	}
}

__global__ void kernelGetCollisions(
	int2 *pairs, int num, 
	REAL3 *cx, REAL3 *cx0,tri3f *ctris, int2 *pairRets, int4 *dv, int *VF_EE, REAL* dist,int *CCDres, uint *pairIdx, int *t, REAL *thickness,
	int stride)
{
	
	int idxx = blockDim.x * blockIdx.x + threadIdx.x;

	for (int i = 0; i < stride; i++) {

		int j = idxx*stride + i;
		if (j >= num)
			return;

		int idx = j;

		int2 pair = pairs[idx];
		int fid1 = pair.x;
		int fid2 = pair.y;

		int ss = 0;

		tri3f t1 = ctris[fid1];
		tri3f t2 = ctris[fid2];

		uint tt1[3];
		uint tt2[3];
		tt1[0] = t1.id0();
		tt1[1] = t1.id1();
		tt1[2] = t1.id2();

		tt2[0] = t2.id0();
		tt2[1] = t2.id1();
		tt2[2] = t2.id2();


#ifdef FOR_DEBUG
		bool find = false;
		if (fid1 == 369 && fid2 == 3564)
			find = true;
		if (fid2 == 369 && fid1 == 3564)
			find = true;
#endif
		REAL3 p[3];
		p[0] = cx[t1.id0()];
		p[1] = cx[t1.id1()];
		p[2] = cx[t1.id2()];

		REAL3 pp[3];
		pp[0] = cx0[t1.id0()];
		pp[1] = cx0[t1.id1()];
		pp[2] = cx0[t1.id2()];

		REAL3 q[3];
		q[0] = cx[t2.id0()];
		q[1] = cx[t2.id1()];
		q[2] = cx[t2.id2()];

		REAL3 qq[3];
		qq[0] = cx0[t2.id0()];
		qq[1] = cx0[t2.id1()];
		qq[2] = cx0[t2.id2()];

		bool iscovetex = false;
		for (int st = 0; st < 3; st++)
		{
			for (int ss = 0; ss < 3; ss++)
			{
				if (tt1[st] == tt2[ss])
				{
					iscovetex = true;
				}
			}
		}

		if (iscovetex)
			continue;

		//tri_CCD(t1.id0(), t1.id1(), t1.id2(), t2.id0(), t2.id1(), t2.id2(), cx, cx0, pairRets, pairIdx, fid1, fid2, t);

		tri_DCD(t1.id0(), t1.id1(), t1.id2(), t2.id0(), t2.id1(), t2.id2(), cx, cx0, pairRets, dv, VF_EE, dist,CCDres,t, pairIdx, fid1, fid2, thickness);
	}

/*
		 ///*
		 //VF
		 for (int st = 0; st < 3; st++)
		 {
			 doImpactVF(
				 p[st], q[0], q[1], q[2],
				  pp[st], qq[0], qq[1], qq[2],
				 time
			 );
		 }

		 //VF
		 for (int st = 0; st < 3; st++)
		 {
			 doImpactVF(q[st], p[0], p[1], p[2],
				  qq[st], pp[0], pp[1], pp[2],
				 time
			 );
		 }

		 //EE
		 int idx0[2];
		 int idx1[2];
		 for (int st = 0; st < 3; st++)
		 {
			 idx0[0] = st;
			 idx0[1] = (st + 1) % 3;
			 for (int ss = 0; ss < 3; ss++)
			 {
				 idx1[0] = ss;
				 idx1[1] = (ss + 1) % 3;
				 doImpactEE(
					 cx[idx0[0]], cx[idx0[1]], cx[idx1[0]], cx[idx1[1]],
					 cx0[idx0[0]], cx0[idx0[1]], cx0[idx1[0]], cx0[idx1[1]],
					 time
				 );
			 }
		 }
		 

#ifdef FOR_DEBUG
		if (find) {
			printf("%d: %lf, %lf, %lf\n", t1.id0(), p0.x, p0.y, p0.z);
			printf("%d: %lf, %lf, %lf\n", t1.id1(), p1.x, p1.y, p1.z);
			printf("%d: %lf, %lf, %lf\n", t1.id2(), p2.x, p2.y, p2.z);
			printf("%d: %lf, %lf, %lf\n", t2.id0(), q0.x, q0.y, q0.z);
			printf("%d: %lf, %lf, %lf\n", t2.id1(), q1.x, q1.y, q1.z);
			printf("%d: %lf, %lf, %lf\n", t2.id2(), q2.x, q2.y, q2.z);
		}
#endif
		
		REAL3 p0 = cx[t1.id0()];
		REAL3 p1 = cx[t1.id1()];
		REAL3 p2 = cx[t1.id2()];
		REAL3 q0 = cx[t2.id0()];
		REAL3 q1 = cx[t2.id1()];
		REAL3 q2 = cx[t2.id2()];

		if (tri_contact(p0, p1, p2, q0, q1, q2))
			addPair(fid1, fid2, pairRets, pairIdx);
		

		
		if (time >= 0 && time <= 1)
		{
			int index = addPair(fid1, fid2, pairRets, pairIdx);
			if (index != -1)
			{
				t[index] = time;
			}
		}
	
	}
	*/

}

//===============================================

int g_pair::getCollisions(bool self, g_pairCCD &ret, int *time, REAL* thickness)
{
	int num = length();
	printf("pair = %d\n", num);
#ifdef OUTPUT_TXT
	//if (self)
		//printf("self pair = %d\n", num);
	//else
		//printf("inter-obj pair = %d\n", num);
#endif

	if (num == 0)
		return 0;

	ret.clear();

	int stride = 4;
#ifdef FIX_BT_NUM
	BLK_PAR3(num, stride, 32);
#else
	BLK_PAR3(num, stride, getBlkSize((void *)kernelGetCollisions));
#endif

	int *tem_time;
	int tem = 0;
	cutilSafeCall(cudaMalloc((void**)&tem_time, 1 * sizeof(int)));
	cutilSafeCall(cudaMemcpy(tem_time, &tem, 1 * sizeof(int), cudaMemcpyHostToDevice));

	REAL *gthickness;
	cutilSafeCall(cudaMalloc((void**)&gthickness, 1 * sizeof(REAL)));
	cutilSafeCall(cudaMemcpy(gthickness, thickness, 1 * sizeof(REAL), cudaMemcpyHostToDevice));

	kernelGetCollisions << < B, T >> > (_dPairs, num,
		theCloth._dx, theCloth._dx0, theCloth._df, ret._dPairs, ret._dv, ret._dVF_EE, ret.dist,ret.CCD_res, ret._dIdx, tem_time, gthickness, stride);

	getLastCudaError("kernelGetCollisions");
	

	int len = ret.length();
#ifdef OUTPUT_TXT
	//printf("collision num = %d\n", len);
#endif
	printf("collision num = %d\n", len);

	//if (len > 0)
	//{
		cutilSafeCall(cudaMemcpy(time, tem_time, sizeof(int)* 1, cudaMemcpyDeviceToHost));
	//}

	cudaFree(tem_time);

	return len;
}
//===============================================

int getCollisionsGPU(int *rets, int *vf_ee, int *vertex_id, REAL *dist, int *time,int* CCDres, REAL* thickness)
{
	bool update = true;
	int len = 0;

	TIMING_BEGIN
	thePairs[1].clear();


	refitBVH(true);
	theFront[1].propogate(update, 1, false);
	cudaThreadSynchronize();

	len = thePairs[1].getCollisions(true, retPairsCCD, time, thickness);
	cudaThreadSynchronize();
	TIMING_END("$$$get_collisions_gpu")


	if (len > 0) {
		cutilSafeCall(cudaMemcpy(rets, retPairsCCD._dPairs, sizeof(uint) * 2 * len, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaMemcpy(vf_ee, retPairsCCD._dVF_EE, sizeof(int) * len, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaMemcpy(vertex_id, retPairsCCD._dv, sizeof(int) * 4 * len, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaMemcpy(dist, retPairsCCD.dist, sizeof(double) * len, cudaMemcpyDeviceToHost));

		cutilSafeCall(cudaMemcpy(CCDres, retPairsCCD.CCD_res, sizeof(int) * len, cudaMemcpyDeviceToHost));
	}
	return len;
}
