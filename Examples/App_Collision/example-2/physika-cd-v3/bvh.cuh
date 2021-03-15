
typedef struct {
	int _num;
	int *_bvh;
	int *_bvh_leaf;
	g_box *_bxs;
	g_cone *_cones;
	
	int _max_level;
	uint *_level_idx;

	// for contour test
	int _ctNum;
	uint *_ctIdx, *_ctLst;
	g_cone *_triCones; //directly borrow from g_mesh ...
	bool *_ctFlags;
	REAL3 *_ctPts, *_ctVels;

	uint2 *_fids;
	g_box *_triBxs; //directly borrow form g_mesh ...

	g_box *hBxs; // for debug;

	void getBxs() {
		if (hBxs == NULL)
			hBxs = new g_box [_num];

		cudaMemcpy(hBxs, _bxs, _num*sizeof(g_box), cudaMemcpyDeviceToHost);
	}

	void printBxs(char *path) {
		FILE *fp = fopen(path, "wt");
		for (int i=0;i<_num;i++)
			hBxs[i].print(fp);
		fclose(fp);
	}

	void selfCollisionCulling(REAL3 *x, REAL3 *ox, bool ccd, uint *counting);

	void destory()
	{
		printf("unfinished!\n");
	}
} g_bvh;

//the length of the front <= triangle num
typedef struct {
	uint3 *_dFront; // node_id, parent_id, valid 0/1
	uint *_dIdx;
	uint _iMax;

	void init(int max_num) {
		//_iMax = max_num;
		_iMax = max_num * 2; //allow invalid nodes ...

		//start from the root, so at lest one ...
		uint dummy[] = {1};
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMalloc((void**)&_dFront, _iMax*sizeof(uint3)) );
		cutilSafeCall(cudaMemset(_dFront, 0, _iMax*sizeof(uint3)) );
		printf("max self-cone front num = %d", _iMax);

		reportMemory("g_cone_front.init");
	}

	void destroy() {
		cudaFree(_dFront);
		cudaFree(_dIdx);
	}

	void reset() {
		//start from the root
		uint dummy[] = {1};
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));

		uint3 dummy2;
		dummy2.x = dummy2.y = dummy2.z = 0;
		cutilSafeCall(cudaMemcpy(_dFront, &dummy2, 1 * sizeof(uint3), cudaMemcpyHostToDevice));
	}

	void push(int length, uint3 *data) {
		uint dummy[] = {length};
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		if (length)
			cutilSafeCall(cudaMemcpy(_dFront, data,length*sizeof(uint3), cudaMemcpyHostToDevice));
	}

	int propogate(bool ccd);
} g_cone_front;

#define SAFE_FRONT_NUM  54000000
#define MAX_FRONT_NUM   55000000

typedef struct {
	uint4 *_dFront; // left, right, valid 0/1, dummy
	uint *_dIdx;

	void init() {
		uint dummy[10] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 10*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,10*sizeof(uint), cudaMemcpyHostToDevice));

		cutilSafeCall(cudaMalloc((void**)&_dFront, MAX_FRONT_NUM*sizeof(uint4)) );
		cutilSafeCall(cudaMemset(_dFront, 0, MAX_FRONT_NUM*sizeof(uint4)) );
		reportMemory("g_front.init");
	}

	void destroy() {
		cudaFree(_dFront);
		cudaFree(_dIdx);
	}

	void push(int length, uint4 *data) {
		uint dummy[] = {length};
		cutilSafeCall(cudaMemcpy(_dIdx, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		if (length)
			cutilSafeCall(cudaMemcpy(_dFront, data,length*sizeof(uint4), cudaMemcpyHostToDevice));
	}

	int propogate(bool &update, bool self, bool ccd);

	uint length() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		return dummy[0];
	}

	void setLength(uint len) {
		cutilSafeCall(cudaMemcpy(_dIdx, &len, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}
} g_front;

inline __device__ int getParent(uint i, int *bvh_ids, int num)
{
	//return i-bvh_ids[num+i];
	return bvh_ids[num + i];
}

inline __device__ int getTriID(uint i, int *bvh_ids)
{
	return bvh_ids[i];
}

inline __device__ int getLeftChild(uint i, int *bvh_ids)
{
	return i-bvh_ids[i];
}

inline __device__ int getRightChild(uint i, int *bvh_ids)
{
	return i-bvh_ids[i]+1;
}

inline __device__ bool isLeaf(uint i, int *bvh_ids)
{
	return bvh_ids[i] >= 0;
}

inline __device__ bool overlaps(uint i, uint j, g_box *Abxs, g_box *Bbxs)
{
	return Abxs[i].overlaps(Bbxs[j]);
}

inline __device__ void refit(int i, int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes, g_cone *bvh_cones, g_cone *tri_cones)
{
	if (isLeaf(i, bvh_ids)) // isLeaf
	{
		int fid = getTriID(i, bvh_ids);
		bvh_boxes[i] = tri_boxes[fid];

		if (bvh_cones)
			bvh_cones[i].set(tri_cones[fid]);
	}
	else
	{
		int left = getLeftChild(i, bvh_ids);
		int right = getRightChild(i, bvh_ids);

		bvh_boxes[i].set(bvh_boxes[left], bvh_boxes[right]);

		if (bvh_cones) {
			bvh_cones[i].set(bvh_cones[left]);
			bvh_cones[i] += bvh_cones[right];
		}
	}
}

__global__ void refit_serial_kernel(int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes,
	g_cone *bvh_cones, g_cone *tri_cones,
	int num)
{
	for (int i=num-1; i>=0; i--) {
		refit(i, bvh_ids, bvh_boxes, tri_boxes, bvh_cones, tri_cones);
	}
}

__global__ void refit_kernel(int *bvh_ids, g_box *bvh_boxes, g_box *tri_boxes,
	g_cone *bvh_cones, g_cone *tri_cones,
	int st, int num)
{
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
	if (idx >= num)
		return;

	refit(idx + st, bvh_ids, bvh_boxes, tri_boxes, bvh_cones, tri_cones);
}
