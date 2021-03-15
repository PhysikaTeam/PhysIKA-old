typedef enum {
	I_NULL = -1,
	I_VF = 0,
	I_EE = 1
} ImpactType;

struct g_impNode {
	uint _n;
	bool _f;
	REAL3 _ox;
	REAL3 _x;
	REAL _m;

	 __host__ __device__ g_impNode(uint n, bool f, const REAL3 &ox, const REAL3 &x, REAL m) {
		_n = n;
		_f = f;
		_ox = ox;
		_x = x;
		_m = m;
	}
};

struct g_impact {
	uint _nodes[4];
	bool _frees[4];
	REAL _w[4];

    ImpactType _type;
    REAL _t;
	REAL3 _n;

	CU_FORCEINLINE __host__ __device__ g_impact()
	{
		_nodes[0] = _nodes[1] = _nodes[2] = _nodes[3] = -1;
		_frees[0] = _frees[1] = _frees[2] = _frees[3] = false;
		_w[0] = _w[1] = _w[2] = _w[3] = 0.0;
		_type = I_NULL;
		_t = 0.0;
		_n = zero3f();
	}

	CU_FORCEINLINE __host__ __device__ g_impact(ImpactType type, uint n0, uint n1, uint n2, uint n3, bool f0, bool f1, bool f2, bool f3)
	{
		_type = type;
		_nodes[0]  = n0, _nodes[1]  = n1, _nodes[2]  = n2, _nodes[3]  = n3;
		_frees[0] = f0, _frees[1] = f1, _frees[2] = f2, _frees[3] = f3;
    }
};

#define MAX_IMPACT_NUM 100000
#define MAX_IMP_NODE_NUM 400000

struct g_impacts {
	g_impact *_dImps;
	uint *_dImpNum;
	uint _hLength;

	g_impNode *_dNodes;
	uint *_dNodeNum;
	uint _hNodeNum;

	void init() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMalloc((void**)&_dImpNum, 1*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dImpNum, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMalloc((void**)&_dImps, MAX_IMPACT_NUM*sizeof(g_impact)) );
		cutilSafeCall(cudaMemset(_dImps, 0, MAX_IMPACT_NUM*sizeof(g_impact)) );

		cutilSafeCall(cudaMalloc((void**)&_dNodeNum, 1*sizeof(uint)) );
		cutilSafeCall(cudaMemcpy(_dNodeNum, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMalloc((void**)&_dNodes, MAX_IMP_NODE_NUM*sizeof(g_impNode)) );
		cutilSafeCall(cudaMemset(_dNodes, 0, MAX_IMP_NODE_NUM*sizeof(g_impNode)) );
		reportMemory("g_impacts.init");

		_hLength = 0;
		_hNodeNum = 0;

	}

	void clear() {
		uint dummy[] = {0};
		cutilSafeCall(cudaMemcpy(_dImpNum, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		cutilSafeCall(cudaMemcpy(_dNodeNum, dummy,1*sizeof(uint), cudaMemcpyHostToDevice));
		_hLength = 0;
		_hNodeNum = 0;
	}

	void destroy() {
		cudaFree(_dImps);
		cudaFree(_dImpNum);
		cudaFree(_dNodes);
		cudaFree(_dNodeNum);
	}

	int length() {
		return _hLength;
	}

	g_impact *data() {
		return _dImps;
	}

	g_impNode *nodes() {
		return _dNodes;
	}

	int updateLength() {
		cutilSafeCall(cudaMemcpy(&_hLength, _dImpNum, 1*sizeof(uint), cudaMemcpyDeviceToHost));
		cutilSafeCall(cudaMemcpy(&_hNodeNum, _dNodeNum, 1*sizeof(uint), cudaMemcpyDeviceToHost));
		assert(_hNodeNum == 4*_hLength);

		return _hLength;

	}
};

CU_FORCEINLINE __device__ void addImpact(g_impact *imps, uint *idx, g_impact &imp)
{
	if (*idx < MAX_IMPACT_NUM) 
	{
		uint offset = atomicAdd(idx, 1);
		imps[offset] = imp;
	}
}

CU_FORCEINLINE __device__ void addNodeInfo(g_impNode *nodes, uint *idx,
								   uint id, bool free, const REAL3 &ox, const REAL3 &x, REAL m)
{
	if (*idx < MAX_IMP_NODE_NUM) 
	{
		uint offset = atomicAdd(idx, 1);
		nodes[offset] = g_impNode(id, free, ox, x, m);
	}
}


