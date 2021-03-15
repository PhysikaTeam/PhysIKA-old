
typedef struct _g_pair {
	int2 *_dPairs;
	uint *_dIdx;
	int _offset;
	int _length;
	int _max_length;

	void init(int length) {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1 * sizeof(uint)));
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));

		_length = length;
		cutilSafeCall(cudaMalloc((void**)&_dPairs, length*sizeof(int2)));
		cutilSafeCall(cudaMemset(_dPairs, 0, length*sizeof(uint2)));
		reportMemory("g_pair.init");

		_offset = 0;
		_max_length=0;
	}

	void append(_g_pair &pairs){
		uint len0[1];
		cutilSafeCall(cudaMemcpy(len0, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		uint len1[1];
		cutilSafeCall(cudaMemcpy(len1, pairs._dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		uint newlen[1];
		cutilSafeCall(cudaMemcpy(_dPairs + len0[0], pairs._dPairs, len1[0] * sizeof(int2), cudaMemcpyDeviceToDevice));
		newlen[0] = len0[0] + len1[0];
		cutilSafeCall(cudaMemcpy(_dIdx, newlen, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}

	void clear() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
		_offset = 0;
	}

	int getCollisions(bool self, struct _g_pairCCD &rets, int *time, REAL* thickness);
	int getProximityConstraints(bool self, REAL mu, REAL mu_obs, REAL mrt, REAL mcs);
	int getImpacts(bool self, REAL mu, REAL mu_obs, _g_pair &vfPairs, _g_pair &eePairs, int &vfLen, int &eeLen);

	void destroy() {
		cudaFree(_dPairs);
		cudaFree(_dIdx);
	}

	uint length() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		if (dummy[0] > _max_length)
			_max_length = dummy[0];

		return dummy[0];
	}

	void setLength(uint len) {
		cutilSafeCall(cudaMemcpy(_dIdx, &len, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}

	int maxLength() const { return _max_length; }
} g_pair;


typedef struct _g_pairCCD {
	int2 *_dPairs;  //id of face pair
	int *_dVF_EE;//VF OR EE
	int4 *_dv;  //vertexs
	double *dist;   //dist of vf or ee
	int *CCD_res;   //CCD result

	uint *_dIdx;
	int _offset;
	int _length;
	int _max_length;

	void init(int length) {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMalloc((void**)&_dIdx, 1 * sizeof(uint)));
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));

		_length = length;
		cutilSafeCall(cudaMalloc((void**)&_dPairs, length * sizeof(int2)));
		cutilSafeCall(cudaMemset(_dPairs, 0, length * sizeof(int2)));

		cutilSafeCall(cudaMalloc((void**)&_dVF_EE, length * sizeof(int)));
		cutilSafeCall(cudaMemset(_dVF_EE, 0, length * sizeof(int)));

		cutilSafeCall(cudaMalloc((void**)&_dv, length * sizeof(int4)));
		cutilSafeCall(cudaMemset(_dv, 0, length * sizeof(int4)));

		cutilSafeCall(cudaMalloc((void**)&dist, length * sizeof(double)));
		cutilSafeCall(cudaMemset(dist, 0, length * sizeof(double)));

		cutilSafeCall(cudaMalloc((void**)&CCD_res, length * sizeof(int)));
		cutilSafeCall(cudaMemset(CCD_res, 0, length * sizeof(int)));

		reportMemory("g_pair.init");

		_offset = 0;
		_max_length = 0;
	}

	void append(_g_pairCCD &pairs) {
		uint len0[1];
		cutilSafeCall(cudaMemcpy(len0, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		uint len1[1];
		cutilSafeCall(cudaMemcpy(len1, pairs._dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		uint newlen[1];
		cutilSafeCall(cudaMemcpy(_dPairs + len0[0], pairs._dPairs, len1[0] * sizeof(int2), cudaMemcpyDeviceToDevice));

		cutilSafeCall(cudaMemcpy(_dv + len0[0], pairs._dv, len1[0] * sizeof(int4), cudaMemcpyDeviceToDevice));

		cutilSafeCall(cudaMemcpy(_dVF_EE + len0[0], pairs._dVF_EE, len1[0] * sizeof(int), cudaMemcpyDeviceToDevice));

		cutilSafeCall(cudaMemcpy(dist + len0[0], pairs.dist, len1[0] * sizeof(double), cudaMemcpyDeviceToDevice));

		cutilSafeCall(cudaMemcpy(CCD_res + len0[0], pairs.CCD_res, len1[0] * sizeof(double), cudaMemcpyDeviceToDevice));

		newlen[0] = len0[0] + len1[0];
		cutilSafeCall(cudaMemcpy(_dIdx, newlen, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}

	void clear() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(_dIdx, dummy, 1 * sizeof(uint), cudaMemcpyHostToDevice));
		_offset = 0;
	}

	int getCollisions(bool self, struct _g_pairCCD &rets, int *time, REAL thickness);
	int getProximityConstraints(bool self, REAL mu, REAL mu_obs, REAL mrt, REAL mcs);
	int getImpacts(bool self, REAL mu, REAL mu_obs, _g_pair &vfPairs, _g_pair &eePairs, int &vfLen, int &eeLen);

	void destroy() {
		cudaFree(_dPairs);
		cudaFree(_dVF_EE);
		cudaFree(_dv);
		cudaFree(dist);
		cudaFree(_dIdx);
		cudaFree(CCD_res);
	}

	uint length() {
		uint dummy[] = { 0 };
		cutilSafeCall(cudaMemcpy(dummy, _dIdx, 1 * sizeof(uint), cudaMemcpyDeviceToHost));
		if (dummy[0] > _max_length)
			_max_length = dummy[0];

		return dummy[0];
	}

	void setLength(uint len) {
		cutilSafeCall(cudaMemcpy(_dIdx, &len, 1 * sizeof(uint), cudaMemcpyHostToDevice));
	}

	int maxLength() const { return _max_length; }
} g_pairCCD;
