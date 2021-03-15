#pragma once

typedef struct {
	uint  numFace, numVert;
	REAL3 *_dx, *_dx0;
	tri3f *_df;
	g_box *_dfBx;

	// init function
	void init()
	{
		numFace = 0;
		numVert = 0;
		_dx0 = _dx = NULL;
		_df = NULL;
		_dfBx = NULL;
	}

	void destroy()
	{
		if (_dx == NULL) return;

		checkCudaErrors(cudaFree(_dx));
		checkCudaErrors(cudaFree(_dx0));
		checkCudaErrors(cudaFree(_df));
		checkCudaErrors(cudaFree(_dfBx));
	}

	void computeWSdata(REAL thickness, bool ccd);
} g_mesh;
