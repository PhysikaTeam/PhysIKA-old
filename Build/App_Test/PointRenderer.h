#ifndef MFD_POINTRENDER_H
#define MFD_POINTRENDER_H

#include "IRenderer.h"
#include "Vec.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Typedef.h"


using namespace Physika;

class PointRenderer : public IRenderer
{
public:
	PointRenderer() { data = NULL; n = 0; }
	PointRenderer(Vectorold3f** _data, int* _n);
	~PointRenderer(void);

	virtual void Render(const Camera &camera);

	void SetMapping(float** _mapping, float* _g) { mapping = _mapping; green = _g; }
	void SetMarker(bool** _marker) { marker = _marker; }

private:
	Vectorold3f** data;
	float** mapping;
	float* green;
	bool** marker;
	int* n;
};


class CudaPointRender : public IRenderer
{
public:
	CudaPointRender(Array<float3>* pos);
	~CudaPointRender();

	virtual void Render(const Camera &camera);
	unsigned int CreateVBO(void* data, unsigned int size);
	void UnMapCudaBuffer();
	float3* GetBuffer();
	void UnmapCudaColorBuffer();
	float4* GetColorBuffer();
	void SetColorIndex(Array<float>* ind) { colorIndex = ind; }

private:
	Array<float3>* posBuf;
	Array<float>* colorIndex;
	int m_vbo;
	int m_clr_vbo;
	struct cudaGraphicsResource *cuda_vbo_resource;
	struct cudaGraphicsResource* cuda_vbo_color;
};


#endif