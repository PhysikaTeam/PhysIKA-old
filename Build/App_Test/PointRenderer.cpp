#include <iostream>
#include "Physika_Core/Utilities/cuda_helper_math.h"
#include "PointRenderer.h"
#include "Vec.h"
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>  
#include "CUDAFunction.h"


PointRenderer::PointRenderer(Vectorold3f** _data, int* _n)
{
	data = _data;
	n = _n;
	mapping = NULL;
	marker = NULL;
}


PointRenderer::~PointRenderer(void)
{
}

void PointRenderer::Render( const Camera &camera )
{
	Color color;
// 	Vector3f color;
// 	float maxvalue = 0.0f;
// 	if (data != NULL)
// 	{
// 		for (int i = 0; i < n; i++)
// 		{
// 			if (data[i] > maxvalue)
// 			{
// 				maxvalue = data[i];
// 			}
// 		}
// 	}

	if (true) { // draw simple gl_points (fastest)
		glDisable(GL_LIGHTING);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
		glPointSize(4);
		glBegin(GL_POINTS);
		if (marker == NULL)
		{
			for (int i=0; i< *n; i++) {
				glColor4f(0.0f, 1.0f, 0.0f, 1.0f);
				glVertex3f((*data)[i].x, (*data)[i].y, (*data)[i].z);
			}
		}
		else
		{
			for (int i=0; i< *n; i++) {
				if ((*marker)[i])
				{
					// 			if (mapping != NULL)
					// 			{
					// 				color.set(120.0f*((mapping[i])/maxvalue)+120.0f, 1.0, 1.0);
					// 				color.HSVtoRGB();
					//				glColor3f(color.x, color.y, color.z);
					// 			}
					// 			else
					// 				glColor3f(posArr[i].x, posArr[i].y, posArr[i].z);

					//			if (MATERIALTYPE(attriArr[i]) != MATERIAL_FLUID)
					// 			{
					// 				glColor3f(0.8f, 0.8f, 0.8f);
					// 			}
					// 			else
					// 				glColor3f(0.0f, 1.0f, 0.0f);


					if (mapping != NULL)
					{
						color.HSVtoRGB((*mapping)[i]/(*green)*120.0f+120.0f, 1.0f, 1.0f);
						glColor4f(color.r, color.g, color.b, (*mapping)[i]/(*green));
					}
					else
						glColor4f(0.0f, 1.0f, 1.0f, 1.0f);

					glVertex3f((*data)[i].x, (*data)[i].y, (*data)[i].z);
				}
			}
		}
		
		glEnd();
	}
}

CudaPointRender::CudaPointRender(Array<float3>* pos)
{
	posBuf = pos;
	m_vbo = CreateVBO(NULL, pos->Size()*sizeof(float3));
	cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, m_vbo, cudaGraphicsMapFlagsWriteDiscard);

	m_clr_vbo = CreateVBO(NULL, pos->Size() * sizeof(float4));
	cudaGraphicsGLRegisterBuffer(&cuda_vbo_color, m_clr_vbo, cudaGraphicsMapFlagsWriteDiscard);
}

CudaPointRender::~CudaPointRender()
{

}


void CudaPointRender::Render(const Camera &camera)
{
	float3* dst = GetBuffer();
	float4* clr = GetColorBuffer();
	CUDAFunction::CopyData(dst, clr, *posBuf, *colorIndex);
	UnMapCudaBuffer();
	UnmapCudaColorBuffer();

	glColor3f(1, 1, 1);
	glPointSize(2);

	glBindBuffer(GL_ARRAY_BUFFER, m_clr_vbo);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(4, GL_FLOAT, 0, 0);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, 0);
	

	glDrawArrays(GL_POINTS, 0, posBuf->Size());

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glDisableClientState(GL_VERTEX_ARRAY);
}

unsigned int CudaPointRender::CreateVBO(void* data, unsigned int size)
{
	GLuint vbo;
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, size, data, GL_DYNAMIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	return vbo;
}

float3* CudaPointRender::GetBuffer()
{
	float3* pBuf;	size_t numBytes;	cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&pBuf, &numBytes, cuda_vbo_resource);

	return pBuf;
}

void CudaPointRender::UnmapCudaColorBuffer()
{
	cudaGraphicsUnmapResources(1, &cuda_vbo_color, 0);
}

void CudaPointRender::UnMapCudaBuffer()
{
	cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

float4* CudaPointRender::GetColorBuffer()
{
	float4* clrBuf;	size_t numBytes;

	cudaGraphicsMapResources(1, &cuda_vbo_color, 0);
	cudaGraphicsResourceGetMappedPointer((void**)&clrBuf, &numBytes, cuda_vbo_color);

	return clrBuf;
}
