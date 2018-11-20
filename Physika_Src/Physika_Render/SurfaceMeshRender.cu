#pragma once

#include <GL/glew.h>
#include "SurfaceMeshRender.h"
#include "Physika_Framework/Topology/TriangleSet.h"
#include "Physika_Core/Vectors/vector.h"
#include "Physika_Framework/Framework/Node.h"
#include "Physika_Core/Utilities/cuda_helper_math.h"
#include "OpenGLContext.h"


namespace Physika
{
	IMPLEMENT_CLASS(SurfaceMeshRender)

		SurfaceMeshRender::SurfaceMeshRender()
		: VisualModule()
		, m_color(Vector3f(0.0f, 0.0, 1.0f))
	{
	}

	SurfaceMeshRender::~SurfaceMeshRender()
	{
	}

	bool SurfaceMeshRender::initializeImpl()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return false;
		}

		auto triSet = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(parent->getTopologyModule());
		if (triSet == nullptr)
		{
			Log::sendMessage(Log::Error, "TriangleModule: The topology module is not supported!");
			return false;
		}

		if (!triSet->isInitialized())
		{
			triSet->initialize();
		}

//		point_render_util = std::make_shared<PointRenderUtil>();

		auto triangles = triSet->getTriangles();
// 
//		m_lineRender->setColors(color);

		m_triangleRender = std::make_shared<TriangleRender>();
		m_triangleRender->resize(triangles->size());
	}

	__global__ void SetupTriangles(
		DeviceArray<float3> originVerts,
		DeviceArray<float3> vertices,
		DeviceArray<float3> normals,
		DeviceArray<float3> colors,
		DeviceArray<TopologyModule::Triangle> triangles
		)
	{
		int pId = threadIdx.x + (blockIdx.x * blockDim.x);
		if (pId >= triangles.size()) return;

		TopologyModule::Triangle tri = triangles[pId];
		float3 v1 = originVerts[tri[0]];
		float3 v2 = originVerts[tri[1]];
		float3 v3 = originVerts[tri[2]];

		vertices[3 * pId + 0] = v1;
		vertices[3 * pId + 1] = v2;
		vertices[3 * pId + 2] = v3;

		float3 triN = cross(v2-v1, v3-v1);
		triN = normalize(triN);

		normals[3 * pId + 0] = triN;
		normals[3 * pId + 1] = triN;
		normals[3 * pId + 2] = triN;

		colors[3 * pId + 0] = make_float3(0.0f, 1.0f, 0.0f);
		colors[3 * pId + 1] = make_float3(0.0f, 1.0f, 0.0f);
		colors[3 * pId + 2] = make_float3(0.0f, 1.0f, 0.0f);
	}

	void SurfaceMeshRender::updateRenderingContext()
	{
		Node* parent = getParent();
		if (parent == NULL)
		{
			Log::sendMessage(Log::Error, "Should insert this module into a node!");
			return;
		}

		auto triSet = TypeInfo::CastPointerDown<TriangleSet<DataType3f>>(parent->getTopologyModule());
		if (triSet == nullptr)
		{
			Log::sendMessage(Log::Error, "TriangleModule: The topology module is not supported!");
			return;
		}

		auto verts = (DeviceArray<float3>*)triSet->getPoints();
		auto triangles = triSet->getTriangles();

		DeviceArray<float3> vertices;
		DeviceArray<float3> normals;
		DeviceArray<float3> colors;

		vertices.resize(3 * triangles->size());
		normals.resize(3 * triangles->size());
		colors.resize(3 * triangles->size());

		uint pDims = cudaGridSize(triangles->size(), BLOCK_SIZE);

		SetupTriangles << <pDims, BLOCK_SIZE >> >(*verts, vertices, normals, colors, *triangles);


		m_triangleRender->setVertexArray(vertices);
		m_triangleRender->setColorArray(colors);
		m_triangleRender->setNormalArray(normals);
	}

	void SurfaceMeshRender::display()
	{
		glMatrixMode(GL_MODELVIEW_MATRIX);
		glPushMatrix();

		glRotatef(m_rotation.x(), m_rotation.y(), m_rotation.z(), m_rotation.w());
		glTranslatef(m_translation[0], m_translation[1], m_translation[2]);
		glScalef(m_scale[0], m_scale[1], m_scale[2]);

		m_triangleRender->display();

		glPopMatrix();
	}

	void SurfaceMeshRender::setColor(Vector3f color)
	{
		m_color = color;
	}

}