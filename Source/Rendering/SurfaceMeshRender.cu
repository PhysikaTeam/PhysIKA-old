#pragma once
#include <GL/glew.h>
#include "SurfaceMeshRender.h"
#include "Framework/Topology/TriangleSet.h"
#include "Core/Vector.h"
#include "Framework/Framework/Node.h"
#include "Core/Utility.h"
#include "OpenGLContext.h"

namespace PhysIKA {
IMPLEMENT_CLASS(SurfaceMeshRender)

SurfaceMeshRender::SurfaceMeshRender()
    : VisualModule()
    , m_color(Vector3f(0.2f, 0.3, 0.0f))
{
}

SurfaceMeshRender::~SurfaceMeshRender()
{
    vertices.release();
    normals.release();
    colors.release();
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

    //		point_render_util = std::make_shared<PointRenderUtil>();

    auto triangles = triSet->getTriangles();

    m_triangleRender = std::make_shared<TriangleRender>();
    m_triangleRender->resize(triangles->size());

    vertices.resize(3 * triangles->size());
    normals.resize(3 * triangles->size());
    colors.resize(3 * triangles->size());
}

__global__ void SetupTriangles(
    DeviceArray<float3>                   originVerts,
    DeviceArray<float3>                   vertices,
    DeviceArray<float3>                   normals,
    DeviceArray<float3>                   colors,
    DeviceArray<TopologyModule::Triangle> triangles,
    float3                                color)
{
    int pId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (pId >= triangles.size())
        return;

    TopologyModule::Triangle tri = triangles[pId];
    float3                   v1  = originVerts[tri[0]];
    float3                   v2  = originVerts[tri[1]];
    float3                   v3  = originVerts[tri[2]];

    vertices[3 * pId + 0] = v1;
    vertices[3 * pId + 1] = v2;
    vertices[3 * pId + 2] = v3;

    float3 triN = cross(v2 - v1, v3 - v1);
    triN        = normalize(triN);

    normals[3 * pId + 0] = triN;
    normals[3 * pId + 1] = triN;
    normals[3 * pId + 2] = triN;

    colors[3 * pId + 0] = color;
    colors[3 * pId + 1] = color;
    colors[3 * pId + 2] = color;
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

    auto verts     = triSet->getPoints();
    auto triangles = triSet->getTriangles();

    uint pDims = cudaGridSize(triangles->size(), BLOCK_SIZE);

    DeviceArray<float3>* fverts = ( DeviceArray<float3>* )&verts;
    SetupTriangles<<<pDims, BLOCK_SIZE>>>(*fverts, vertices, normals, colors, *triangles, make_float3(m_color[0], m_color[1], m_color[2]));

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

}  // namespace PhysIKA