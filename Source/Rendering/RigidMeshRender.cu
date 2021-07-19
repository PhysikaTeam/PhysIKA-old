#pragma once
#include <GL/glew.h>
#include "RigidMeshRender.h"
#include "Framework/Topology/TriangleSet.h"
#include "Core/Vector.h"
#include "Framework/Framework/Node.h"
#include "Core/Utility.h"
#include "OpenGLContext.h"

namespace PhysIKA {
IMPLEMENT_CLASS(RigidMeshRender)

RigidMeshRender::RigidMeshRender(std::shared_ptr<Frame<DataType3f>> frame)
    : VisualModule()
    , m_color(Vector3f(0.2f, 0.3, 0.0f))
    , m_meshFrame(frame)
{
}

__global__ void SetupRigidTriangles(
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

RigidMeshRender::~RigidMeshRender()
{
    vertices.release();
    normals.release();
    colors.release();
}

bool RigidMeshRender::initializeImpl()
{
    Node* parent = getParent();
    if (parent == NULL)
    {
        Log::sendMessage(Log::Error, "Should insert this module into a node!");
        return false;
    }

    auto triSet = TypeInfo::cast<TriangleSet<DataType3f>>(parent->getTopologyModule());
    if (triSet == nullptr)
    {
        Log::sendMessage(Log::Error, "TriangleModule: The topology module is not supported!");
        return false;
    }

    //		point_render_util = std::make_shared<PointRenderUtil>();

    auto triangles = triSet->getTriangles();

    m_triangleRender = std::make_shared<RigidTriangleRender>();
    m_triangleRender->resize(triangles->size());

    vertices.resize(3 * triangles->size());
    normals.resize(3 * triangles->size());
    colors.resize(3 * triangles->size());

    //updateRenderingContext();
    {
        auto                 verts     = triSet->getPoints();
        auto                 triangles = triSet->getTriangles();
        uint                 pDims     = cudaGridSize(triangles->size(), BLOCK_SIZE);
        DeviceArray<float3>* fverts    = ( DeviceArray<float3>* )&verts;
        SetupRigidTriangles<<<pDims, BLOCK_SIZE>>>(*fverts, vertices, normals, colors, *triangles, make_float3(m_color[0], m_color[1], m_color[2]));

        m_triangleRender->setVertexArray(vertices);
        m_triangleRender->setColorArray(colors);
        m_triangleRender->setNormalArray(normals);
    }
}

void RigidMeshRender::updateRenderingContext()
{
}

void RigidMeshRender::display()
{

    glMatrixMode(GL_MODELVIEW_MATRIX);
    glPushMatrix();

    // Set uniform global transformation.
    Vector3f rotateAxis;
    float    rotateRadian;
    m_rotation.getRotation(rotateRadian, rotateAxis);
    glRotatef(rotateRadian, rotateAxis[0], rotateAxis[1], rotateAxis[2]);
    //glRotatef(m_rotation.x(), m_rotation.y(), m_rotation.z(), m_rotation.w());
    glTranslatef(m_translation[0], m_translation[1], m_translation[2]);
    glScalef(m_scale[0], m_scale[1], m_scale[2]);

    // Update triangle mesh transformation
    m_triangleRender->setRotation(Quaternion<float>(m_meshFrame->getOrientation()));
    m_triangleRender->setTranslatioin(m_meshFrame->getCenter());

    // Display triangle mesh.
    m_triangleRender->display();

    glPopMatrix();
}

void RigidMeshRender::setColor(Vector3f color)
{
    m_color = color;
}

}  // namespace PhysIKA