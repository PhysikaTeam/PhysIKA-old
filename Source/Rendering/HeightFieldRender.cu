#pragma once

#include <GL/glew.h>
#include "HeightFieldRender.h"
#include "Framework/Topology/HeightField.h"
#include "Core/Vector.h"
#include "Core/Utility.h"
#include "Framework/Framework/Node.h"
#include "OpenGLContext.h"
#include "Color.h"

namespace PhysIKA {
IMPLEMENT_CLASS(HeightFieldRenderModule)

HeightFieldRenderModule::HeightFieldRenderModule()
    : VisualModule()
    , m_mode(HeightFieldRenderModule::Instance)
    , m_color(Vector3f(0.8, 0.8, 0.8))
{
}

HeightFieldRenderModule::~HeightFieldRenderModule()
{
}

bool HeightFieldRenderModule::initializeImpl()
{
    Node* parent = getParent();
    if (parent == NULL)
    {
        Log::sendMessage(Log::Error, "Should insert this module into a node!");
        return false;
    }

    auto pSet = TypeInfo::CastPointerDown<HeightField<DataType3f>>(parent->getTopologyModule());
    if (pSet == nullptr)
    {
        Log::sendMessage(Log::Error, "HeightFieldRenderModule: The topology module is not supported!");
        return false;
    }

    Log::sendMessage(Log::Info, "HeightFieldRenderModule successfully initialized!");
}

__global__ void PRM_MappingColor(
    DeviceArray<glm::vec3> color,
    DeviceArray<Vector3f>  index,
    float                  minIndex,
    float                  maxIndex)
{
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tId >= color.size())
        return;

    float index_i = index[tId].norm();

    index_i = index_i > maxIndex ? maxIndex : index_i;
    index_i = index_i < minIndex ? minIndex : index_i;

    float a = (index_i - minIndex) / (maxIndex - minIndex);

    Color hsv;
    hsv.HSVtoRGB(240, 1 - a, 1);

    color[tId] = glm::vec3(hsv.r, hsv.g, hsv.b);
}

__global__ void PRM_MappingColor(
    DeviceArray<glm::vec3> color,
    DeviceArray<float>     index,
    float                  refV,
    float                  minIndex,
    float                  maxIndex)
{
    int tId = threadIdx.x + (blockIdx.x * blockDim.x);
    if (tId >= color.size())
        return;

    float index_i = index[tId];

    index_i = index_i > maxIndex ? maxIndex : index_i;
    index_i = index_i < minIndex ? minIndex : index_i;

    float a = (index_i - refV) / (maxIndex - minIndex);

    Color hsv;
    hsv.HSVtoRGB(a * 120 + 120, 1, 1);

    color[tId] = glm::vec3(hsv.r, hsv.g, hsv.b);
}

__global__ void SetupTriangles(
    DeviceArray<float3>  vertices,
    DeviceArray<float3>  normals,
    DeviceArray<float3>  colors,
    DeviceArray2D<float> heights,
    DeviceArray2D<float> terrain,
    float                dx,
    float                dz,
    float3               origin,
    float3               colorWater,
    float3               colorTerrain)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i < heights.Nx() - 1 && j < heights.Ny() - 1)
    {
        int id = i + j * (heights.Nx() - 1);

        float3 v1 = origin + make_float3(i * dx, heights(i, j), j * dz);
        float3 v2 = origin + make_float3((i + 1) * dx, heights(i + 1, j), j * dz);
        float3 v3 = origin + make_float3(i * dx, heights(i, j + 1), (j + 1) * dz);
        float3 v4 = origin + make_float3((i + 1) * dx, heights(i + 1, j + 1), (j + 1) * dz);

        //float3 v1 = origin + make_float3(i*dx, 0.5f, j*dz);
        //float3 v2 = origin + make_float3((i + 1)*dx, 0.5f, ((j + 1))*dz);
        //float3 v3 = origin + make_float3(i*dx, 0.5f, (j + 1)*dz);
        //float3 v4 = origin + make_float3((i + 1)*dx, 0.5f, (j + 1)*dz);

        vertices[3 * (2 * id) + 0] = v1;
        vertices[3 * (2 * id) + 1] = v2;
        vertices[3 * (2 * id) + 2] = v3;

        float3 triN1 = cross(v2 - v1, v3 - v1);
        triN1        = normalize(triN1);

        normals[3 * (2 * id) + 0] = triN1;
        normals[3 * (2 * id) + 1] = triN1;
        normals[3 * (2 * id) + 2] = triN1;

        colors[3 * (2 * id) + 0] = heights(i, j) > terrain(i, j) ? colorWater : colorTerrain;
        colors[3 * (2 * id) + 1] = heights(i + 1, j) > terrain(i + 1, j) ? colorWater : colorTerrain;
        colors[3 * (2 * id) + 2] = heights(i, j + 1) > terrain(i, j + 1) ? colorWater : colorTerrain;

        vertices[3 * (2 * id) + 3] = v3;
        vertices[3 * (2 * id) + 4] = v2;
        vertices[3 * (2 * id) + 5] = v4;

        float3 triN2 = cross(v2 - v3, v4 - v3);
        triN2        = normalize(triN2);

        normals[3 * (2 * id) + 3] = triN2;
        normals[3 * (2 * id) + 4] = triN2;
        normals[3 * (2 * id) + 5] = triN2;

        colors[3 * (2 * id) + 3] = heights(i, j + 1) > terrain(i, j + 1) ? colorWater : colorTerrain;
        colors[3 * (2 * id) + 4] = heights(i + 1, j) > terrain(i + 1, j) ? colorWater : colorTerrain;
        colors[3 * (2 * id) + 5] = heights(i + 1, j + 1) > terrain(i + 1, j + 1) ? colorWater : colorTerrain;
    }
}

void HeightFieldRenderModule::updateRenderingContext()
{
    Node* parent = getParent();
    if (parent == NULL)
    {
        Log::sendMessage(Log::Error, "Should insert this module into a node!");
        return;
    }

    auto hf = TypeInfo::CastPointerDown<HeightField<DataType3f>>(parent->getTopologyModule());
    if (hf == nullptr)
    {
        Log::sendMessage(Log::Error, "HeightFieldRenderModule: The topology module is not supported!");
        return;
    }

    auto heights        = hf->getHeights();
    auto terrain        = hf->getTerrain();
    int  numOfTriangles = (heights.Nx() - 1) * (heights.Ny() - 1) * 2;
    //printf("heights nx is %d, ny is %d\n", heights.Nx(), heights.Ny());
    vertices.resize(3 * numOfTriangles);
    normals.resize(3 * numOfTriangles);
    colors.resize(3 * numOfTriangles);

    uint3 total_size;
    total_size.x = heights.Nx() - 1;
    total_size.y = heights.Ny() - 1;
    total_size.z = 1;

    auto ori = hf->getOrigin();

    cuExecute3D(total_size, SetupTriangles, vertices, normals, colors, heights, terrain, hf->getDx(), hf->getDz(), make_float3(ori[0], ori[1], ori[2]), make_float3(0.0, 0.2, 1), make_float3(1.0, 0.0, 0.0));
    char str[200];
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        //sprintf(str, "CUDA error: %d : %s at %s:%d \n", err, cudaGetErrorString(err), __FILE__, __LINE__);
        std::string str = cudaGetErrorString(err);
        sprintf("CUDA error:  %s \n", str.c_str());
        throw std::runtime_error(std::string(str));
    }

    if (m_triangleRender == nullptr)
    {
        m_triangleRender = std::make_shared<TriangleRender>();
    }

    if (m_triangleRender->numberOfTrianlges() != numOfTriangles)
    {
        m_triangleRender->resize(numOfTriangles);
    }

    m_triangleRender->setVertexArray(vertices);
    m_triangleRender->setColorArray(colors);
    m_triangleRender->setNormalArray(normals);
}

void HeightFieldRenderModule::display()
{
    glMatrixMode(GL_MODELVIEW_MATRIX);
    glPushMatrix();

    glRotatef(m_rotation.x(), m_rotation.y(), m_rotation.z(), m_rotation.w());
    glTranslatef(m_translation[0], m_translation[1], m_translation[2]);
    glScalef(m_scale[0], m_scale[1], m_scale[2]);

    if (m_triangleRender != nullptr)
        m_triangleRender->display();

    glPopMatrix();
}

void HeightFieldRenderModule::setRenderMode(RenderMode mode)
{
    m_mode = mode;
}

void HeightFieldRenderModule::setColor(Vector3f color)
{
    m_color = color;
}

void HeightFieldRenderModule::setReferenceColor(float v)
{
    m_refV = v;
}

}  // namespace PhysIKA