#include "TriangleSet.h"
#include <fstream>
#include <iostream>
#include <sstream>
#include "Core/Utility.h"

namespace PhysIKA {
template <typename TDataType>
TriangleSet<TDataType>::TriangleSet()
    : EdgeSet<TDataType>()
{
    std::vector<Coord>    positions;
    std::vector<Coord>    normals;
    std::vector<Triangle> triangles;
    float                 dx = 0.1;
    int                   Nx = 11;
    int                   Nz = 11;

    for (int k = 0; k < Nz; k++)
    {
        for (int i = 0; i < Nx; i++)
        {
            positions.push_back(Coord(Real(i * dx), Real(0.0), Real(k * dx)));
            normals.push_back(Coord(0, 1, 0));
            if (k < Nz - 1 && i < Nx - 1)
            {
                Triangle tri1(i + k * Nx, i + 1 + k * Nx, i + 1 + (k + 1) * Nx);
                Triangle tri2(i + k * Nx, i + 1 + (k + 1) * Nx, i + (k + 1) * Nx);
                triangles.push_back(tri1);
                triangles.push_back(tri2);
            }
        }
    }
    this->setPoints(positions);
    this->setNormals(normals);
    this->setTriangles(triangles);
}

template <typename TDataType>
TriangleSet<TDataType>::~TriangleSet()
{
}

template <typename TDataType>
void TriangleSet<TDataType>::updatePointNeighbors()
{
}

template <typename TDataType>
bool TriangleSet<TDataType>::initializeImpl()
{

    return true;
}

template <typename TDataType>
void TriangleSet<TDataType>::setTriangles(std::vector<Triangle>& triangles)
{
    m_triangls.resize(triangles.size());
    Function1Pt::copy(m_triangls, triangles);
}

template <typename TDataType>
void TriangleSet<TDataType>::loadObjFile(std::string filename)
{
    if (filename.size() < 5 || filename.substr(filename.size() - 4) != std::string(".obj"))
    {
        std::cerr << "Error: Expected OBJ file with filename of the form <name>.obj.\n";
        exit(-1);
    }

    std::ifstream infile(filename);
    if (!infile)
    {
        std::cerr << "Failed to open. Terminating.\n";
        exit(-1);
    }

    int                   ignored_lines = 0;
    std::string           line;
    std::vector<Coord>    vertList;
    std::vector<Coord>    normalList;
    std::vector<Triangle> faceList;
    while (!infile.eof())
    {
        std::getline(infile, line);

        //.obj files sometimes contain vertex normals indicated by "vn"
        if (line.substr(0, 1) == std::string("v") && line.substr(0, 2) != std::string("vn"))
        {
            std::stringstream data(line);
            char              c;
            Coord             point;
            data >> c >> point[0] >> point[1] >> point[2];
            vertList.push_back(point);
        }
        else if (line.substr(0, 1) == std::string("f"))
        {
            std::stringstream data(line);
            char              c;
            int               v0, v1, v2;
            data >> c >> v0 >> v1 >> v2;
            faceList.push_back(Triangle(v0 - 1, v1 - 1, v2 - 1));
        }
        else if (line.substr(0, 2) == std::string("vn"))
        {
            std::stringstream data(line);
            char              c;
            Coord             normal;
            data >> c >> normal[0] >> normal[1] >> normal[2];
            normalList.push_back(normal);
        }
        else
        {
            ++ignored_lines;
        }
    }
    infile.close();

    if (normalList.size() != vertList.size())
    {
        normalList.resize(vertList.size());
    }

    this->setPoints(vertList);
    this->setNormals(normalList);
    setTriangles(faceList);
}

}  // namespace PhysIKA