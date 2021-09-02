#ifndef _CXZ_VTK_H_
#define _CXZ_VTK_H_

#include <vector>
#include <iostream>
#include <fstream>
#include "../inc/vector.h"
#include "../inc/parameter.h"

class SimpleVertex
{
public:
    cxz::MyVector3 vertex;
    SimpleVertex(const double& p1, const double& p2, const double& p3);

private:
    bool sign;
    void delete_vertex();
};

class SimpleFace
{
public:
    size_t              vertex_number;
    std::vector<size_t> vertice_index;
    bool                if_normal;

    SimpleFace(const std::vector<size_t>& input_vertice, const std::vector<SimpleVertex>& vertice);
};

class SimpleCell
{
public:
    size_t                  face_number;
    size_t                  vertex_number;
    std::vector<SimpleFace> faces;
    std::vector<size_t>     vertex_index;
    size_t                  if_normal;
    double                  volume;

public:
    SimpleCell();
    SimpleCell(const std::vector<SimpleFace>& input_faces);
    void   clear();
    void   AddFace(const SimpleFace& input_face);
    double compute_cell_volume(const std::vector<SimpleVertex>& vertice);

    size_t vtk_format_number();
    void   cell_write_to_file(std::ofstream& file);
};

class SimpleVtk
{
public:
    std::vector<SimpleVertex> vertice;
    std::vector<SimpleCell>   cells;
    size_t                    vertice_number;
    size_t                    cells_number;
    double                    volume;
    double                    limit_volume;

    SimpleVtk();
    int read_file_vtk42(const char* const path);
    int write_to_file(const char* const path = "cell.vtk");
    int write_to_file_part(const std::vector<size_t>& cell_index, const char* const path = "cell.vtk");
};

#endif