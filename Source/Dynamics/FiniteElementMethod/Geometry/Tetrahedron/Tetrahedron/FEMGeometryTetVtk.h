#pragma once

#include <vector>
#include <iostream>
#include <fstream>
#include "FEMGeometryTetVector.h"
#include "FEMGeometryTetParameter.h"

/**
 * @brief FEM Geometry SimpleVertex
 * 
 */
class SimpleVertex
{
public:
    cxz::MyVector3 vertex;
    /**
     * @brief Construct a new Simple Vertex object
     * 
     * @param p1 
     * @param p2 
     * @param p3 
     */
    SimpleVertex(const double& p1, const double& p2, const double& p3);

private:
    bool sign;
    /**
     * @brief Delete the Vertex object
     * 
     */
    void delete_vertex();
};

/**
 * @brief FEM Geometry SimpleFace
 * 
 */
class SimpleFace
{
public:
    size_t              vertex_number;
    std::vector<size_t> vertice_index;
    bool                if_normal;

    /**
     * @brief Construct a new Simple Face object
     * 
     * @param input_vertice 
     * @param vertice 
     */
    SimpleFace(const std::vector<size_t>& input_vertice, const std::vector<SimpleVertex>& vertice);
};

/**
 * @brief FEM Geometry SimpleCell
 * 
 */
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
    /**
     * @brief Construct a new SimpleCell object
     * 
     */
    SimpleCell();

    /**
     * @brief Construct a new SimpleCell object
     * 
     * @param input_faces 
     */
    SimpleCell(const std::vector<SimpleFace>& input_faces);

    /**
     * @brief Clear the SimpleCell Object
     * 
     */
    void   clear();

    /**
     * @brief Add face to the object
     * 
     * @param input_face 
     */
    void   AddFace(const SimpleFace& input_face);

    /**
     * @brief Compute the volume of the cell
     * 
     * @param vertice 
     * @return double 
     */
    double compute_cell_volume(const std::vector<SimpleVertex>& vertice);

    /**
     * @brief Get the number in vtk format
     * 
     * @return size_t 
     */
    size_t vtk_format_number();

    /**
     * @brief Write the cell data to a file
     * 
     * @param file 
     */
    void   cell_write_to_file(std::ofstream& file);
};

/**
 * @brief FEM Geometry SimpleVtk
 * 
 */
class SimpleVtk
{
public:
    std::vector<SimpleVertex> vertice;
    std::vector<SimpleCell>   cells;
    size_t                    vertice_number;
    size_t                    cells_number;
    double                    volume;
    double                    limit_volume;

    /**
     * @brief Construct a new Simple Vtk object
     * 
     */
    SimpleVtk();

    /**
     * @brief Read data from a vtk file
     * 
     * @param path 
     * @return int 
     */
    int read_file_vtk42(const char* const path);

    /**
     * @brief Write the data to a vtk file
     * 
     * @param path 
     * @return int 
     */
    int write_to_file(const char* const path = "cell.vtk");

    /**
     * @brief write the data to a part of a vtk file
     * 
     * @param cell_index 
     * @param path 
     * @return int 
     */
    int write_to_file_part(const std::vector<size_t>& cell_index, const char* const path = "cell.vtk");
};
