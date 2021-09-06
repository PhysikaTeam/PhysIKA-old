#pragma once

#include "FEMGeometryTetVtk.h"
#include "FEMGeometryTetParameter.h"

/**
 * @brief FEM Geometry SimpleVoxel
 * 
 */
class SimpleVoxel
{
public:
    size_t              if_cell_cube;
    size_t              cell_index;
    size_t              coord_index[3];
    size_t              vertice_index[8];
    std::vector<size_t> adjacent_voxels;

    /**
     * @brief Construct a new Simple Voxel object
     * 
     * @param input_index 
     * @param index 
     */
    SimpleVoxel(const size_t (&input_index)[3], const size_t& index);

    /**
     * @brief Write the object to a file
     * 
     * @param file 
     */
    void voxel_write_to_file(std::ofstream& file);

    /**
     * @brief Write the tetrahedron oject to a file
     * 
     * @param file 
     * @param vertice 
     */
    void tet_write_to_file(std::ofstream& file, const std::vector<cxz::MyVector3>& vertice);
};

/**
 * @brief FEM Geometry VoxelMesh
 * 
 */
class VoxelMesh
{
public:
    double                           volume;
    cxz::MyVector3                   normal_voxel;
    cxz::MyVector3                   relative_coord_origin;
    size_t                           voxel_number;
    size_t                           flag_vertice_number;
    size_t                           vertice_number;
    std::vector<SimpleVoxel>         voxel;
    std::vector<cxz::MyVector3>      vertice;
    std::vector<std::vector<size_t>> vertice_voxel;
    std::vector<std::vector<size_t>> bcc_segment;
    std::vector<std::vector<size_t>> bcc_tet;
    std::vector<size_t>              true_bcc_tet;

    /**
     * @brief Construct a new Voxel Mesh object
     * 
     * @param vtk 
     */
    VoxelMesh(SimpleVtk& vtk);

    /**
     * @brief Initialize the VoxelMesh object
     * 
     * @param vtk 
     * @param part 
     */
    void init_voxel_mesh(SimpleVtk& vtk, std::vector<size_t>& part);

    /**
     * @brief Generate the tetrahedron
     * 
     * @param vtk 
     */
    void bcc_generate_tet(const SimpleVtk& vtk);

    /**
     * @brief Write the hexahedron data to a vtk file
     * 
     * @param path 
     */
    void write_hex_to_vtk_file(const char* const path = "hex.vtk");

    /**
     * @brief Write the tetrahedron data to a vtk file
     * 
     * @param path 
     */
    void write_tet_to_vtk_file(const char* const path = "tet.vtk");

    /**
     * @brief Write the bbc tetrahedron data to a vtk file
     * 
     * @param path 
     */
    void write_bcc_tet_to_vtk_file(const char* const path = "tet_final.vtk");

    /**
     * @brief Write all bbc tetrahedron data to a vtk file
     * 
     * @param path 
     */
    void write_all_bcc_tet_to_vtk_file(const char* const path = "tet_simple.vtk");

private:
    /**
     * @brief Initialize the parameters of the object
     * 
     * @param vtk 
     * @return int 
     */
    int  init_parameter(const SimpleVtk& vtk);

    /**
     * @brief Initialize the voxel data
     * 
     * @param vtk 
     * @param part 
     */
    void init_vtk_to_voxel(const SimpleVtk& vtk, std::vector<size_t>& part);

    /**
     * @brief Initialize the adjacent voxels
     * 
     * @param vtk 
     */
    void init_adjacent_voxels(const SimpleVtk& vtk);

    /**
     * @brief Gemerate the order
     * 
     * @param order_list 
     */
    void generate_order(std::vector<size_t>& order_list);

    /**
     * @brief Initialize the vertice coordinates
     * 
     * @param order_list 
     */
    void init_vertice_coord(const std::vector<size_t>& order_list);

    /**
     * @brief Compute the distance between two SimpleVoxel objects
     * 
     * @param voxel_1 
     * @param voxel_2 
     * @return size_t 
     */
    size_t compute_index_distance(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2);

    /**
     * @brief Judge whether nearby
     * 
     * @param voxel_1 
     * @param voxel_2 
     * @param vtk 
     * @return true 
     * @return false 
     */
    bool   judge_nearby(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2, const SimpleVtk& vtk);

    /**
     * @brief Judge the nearby cell
     * 
     * @param voxel_1 
     * @param voxel_2 
     * @param vtk 
     * @return true 
     * @return false 
     */
    bool   judge_nearby_cell(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2, const SimpleVtk& vtk);

    /**
     * @brief Judge whether it is the same face
     * 
     * @param face_1 
     * @param face_2 
     * @return true 
     * @return false 
     */
    bool   judge_same_face(const SimpleFace& face_1, const SimpleFace& face_2);

    /**
     * @brief Judge whether it is the corresponding vertex index
     * 
     * @param voxel_1 
     * @param voxel_2 
     * @param index1 
     * @param index2 
     * @return true 
     * @return false 
     */
    bool   judge_corresponding_vertex_index(const size_t& voxel_1, const size_t& voxel_2, std::vector<size_t>& index1, std::vector<size_t>& index2);

    /**
     * @brief Get the coordinates by index
     * 
     * @param coord 
     * @param index 
     */
    void           coord_to_index(const cxz::MyVector3& coord, size_t (&index)[3]);

    /**
     * @brief Get the coordinates by index
     * 
     * @param coord_index 
     * @param point_index 
     * @return cxz::MyVector3 
     */
    cxz::MyVector3 index_to_coord(const size_t (&coord_index)[3], const size_t& point_index);

    /**
     * @brief Get the orientation between two voxel objects
     * 
     * @param voxel_1 
     * @param voxel_2 
     * @return size_t 
     */
    size_t orientation_between_voxel(const SimpleVoxel& voxel_1, const SimpleVoxel& voxel_2);

    /**
     * @brief Insert the bcc center
     * 
     */
    void   insert_bcc_center();

    /**
     * @brief Initialize the bcc segment
     * 
     */
    void   init_bcc_segment();

    /**
     * @brief Initialize the bcc tetrahedron
     * 
     */
    void   init_bcc_tet();

    /**
     * @brief Initialize the true bcc tetrahedron
     * 
     * @param vtk 
     */
    void   init_true_bcc_tet(const SimpleVtk& vtk);

    /**
     * @brief Judge whether the cells intersect
     * 
     * @param tet_index 
     * @param cell_index 
     * @param vtk 
     * @return true 
     * @return false 
     */
    bool   judge_tet_cell_intersect(const size_t& tet_index, const size_t& cell_index, const SimpleVtk& vtk);
};
