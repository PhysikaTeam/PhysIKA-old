#ifndef _CXZ_VOXEL_H_
#define _CXZ_VOXEL_H_

#include "../inc/vtk.h"
#include "../inc/parameter.h"

class SimpleVoxel
{
public:
    size_t if_cell_cube;
    size_t cell_index;
    size_t coord_index[3];
    size_t vertice_index[8];
    std::vector<size_t> adjacent_voxels;

    SimpleVoxel(const size_t (&input_index)[3],const size_t &index);
    void voxel_write_to_file(std::ofstream &file);
    void tet_write_to_file(std::ofstream &file,const std::vector<cxz::MyVector3> &vertice);
};

class VoxelMesh
{
public:
    double volume;
    cxz::MyVector3 normal_voxel;
    cxz::MyVector3 relative_coord_origin;
    size_t voxel_number;
    size_t flag_vertice_number;
    size_t vertice_number;
    std::vector<SimpleVoxel> voxel;
    std::vector<cxz::MyVector3> vertice;
    std::vector<std::vector<size_t>> vertice_voxel;
    std::vector<std::vector<size_t>> bcc_segment;
    std::vector<std::vector<size_t>> bcc_tet;
    std::vector<size_t> true_bcc_tet;
    VoxelMesh(SimpleVtk &vtk);
    void init_voxel_mesh(SimpleVtk &vtk,std::vector<size_t> &part);
    void bcc_generate_tet(const SimpleVtk &vtk);
    void write_hex_to_vtk_file(const char* const path = "hex.vtk");
    void write_tet_to_vtk_file(const char* const path = "tet.vtk");
    void write_bcc_tet_to_vtk_file(const char* const path = "tet_final.vtk");
    void write_all_bcc_tet_to_vtk_file(const char* const path = "tet_simple.vtk");
private:
    int init_parameter(const SimpleVtk &vtk);    
    void init_vtk_to_voxel(const SimpleVtk &vtk,std::vector<size_t> &part);
    void init_adjacent_voxels(const SimpleVtk &vtk);
    void generate_order(std::vector<size_t> &order_list);
    void init_vertice_coord(const std::vector<size_t> &order_list);

    size_t compute_index_distance(const SimpleVoxel &voxel_1,const SimpleVoxel &voxel_2);
    bool judge_nearby(const SimpleVoxel &voxel_1,const SimpleVoxel &voxel_2,const SimpleVtk &vtk);
    bool judge_nearby_cell(const SimpleVoxel &voxel_1,const SimpleVoxel &voxel_2,const SimpleVtk &vtk);
    bool judge_same_face(const SimpleFace &face_1,const SimpleFace &face_2);
    bool judge_corresponding_vertex_index(const size_t &voxel_1,const size_t &voxel_2,std::vector<size_t> &index1,std::vector<size_t> &index2);

    void coord_to_index(const cxz::MyVector3 &coord,size_t (&index)[3]);
    cxz::MyVector3 index_to_coord(const size_t (&coord_index)[3],const size_t &point_index);

    size_t orientation_between_voxel(const SimpleVoxel &voxel_1,const SimpleVoxel &voxel_2);
    void insert_bcc_center();
    void init_bcc_segment();
    void init_bcc_tet();
    void init_true_bcc_tet(const SimpleVtk &vtk);
    bool judge_tet_cell_intersect(const size_t &tet_index,const size_t &cell_index,const SimpleVtk &vtk);
};

#endif