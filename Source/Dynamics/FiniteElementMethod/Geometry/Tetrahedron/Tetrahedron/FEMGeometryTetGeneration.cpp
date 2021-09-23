#include <iostream>
#include <array>
#include <fstream>
#include "FEMGeometryTetManiCheckInterface.h"
#include "FEMGeometryTetCutCellGenMesh.h"
#include "FEMGeometryTetCutCellGenVert.h"
#include "FEMGeometryTetVtk.h"
#include "FEMGeometryTetVoxel.h"
#include "FEMGeometryTetSeparateVtk.h"
#include "FEMGeometryTetGeneration.h"

using namespace std;
using namespace Eigen;

int tet_generation(const char* in_obj, const char* out_tet, int cut_num)
{
    array<bool, 3> model_stat = manifold_checking(in_obj);
    if (model_stat[0] + model_stat[1] + model_stat[2] != 3)
    {
        printf("[  \033[1;31merror\033[0m  ] the input model is not watertight manifold. only watertight manifold model is supported\n");
        return -1;
    }

    size_t num_span = cut_num;
    Mesh   mesh(in_obj);
    mesh.set_cut_num(num_span);
    mesh.set_cut_line();
    Vert::init_vert(&mesh);
    Ring::init_ring(&mesh);
    Edge::init_edge(&mesh);
    Triangle::init_triangle(&mesh);

    mesh.cut_mesh();

    string str_cut_cell = "demo.cut_cell";
    mesh.write_cell_to_file(str_cut_cell.c_str());

    //Separate cells that are not connected in the same Voxel
    SimpleVtk      vtk1, vtk2;
    vector<size_t> part;
    vtk1.read_file_vtk42(str_cut_cell.c_str());
    separate_vtk(vtk1, vtk2);

    //build mesh
    VoxelMesh voxel_mesh(vtk2);
    voxel_mesh.init_voxel_mesh(vtk2, part);
    voxel_mesh.bcc_generate_tet(vtk2);

    // voxel_mesh.write_hex_to_vtk_file(out_tet);
    // voxel_mesh.write_all_bcc_tet_to_vtk_file(out_tet);
    voxel_mesh.write_bcc_tet_to_vtk_file(out_tet);

    return 0;
}
