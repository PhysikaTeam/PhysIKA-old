#include <iostream>
#include <array>
#include <fstream>
#include "FEMGeometryTetManiCheckInterface.h"
#include "FEMGeometryTetCutCellGenMesh.h"
#include "FEMGeometryTetCutCellGenVert.h"
#include "FEMGeometryTetVtk.h"
#include "FEMGeometryTetVoxel.h"
#include "FEMGeometryTetSeparateVtk.h"
#include "FEMGeometrySurf2Tet.h"

using namespace std;
using namespace Eigen;

int surf2tet(const string surf_file, const string tet_file, const size_t num_span)
{
    //if (argc != 3 && argc != 4)
    //{
    //	printf("[  \033[1;31musage\033[0m  ] exe input_obj output_tet tet_ratio=9\nnote: tet_ratio, with default value 9, is roughly equal to the ratio of model bounding box to the tet bouding box\n");
    //	return -2;
    //}
    array<bool, 3> model_stat = manifold_checking(surf_file.c_str());
    if (model_stat[0] + model_stat[1] + model_stat[2] != 3)
    {
        printf("[  \033[1;31merror\033[0m  ] the input model is not watertight manifold. only watertight manifold model is supported\n");
        return -1;
    }

    Mesh mesh(surf_file.c_str());
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

    // voxel_mesh.write_hex_to_vtk_file(argv[2]);
    // voxel_mesh.write_all_bcc_tet_to_vtk_file(argv[2]);
    voxel_mesh.write_bcc_tet_to_vtk_file(tet_file.c_str());

    return 0;
}
