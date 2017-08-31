



#include<iostream>
#include<string>
#include<vector>
#include "Physika_Geometry\Volumetric_Meshes\volumetric_mesh.h"
#include "Physika_Geometry\Volumetric_Meshes\tet_mesh.h"
#include "Physika_Render\Volumetric_Mesh_Render\volumetric_mesh_render.h"
#include "Physika_Render\Surface_Mesh_Render\surface_mesh_render.h"
#include "Physika_IO\Volumetric_Mesh_IO\volumetric_mesh_io.h"
#include "Physika_Render\Color\color.h"
#include "Physika_Core\Vectors\vector.h"
#include "Physika_GUI\Glut_Window\glut_window.h"
#include "Physika_Geometry\Boundary_Meshes\surface_mesh.h"
#include "Physika_IO\Surface_Mesh_IO\surface_mesh_io.h"
using std::string;
using Physika::VolumetricMesh;
using Physika::TetMesh;
using Physika::VolumetricMeshRender;
using Physika::SurfaceMeshRender;
using Physika::VolumetricMeshIO;
using std::vector;
using Physika::Color;
using Physika::Vector;
using Physika::GlutWindow;
using Physika::SurfaceMesh;
using Physika::SurfaceMeshIO;
using std::cout;
using std::endl;

int main(){
    string fine_mesh_file0 = "tetfine.smesh";
    VolumetricMesh<double, 3> *fine_mesh0;
    fine_mesh0 = VolumetricMeshIO<double, 3>::load(fine_mesh_file0);
    cout<<fine_mesh0->isBoundaryVertex(1)<<endl;
    std::vector<unsigned int> face;
    face.push_back(0);face.push_back(1);face.push_back(2);
    cout<<fine_mesh0->isBoundaryFace(face)<<endl;
    return 0;
}
