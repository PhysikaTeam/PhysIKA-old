#include "global.h"
#include "Image.h"
#include "Pixel.h"
#include "sun.h"
#include "mesh.h"
#include "Cloud.h"

#include "total.h"
#include "main.h"
#include <Eigen/Sparse>
#include <vector>
#include "MeshDeformation.h"
#include <iostream>
using namespace std;

EXT_INFO ext_info;

void half_mesh(string& upload_filename) {
	Tool tool;
	Image image;
	Pixel pixel;
	Sun sun;
	Cloud cloud;
	Sky sky;
	//--------
	Mesh mesh;
	//--------
	image.Initial();
	pixel.Initial();
	sun.Initial();
	//--------
	mesh.Initial();
	//--------
	sky.Initial();
	cloud.Initial();
	image.ReadImage(upload_filename.c_str(), tool); //"E:/zhangqf/project/Sim_Cloud/natural_img/0-250.png" 28 48 90 233 250 330
	pixel.CreatePixelType(image, tool);
	sun.CreateSunColor(pixel, image);
	pixel.CreatePerfectBoundary(image, mesh);
	float a = 0.3*M_PI;
	cout << a << endl;
	sun.CreateSun(a, a);

	//输入：云的轮廓点
	//其中步骤：1.把凹多边形拆分为多个凸多边形 2.remesh方法，内部生成细分三角网格
	mesh.CreatBaseMesh();
	//半面mesh，有内部点
	//-----------------------------------

	sky.CreateSkyPossion(image, pixel);
	cloud.CreatePropagationPath(image, sun, pixel);
	cloud.PropagationCylinders(image, sky, sun);
	cloud.CreateHeightFieldHalf(image, pixel, sky);
	cloud.RemoveOneLoopPixelBoudary(image, pixel, cloud.heightField);
	cloud.CreateCloudMesh(image, pixel, sky, mesh, cloud.heightField);

	//------------------
	cloud.ExportCloudMesh(mesh, "./half_cloud.off");

	ext_info.sun_color = vector<float>(3);
	ext_info.sun_color[0] = sun.sun_color.R;
	ext_info.sun_color[1] = sun.sun_color.B;
	ext_info.sun_color[2] = sun.sun_color.G;
	ext_info.img_WH = vector<int>(2);
	ext_info.img_WH[0] = image.GetImg_width();
	ext_info.img_WH[1] = image.GetImg_height();

	cout << "finished" << endl;
}

void mesh_deform(string& sim_filename) {
	MeshDeformation* meshDeformation = new MeshDeformation();

	string basemesh("./half_cloud.off");
	meshDeformation->CreateMesh(basemesh);

	cout << "Create Mesh DONE!" << endl;

	cout << "Start Create Optimized Mesh----------" << endl;
	meshDeformation->CreateOptimizedMesh(1);
	cout << "Finish Create Optimized Mesh----------" << endl;
	cout << "Start Create Entire HeightField And Mesh----------" << endl;

	meshDeformation->CreateEntireHeightFieldAndMesh();
	cout << "Finish Create Entire HeightField And Mesh----------" << endl;

	string outmesh(sim_filename); // "../output/new-0001.obj"
	meshDeformation->OutputDeformedMesh(outmesh);

	ext_info.num_vertices = meshDeformation->deformedMesh.n_vertices();
	ext_info.num_faces = meshDeformation->deformedMesh.n_faces();
}

//void main() {
//	half_mesh();
//	mesh_deform();
//
//}
vector<float> get_sun_color() { return ext_info.sun_color; }
vector<int> get_img_WH() { return ext_info.img_WH; }
int get_num_vertices() { return ext_info.num_vertices; }
int get_num_faces() { return ext_info.num_faces; }


void sim_cloud(string upload_filename, string sim_filename) {
	half_mesh(upload_filename);
	mesh_deform(sim_filename);
}