#pragma once
#include "total.h"
#include "MeshDeformation.h"
#include "perlin.h"
#include "Vector.h"
#include <set>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/PolyMesh_ArrayKernelT.hh>
#include <Eigen/Sparse>
#include <Eigen/Core>
#include <Eigen/OrderingMethods>
#include <Eigen/IterativeLinearSolvers>
#include <vector>
#include <omp.h>
using namespace std;

#define M_PI  3.14159265358979323846

MeshDeformation::MeshDeformation(void)
{
	basePlane = 0.0;
	distance2Boundary = NULL;
	noiseList = NULL;
	vertexTypeList = NULL;

	frontHF = NULL;
	behindHF = NULL;
	pixelTypeList = NULL;

	weight_dobashi = 0.65;

	//test
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=1.5;
	//diff_z_scale=1.0;
	//N_cons=1000;

	////test0
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=1.5;
	//diff_z_scale=1.0;
	//N_cons=2000;



	////test3
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=1.0;
	//diff_z_scale=1.0;
	//N_cons=4000;

	////5
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=2;
	//diff_z_scale=1.0;
	//N_cons=200;

	//11
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=2;
	//diff_z_scale=0.8;
	//N_cons=3000;

	////13
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=1.5;
	//diff_z_scale=0.95;
	//N_cons=3000;

	////14
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=1.5;
	//diff_z_scale=1.0;
	//N_cons=2000;

	////64
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=2.0;
	//diff_z_scale=1.0;
	//N_cons=3500;
	////65
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=2.0;
	//diff_z_scale=1.0;
	//N_cons=5000;

	////61
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=2.0;
	//diff_z_scale=1.0;
	//N_cons=3000;

	////e1
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=2.0;
	//diff_z_scale=1.0;
	//N_cons=4000;

	////69
	//boundary_weight=5.0;
	//interior_weight=0.05;
	//dis_scale=1.5;
	//diff_z_scale=1.0;
	//N_cons=4000;

	//1x
	boundary_weight = 5.0;
	interior_weight = 0.05;
	dis_scale = 1.5;
	diff_z_scale = 1.0;
	N_cons = 800; // 3000
}

MeshDeformation::~MeshDeformation(void)
{
	if (distance2Boundary != NULL)
		delete[] distance2Boundary;
	if (noiseList != NULL)
		delete[] noiseList;

	if (vertexTypeList != NULL)
		delete[] vertexTypeList;
	if (frontHF != NULL)
		delete[] frontHF;
	if (behindHF != NULL)
		delete[] behindHF;

	if (pixelTypeList != NULL)
		delete[] pixelTypeList;
}

int MeshDeformation::CreateMesh(string offfile)
{

	if (!OpenMesh::IO::read_mesh(mesh, offfile))
	{
		std::cerr << "Cannot read mesh from .off! " << std::endl;
		return 0;
	}
	NormalizeMesh();
	// write mesh to output.obj
	try
	{
		if (!OpenMesh::IO::write_mesh(mesh, offfile))
		{
			std::cerr << "Cannot write mesh to  mesh.off! " << std::endl;
			return 0;
		}
	}
	catch (std::exception& x)
	{
		std::cerr << x.what() << std::endl;
		return 0;
	}
	OpenMesh::IO::Options opt;
	// If the file did not provide vertex normals, then calculate them
	if (!opt.check(OpenMesh::IO::Options::VertexNormal))
	{
		// we need face normals to update the vertex normals
		mesh.request_face_normals();

		// let the mesh update the normals
		mesh.update_normals();
		//// dispose the face normals, as we don't need them anymore
		//mesh.release_face_normals();
	}

	//����������߽�ڵ㵽boundaryVerList
	CreateBoundaryList();
	CreateDistance2D();

	constraintMesh = openMesh(mesh);
	openMesh::VertexIter v_it, v_end(constraintMesh.vertices_end());
	for (v_it = constraintMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		constraintMeshVerHandleList.push_back(*v_it);
	}

	deformedMesh = openMesh(mesh);

	cout << "read off------------done!" << endl;

	return 1;
}

bool MeshDeformation::OutputDeformedMesh(string meshfile)
{
	// write mesh to output.obj
	try
	{
		OpenMesh::IO::Options opt = OpenMesh::IO::Options::VertexNormal;
		// we need face normals to update the vertex normals
		deformedMesh.request_vertex_normals();
		if (!deformedMesh.has_vertex_normals())
		{
			std::cerr << "ERROR:Standard vertex property 'Normals' not available!\n";
			return 1;
		}
		// we need face normals to update thevertex normals
		deformedMesh.request_face_normals();
		// let the mesh update the normals
		deformedMesh.update_normals();
		// dispose the face normals, as we don'tneed them anymore
		deformedMesh.release_face_normals();

		ofstream outfile(meshfile);
		if (!OpenMesh::IO::write_mesh(deformedMesh, outfile, ".obj", opt))
		{
			std::cerr << "Cannot write mesh to deformed mesh.off! " << std::endl;
			return 1;
		}
		outfile.close();
	}
	catch (std::exception& x)
	{
		std::cerr << x.what() << std::endl;
		return 1;
	}
}

//void MeshDeformation::DrawMesh()
//{
//	float mesh_z_min = MAXVAL;
//
//	openMesh::VertexIter          v_it, v_end(mesh.vertices_end());
//	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
//	{
//		openMesh::Point  pt = mesh.point(v_it);
//		mesh_z_min = min(pt[2], mesh_z_min);
//
//	}
//
//
//
//
//	float z_max = -MAXVAL;
//	float z_min = MAXVAL;
//
//	float sys_plane = mesh_z_min;
//
//	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
//	{
//		openMesh::Point  pt = mesh.point(v_it);
//		z_max = max(pt[2], z_max);
//		z_max = max(2 * sys_plane - pt[2], z_max);
//		z_min = min(pt[2], z_min);
//		z_min = min(2 * sys_plane - pt[2], z_min);
//	}
//
//	float z_center = (z_max + z_min) / 2.0;
//
//
//
//
//
//	mesh.update_normals();
//
//	openMesh::FaceIter          f_it, f_end(mesh.faces_end());
//	for (f_it = mesh.faces_begin(); f_it != f_end; ++f_it)
//	{
//		openMesh::FaceVertexIter  fv_it, fv_end(mesh.fv_end(f_it));
//
//		openMesh::Normal   normal = mesh.normal(f_it);
//		Vector3  normalVec(normal[0], normal[1], normal[2]);
//		glNormal3fv(!normalVec);
//		glBegin(GL_POLYGON);
//		for (fv_it = mesh.fv_begin(f_it); fv_it != fv_end; ++fv_it)
//		{
//			openMesh::Point  pt = mesh.point(fv_it);
//			Vector3 ver(pt[0], pt[1], pt[2] - z_center);
//			glVertex3fv(!ver);
//
//		}
//		glEnd();
//
//
//	}
//
//
//	//back surface
//	for (f_it = mesh.faces_begin(); f_it != f_end; ++f_it)
//	{
//		openMesh::FaceVertexIter  fv_it, fv_end(mesh.fv_end(f_it));
//		Vector3 ver[3];
//		int i = 0;
//		for (fv_it = mesh.fv_begin(f_it); i < 3, fv_it != fv_end; ++fv_it, i++)
//		{
//			openMesh::Point  pt = mesh.point(fv_it);
//			pt[2] = 2 * sys_plane - pt[2] - z_center;
//			ver[i] = Vector3(pt[0], pt[1], pt[2]);
//
//		}
//
//		Vector3 normalVec;
//		ComputeTriangleNormal(!normalVec, !(ver[0]), !(ver[2]), !(ver[1]));
//		glNormal3fv(!normalVec);
//		glBegin(GL_POLYGON);
//		glVertex3fv(!ver[0]);
//		glVertex3fv(!ver[2]);
//		glVertex3fv(!ver[1]);
//		glEnd();
//
//
//	}
//
//
//	//side surface
//	vector<Vector3>  frontBoudaryList;
//	vector<Vector3>  backBoudaryList;
//	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
//	{
//		if (mesh.is_boundary(v_it))
//		{
//			openMesh::Point  pt = mesh.point(v_it);
//			float z = pt[2] - z_center;
//			Vector3 frontBoudaryVer = Vector3(pt[0], pt[1], z);
//			z = 2 * sys_plane - pt[2] - z_center;
//			Vector3 backBoudaryVer = Vector3(pt[0], pt[1], z);
//			frontBoudaryList.push_back(frontBoudaryVer);
//			backBoudaryList.push_back(backBoudaryVer);
//
//		}
//
//	}
//
//	for (int i = 0; i < frontBoudaryList.size() - 1; i++)
//	{
//		Vector3 normalVec, ver[3];
//		ver[0] = backBoudaryList[i];
//		ver[1] = frontBoudaryList[i + 1];
//		ver[2] = frontBoudaryList[i];
//		ComputeTriangleNormal(!normalVec, !(ver[0]), !(ver[1]), !(ver[2]));
//		glNormal3fv(!normalVec);
//		glBegin(GL_POLYGON);
//		glVertex3fv(!ver[0]);
//		glVertex3fv(!ver[1]);
//		glVertex3fv(!ver[2]);
//		glEnd();
//
//		ver[0] = backBoudaryList[i];
//		ver[1] = backBoudaryList[i + 1];
//		ver[2] = frontBoudaryList[i + 1];
//		ComputeTriangleNormal(!normalVec, !(ver[0]), !(ver[1]), !(ver[2]));
//		glNormal3fv(!normalVec);
//		glBegin(GL_POLYGON);
//		glVertex3fv(!ver[0]);
//		glVertex3fv(!ver[1]);
//		glVertex3fv(!ver[2]);
//		glEnd();
//
//	}
//
//
//
//	glDisable(GL_LIGHTING);
//	glColor3f(1, 0, 0);
//	glPointSize(3.0);
//	glBegin(GL_POINTS);
//	for (int i = 0; i < boundaryVerList.size(); i++)
//	{
//		openMesh::Point  pt = mesh.point(boundaryVerList[i]);
//		Vector3 ver(pt[0], pt[1], pt[2] - z_center);
//		/*		glColor3f(0.1,float(i)/boundaryVerList.size(),0);*/
//		glColor3f(1, 0, 0);
//		glVertex3fv(!ver);
//
//	}
//	glEnd();
//
//
//	glDisable(GL_LIGHTING);
//	glColor3f(1, 0, 0);
//	glBegin(GL_POINTS);
//	for (int i = 0; i < constraintVerList.size(); i++)
//	{
//		openMesh::Point  pt = mesh.point(constraintVerList[i]);
//		Vector3 ver(pt[0], pt[1], pt[2] - z_center);
//		glColor3f(0, 1, 0);
//		glVertex3fv(!ver);
//
//	}
//	glEnd();
//
//
//	DrawSelectedVertices();
//
//	glPointSize(1.0);
//}

void MeshDeformation::NormalizeMesh()
{
	openMesh::Point  center;
	float  x_min = MAXVAL;
	float  x_max = -MAXVAL;
	float  y_min = MAXVAL;
	float  y_max = -MAXVAL;
	float  z_min = MAXVAL;
	float  z_max = -MAXVAL;

	openMesh::VertexIter v_it, v_end(mesh.vertices_end());
	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		//openMesh::Point pt = *v_it;
		openMesh::Point pt = mesh.point(*v_it);
		x_min = min(x_min, pt[0]);
		x_max = max(x_max, pt[0]);
		y_min = min(y_min, pt[1]);
		y_max = max(y_max, pt[1]);
		z_min = min(z_min, pt[2]);
		z_max = max(z_max, pt[2]);

	}

	center = openMesh::Point((x_max + x_min) / 2.0, (y_max + y_min) / 2.0, 0);

	float scale = max(x_max - x_min, max(y_max - y_min, z_max - z_min));

	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  pt = (mesh.point(*v_it) - center) / scale*1.8;
		mesh.set_point(*v_it, pt);
	}


	///***************************************************************
	dis_scale *= (z_max - z_min) / scale;
}

void MeshDeformation::NormalizeMesh(openMesh& mesh)
{
	openMesh::Point  center;
	float  x_min = MAXVAL;
	float  x_max = -MAXVAL;
	float  y_min = MAXVAL;
	float  y_max = -MAXVAL;
	float  z_min = MAXVAL;
	float  z_max = -MAXVAL;


	openMesh::VertexIter v_it, v_end(mesh.vertices_end());
	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  pt = mesh.point(*v_it);
		x_min = min(x_min, pt[0]);
		x_max = max(x_max, pt[0]);
		y_min = min(y_min, pt[1]);
		y_max = max(y_max, pt[1]);
		z_min = min(z_min, pt[2]);
		z_max = max(z_max, pt[2]);

	}

	center = openMesh::Point((x_max + x_min) / 2.0, (y_max + y_min) / 2.0, (z_max + z_min) / 2.0);

	float scale = max(x_max - x_min, max(y_max - y_min, z_max - z_min));

	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  pt = (mesh.point(*v_it) - center) / scale*1.8;
		mesh.set_point(*v_it, pt);
	}
}

//Vector3 MeshDeformation::GetPixelLoc(float x, float y, float z)
//{
//	GLdouble modelMatrix[16];
//	GLdouble projMatrix[16];
//	GLint viewport[4];
//	//
//	GLdouble objx = x;
//	GLdouble objy = y;
//	GLdouble objz = z;
//	//
//	GLdouble pix_x;
//	GLdouble pix_y;
//	GLdouble pix_z;
//	//
//	glGetDoublev(GL_MODELVIEW_MATRIX, modelMatrix);
//	glGetDoublev(GL_PROJECTION_MATRIX, projMatrix);
//	glGetIntegerv(GL_VIEWPORT, viewport);
//	//
//	int result = gluProject(objx, objy, objz, modelMatrix, projMatrix, viewport,
//		&pix_x, &pix_y, &pix_z);
//	//
//	Vector3 pixel;
//	//
//	// need to reverse y since OpenGL has y=0 at bottom, but windows has y=0 at top
//	//
//	pixel.x = pix_x;
//	pixel.y = viewport[1] + viewport[3] - pix_y;
//	pixel.z = pix_z;
//	//
//	return  pixel;
//}

//void  MeshDeformation::SelectVertex(int x, int y)
//{
//	vector<pixelStuct>  pixelList;
//	openMesh::VertexIter          v_it, v_end(mesh.vertices_end());
//	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
//	{
//		openMesh::Point  pt = mesh.point(v_it);
//
//		Vector3  curPixel = GetPixelLoc(pt[0], pt[1], pt[2]);
//
//		if (pow(curPixel.x - x, 2) + pow(curPixel.y - y, 2) < 15)
//		{
//			pixelList.push_back(pixelStuct(curPixel, v_it));
//		}
//
//
//	}
//	float  min_depth = MAXVAL;
//	float  min_index = -1;
//	for (int i = 0; i < pixelList.size(); i++)
//	{
//		if (min_depth > pixelList[i].pixel.z)
//		{
//			min_depth = pixelList[i].pixel.z;
//			min_index = i;
//		}
//
//	}
//	if (min_index >= 0)
//		selectedVertices.push_back(pixelList[min_index].v_it);
//}

//void MeshDeformation::DrawSelectedVertices()
//{
//	glDisable(GL_LIGHTING);
//	glColor3f(1, 1, 1);
//	glPointSize(3.0);
//	glBegin(GL_POINTS);
//	for (int i = 0; i < selectedVertices.size(); i++)
//	{
//		openMesh::VertexIter  v_it = selectedVertices[i];
//		openMesh::Point  pt = mesh.point(v_it);
//		glVertex3f(pt[0], pt[1], pt[2]);
//
//	}
//
//	glEnd();
//}

void MeshDeformation::MoveCurSelectedVertex(Vector3 direction,  float dis )
{
	if(selectedVertices.size()==0)
		return;
	openMesh::VertexIter  v_it=selectedVertices[selectedVertices.size()-1];

    Vector3  displace=direction*dis;
	openMesh::Point  pt=mesh.point(*v_it)+openMesh::Point(displace.x,displace.y,displace.z);
	mesh.set_point(*v_it,pt);

}

void MeshDeformation::CreateLaplaceMatrix()
{
	laplaceMat.clear();

	int row;
	int  nSingular = 0;
	openMesh::VertexIter v_it, v_end(mesh.vertices_end());
	for (row = 0, v_it = mesh.vertices_begin(); v_it != v_end; ++v_it, row++)
	{
		openMesh::VertexVertexIter vv_it;

		int  valence = 0;
		for (vv_it = mesh.vv_iter(*v_it); vv_it.is_valid()==true; ++vv_it)
		{
			valence++;
		}
		if (valence == 0)
		{
			/*		cout<<"Singular vertex"<<endl;*/
			MatElement matEle(row, row, 1.0);
			laplaceMat.push_back(matEle);
			nSingular++;
		}
		else
		{
			MatElement matEle(row, row, 1.0);
			laplaceMat.push_back(matEle);

			for (vv_it = mesh.vv_iter(*v_it); vv_it.is_valid()==true; ++vv_it)
			{
				int  col = vv_it->idx();
				MatElement matEle(row, col, -1.0 / valence);
				laplaceMat.push_back(matEle);


			}
		}


	}
	cout << "Laplace matrix------------done!" << "   Vertice with zero valence:   " << nSingular << endl;
}

void MeshDeformation::CreateDifCoorList()
{
	diferentialCoordinateList.clear();


	openMesh::VertexIter v_it, v_end(mesh.vertices_end());
	int row;
	for (row = 0, v_it = mesh.vertices_begin(); v_it != v_end; ++v_it, row++)
	{
		openMesh::VertexVertexIter    vv_it;

		int  valence = 0;
		for (vv_it = mesh.vv_iter(*v_it); vv_it.is_valid() == true; ++vv_it)
		{
			valence++;
		}

		if (valence == 0)
		{
			openMesh::Point pt = mesh.point(*v_it);
			Vector3 difCoor = Vector3(pt[0], pt[1], pt[2]);
			diferentialCoordinateList.push_back(difCoor);
		}

		else
		{
			openMesh::Point  pt = mesh.point(*v_it);
			Vector3 curVer = Vector3(pt[0], pt[1], pt[2]);
			Vector3  sumNeigbor = Vector3(0, 0, 0);
			for (vv_it = mesh.vv_iter(*v_it); vv_it.is_valid() == true; ++vv_it)
			{
				openMesh::Point neigbor_pt = mesh.point(*vv_it);
				Vector3 neiVer = Vector3(neigbor_pt[0], neigbor_pt[1], neigbor_pt[2]);
				sumNeigbor = sumNeigbor + neiVer;
			}

			Vector3  difCoor = curVer - sumNeigbor*(1.0 / valence);

			diferentialCoordinateList.push_back(difCoor);
		}
	}
}

void MeshDeformation::AddConstrains()
{
	openMesh::Point  pt;
	float weight = 1;
	for (int i = 0; i < selectedVertices.size(); i++)
	{
		int row = i + mesh.n_vertices();
		int col = selectedVertices[i]->idx();
		MatElement matEle(row, col, weight);
		laplaceMat.push_back(matEle);

		openMesh::Point  pt = mesh.point(*(selectedVertices[i]));
		Vector3 constrains_coor = Vector3(pt[0], pt[1], pt[2])*weight;
		diferentialCoordinateList.push_back(constrains_coor);

	}
}

bool MeshDeformation::UpdateMesh()
{
	//if (!libcomputenewmeshverInitialize())
	//{
	//	std::cout << "Could not initialize libcomputenewmeshver!" << std::endl;
	//	std::cout << "Really CANNOT initialize" << std::endl;
	//	return false;
	//}
	//try
	//{
	typedef Eigen::SparseMatrix<float> SpMat;

	int* rowList = new int[laplaceMat.size()];
	int* colList = new int[laplaceMat.size()];
	float* weightList = new float[laplaceMat.size()];
	float*  differentialCoorList_X = new float[diferentialCoordinateList.size()];
	float*  differentialCoorList_Y = new float[diferentialCoordinateList.size()];
	float*  differentialCoorList_Z = new float[diferentialCoordinateList.size()];

	int matrix_rows = -1;
	int matrix_cols = -1;
	for (int i = 0; i < laplaceMat.size(); i++)
	{
		rowList[i] = laplaceMat[i].row;
		colList[i] = laplaceMat[i].col;
		//rowList[i] = laplaceMat[i].row + 1;
		//colList[i] = laplaceMat[i].col + 1;
		weightList[i] = laplaceMat[i].element_value;
		if (rowList[i] > matrix_rows)
			matrix_rows = rowList[i];
		if (colList[i] > matrix_cols)
			matrix_cols = colList[i];
	}
	for (int i = 0; i < diferentialCoordinateList.size(); i++)
	{
		differentialCoorList_X[i] = diferentialCoordinateList[i].x;
		differentialCoorList_Y[i] = diferentialCoordinateList[i].y;
		differentialCoorList_Z[i] = diff_z_scale*diferentialCoordinateList[i].z;
	}
	
	SpMat matrix(matrix_rows + 1, matrix_cols + 1);
	vector<Eigen::Triplet<float> > tmp;
 
	for (int i = 0; i < laplaceMat.size(); i++)
	{
		//cout << i << "  "<<rowList[i]<<"  "<<colList[i]<<endl;
		tmp.push_back(Eigen::Triplet<float>(rowList[i], colList[i], weightList[i]));
	}
	matrix.setFromTriplets(tmp.begin(), tmp.end());

	Eigen::VectorXf eDifferentialCoorList_X(diferentialCoordinateList.size());
	Eigen::VectorXf eDifferentialCoorList_Y(diferentialCoordinateList.size());
	Eigen::VectorXf eDifferentialCoorList_Z(diferentialCoordinateList.size());

#pragma omp parallel for 
	for (int i = 0; i < diferentialCoordinateList.size(); i++)
	{
		eDifferentialCoorList_X[i] = differentialCoorList_X[i];
		eDifferentialCoorList_Y[i] = differentialCoorList_Y[i];
		eDifferentialCoorList_Z[i] = differentialCoorList_Z[i];
	}

	Eigen::LeastSquaresConjugateGradient<SpMat> lscg;
	lscg.compute(matrix);

	Eigen::VectorXf** eDifferentialCoorList_Total = new Eigen::VectorXf*[3];
	eDifferentialCoorList_Total[0] = &eDifferentialCoorList_X;
	eDifferentialCoorList_Total[1] = &eDifferentialCoorList_Y;	
	eDifferentialCoorList_Total[2] = &eDifferentialCoorList_Z;

	Eigen::VectorXf eTotal[3];

#pragma omp parallel num_threads(3)
	{
        #pragma omp for
		for (int i = 0; i < 3; i++)
		{
			eTotal[i] = lscg.solve(*(eDifferentialCoorList_Total[i]));
		}
	}

	float*  X = new float[mesh.n_vertices()];
	float*  Y = new float[mesh.n_vertices()];
	float*  Z = new float[mesh.n_vertices()];
#pragma omp parallel for
	for (int i = 0; i < mesh.n_vertices(); i++)
	{
		X[i] = eTotal[0][i];
		Y[i] = eTotal[1][i];
		Z[i] = eTotal[2][i];
	}

	//mX.GetData(X, mesh.n_vertices());
	//mY.GetData(Y, mesh.n_vertices());
	//mZ.GetData(Z, mesh.n_vertices());

	openMesh::VertexIter  v_it, v_end(mesh.vertices_end());
	openMesh::VertexIter  dv_it, dv_end(deformedMesh.vertices_end());
	int i = 0;
	for (i = 0, v_it = mesh.vertices_begin(), dv_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it, ++dv_it, i++)
	{
		openMesh::Point  meshPt = mesh.point(*v_it);
		openMesh::Point  pt;
		pt[0] = X[i];
		pt[1] = Y[i];
		pt[2] = Z[i];
		if (vertexTypeList != NULL)
		{
			if (vertexTypeList[v_it->idx()] > 0)
				deformedMesh.set_point(*dv_it, pt);
			else
				deformedMesh.set_point(*dv_it, meshPt);
		}
		else
		{
			deformedMesh.set_point(*dv_it, pt);
		}
	}
	delete[] X;
	delete[] Y;
	delete[] Z;
	delete[] rowList;
	delete[] colList;
	delete[] weightList;
	delete[] differentialCoorList_X;
	delete[] differentialCoorList_Y;
	delete[] differentialCoorList_Z;
	return true;
}

void MeshDeformation::AddBoudaryContrainst()
{

	float weight = boundary_weight;
	int cur_col_count = diferentialCoordinateList.size();
	for (int i = 0; i < boundaryVerList.size(); i++)
	{
		openMesh::Point  pt = mesh.point(boundaryVerList[i]);
		int row = i + cur_col_count;
		int col = boundaryVerList[i].idx();
		MatElement matEle(row, col, weight);
		laplaceMat.push_back(matEle);

		Vector3 constrains_coor = Vector3(pt[0], pt[1], basePlane)*weight;
		diferentialCoordinateList.push_back(constrains_coor);

	}
}

//void MeshDeformation::DrawDeformedMesh()
//{
//	glEnable(GL_LIGHTING);
//
//	deformedMesh.update_normals();
//
//	openMesh::FaceIter          f_it, f_end(deformedMesh.faces_end());
//	for (f_it = deformedMesh.faces_begin(); f_it != f_end; ++f_it)
//	{
//		openMesh::FaceVertexIter  fv_it, fv_end(deformedMesh.fv_end(f_it));
//
//		openMesh::Normal   normal = deformedMesh.normal(f_it);
//		Vector3  normalVec(normal[0], normal[1], normal[2]);
//		glNormal3fv(!normalVec);
//		glBegin(GL_POLYGON);
//		for (fv_it = deformedMesh.fv_begin(f_it); fv_it != fv_end; ++fv_it)
//		{
//			openMesh::Point  pt = deformedMesh.point(fv_it);
//			Vector3 ver(pt[0], pt[1], pt[2]);
//			glVertex3fv(!ver);
//
//		}
//		glEnd();
//	}
//}

//void MeshDeformation::Draw(DrawType drawType)
//{
//	switch (drawType)
//	{
//	case  InputMesh:
//		DrawMesh();
//		break;
//	case ConstraintMesh:
//		DrawConstraintMesh();
//		break;
//	case CloudPIxelOnBasePlane:
//		DrawDistance2D();
//		break;
//	case Distance:
//		DrawDistance();
//		break;
//	case DeformedMesh:
//		DrawDeformedMesh();
//		break;
//	case Height:
//		DrawHeightField();
//		break;
//	}
//}

bool MeshDeformation::OutputTempMesh(char* meshfile)
{
	// write mesh to output.obj
	try
	{
		if (!OpenMesh::IO::write_mesh(mesh, meshfile))
		{
			std::cerr << "Cannot write mesh to deformed mesh.off! " << std::endl;
			return 1;
		}
	}
	catch (std::exception& x)
	{
		std::cerr << x.what() << std::endl;
		return 1;
	}
}

void MeshDeformation::CreateEntireHeightFieldAndMesh()
{
	int  nFace = deformedMesh.n_faces();
	int  nVer = deformedMesh.n_vertices();

	std::vector<openMesh::Point>   ptList;

	int* accumulatedBoundaryCountList = new int[nVer];
	int accumulated = 0;
	openMesh::VertexIter          v_it, v_end(deformedMesh.vertices_end());
	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		accumulatedBoundaryCountList[v_it->idx()] = accumulated;
		int idx = v_it->idx();

		if (deformedMesh.is_boundary(*v_it))
		{
			accumulated++;
		}

	}
	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  pt = deformedMesh.point(*v_it);
		ptList.push_back(pt);

	}

	//CreateFrontHeightField
	frontHF = new float[PARTICLE_RES*PARTICLE_RES];

	for (int i = 0; i < PARTICLE_RES; i++)
	{
		for (int j = 0; j < PARTICLE_RES; j++)
		{
			if (df.GetDistance(j, i) > 0)
			{
				frontHF[i*PARTICLE_RES + j] = 0.0;
			}
			else
			{
				//nearest neighbor interpolation
				Vector2  pos = df.GetPos(j, i);
				float dis = MAXVAL;
				float  H = 0.0;
				openMesh::VertexIter   v_it, v_end(deformedMesh.vertices_end());
				for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
				{

					openMesh::Point  pt = deformedMesh.point(*v_it);
					Vector2  cur_pos(pt[0], pt[1]);
					float cur_dis = Dot(cur_pos - pos, cur_pos - pos);
					if (cur_dis < dis)
					{
						dis = cur_dis;
						H = pt[2];
					}
				}
				frontHF[i*PARTICLE_RES + j] = H;
			}//else 
		}//for ij 
	}
	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  pt = deformedMesh.point(*v_it);
		if (!deformedMesh.is_boundary(*v_it))
		{
			float dobashi = basePlane - (noiseList[v_it->idx()])*distance2Boundary[v_it->idx()];
			float our = basePlane - pt[2];

			pt[2] = weight_dobashi*dobashi + our*(1 - weight_dobashi);

			ptList.push_back(pt);
		}

	}
	//Create  behind HeightField
	behindHF = new float[PARTICLE_RES*PARTICLE_RES];

	for (int i = 0; i < PARTICLE_RES; i++)
	{
		for (int j = 0; j < PARTICLE_RES; j++)
		{
			if (df.GetDistance(j, i) > 0)
			{
				behindHF[i*PARTICLE_RES + j] = 0.0;
			}
			else
			{
				//nearest neighbor interpolation
				Vector2  pos = df.GetPos(j, i);
				float dis = MAXVAL;
				float  H = 0.0;
				openMesh::VertexIter   v_it, v_end(deformedMesh.vertices_end());
				for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
				{
					openMesh::Point  pt = deformedMesh.point(*v_it);
					if (!deformedMesh.is_boundary(*v_it))
					{
						float dobashi = basePlane - (noiseList[v_it->idx()])*distance2Boundary[v_it->idx()];
						float our = basePlane - pt[2];
						pt[2] = weight_dobashi*dobashi + our*(1 - weight_dobashi);

					}
					Vector2  cur_pos(pt[0], pt[1]);
					float cur_dis = Dot(cur_pos - pos, cur_pos - pos);
					if (cur_dis < dis)
					{
						dis = cur_dis;
						H = pt[2];
					}
				}
				behindHF[i*PARTICLE_RES + j] = H;
			}//else 
		}//for ij 
	}
	int* faceList = new int[nFace * 3 * 2];
	openMesh::FaceIter f_it, f_end(deformedMesh.faces_end());
	int i;
	for (f_it = deformedMesh.faces_begin(); f_it != f_end; ++f_it)
	{
		openMesh::FaceVertexIter  fv_it, fv_end(deformedMesh.fv_end(*f_it));
		int face_id = f_it->idx();

		for (i = 0, fv_it = deformedMesh.fv_begin(*f_it); fv_it != fv_end; ++fv_it, ++i)
		{
			faceList[face_id * 3 + i] = fv_it->idx();
		}
	}
	for (f_it = deformedMesh.faces_begin(); f_it != f_end; ++f_it)
	{
		openMesh::FaceVertexIter  fv_it, fv_end(deformedMesh.fv_end(*f_it));
		int face_id = f_it->idx();
		int  face[3];
		for (i = 0, fv_it = deformedMesh.fv_begin(*f_it); fv_it != fv_end; ++fv_it, ++i)
		{
			if (deformedMesh.is_boundary(*fv_it))
			{
				face[i] = fv_it->idx();
			}
			else
			{
				face[i] = nVer + fv_it->idx() - accumulatedBoundaryCountList[fv_it->idx()];
			}
		}
		faceList[3 * nFace + face_id * 3 + 0] = face[0];
		faceList[3 * nFace + face_id * 3 + 1] = face[2];
		faceList[3 * nFace + face_id * 3 + 2] = face[1];
	}
	
	ofstream  out("./temp.off");
	out << "OFF" << endl;
	out << 2 * nVer - boundaryVerList.size() << " " << 2 * nFace << " " << 0 << endl;
	for (int i = 0; i < ptList.size(); i++)
	{
		out << ptList[i][0] << " " << ptList[i][1] << " " << ptList[i][2] << endl;
	}
	for (int i = 0; i < 2 * nFace; i++)
	{
		out << 3 << " " << faceList[3 * i + 0] << " " << faceList[3 * i + 1] << " " << faceList[3 * i + 2] << endl;
	}
	
	ClearMesh(deformedMesh);
	OpenMesh::IO::read_mesh(deformedMesh, "./temp.off");
	
	NormalizeMesh(deformedMesh);

	delete[] accumulatedBoundaryCountList;

	cout << "Entire mesh and heightfield------------------------done!" << endl;
}

void MeshDeformation::ClearMesh(openMesh& mesh)
{
	openMesh::FaceIter f_it, f_end(deformedMesh.faces_end());
	for (f_it = deformedMesh.faces_begin(); f_it != f_end; ++f_it)
	{
		mesh.delete_face(*f_it);

	}
	openMesh::EdgeIter e_it, e_end(deformedMesh.edges_end());
	for (e_it = deformedMesh.edges_begin(); e_it != e_end; ++e_it)
	{
		mesh.delete_edge(*e_it);

	}
	openMesh::VertexIter v_it, v_end(deformedMesh.vertices_end());
	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		mesh.delete_vertex(*v_it);

	}

	mesh.garbage_collection();
}

void MeshDeformation::SmoothMesh(openMesh& mesh, int N)
{
	// this vertex property stores the computed centers of gravity
	OpenMesh::VPropHandleT<openMesh::Point> cogs;
	mesh.add_property(cogs);
	openMesh::VertexVertexIter    vv_it;
	openMesh::Point               cog;
	openMesh::VertexIter          v_it, v_end(mesh.vertices_end());
	int valence = 0;
	for (int i = 0; i < N; ++i)
	{
		for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
		{
			mesh.property(cogs, *v_it).vectorize(0.0f);
			valence = 0;

			for (vv_it = mesh.vv_iter(*v_it); vv_it.is_valid()==true; ++vv_it)
			{
				mesh.property(cogs, *v_it) += mesh.point(*vv_it);
				++valence;
			}
			if (valence > 0)
				mesh.property(cogs, *v_it) /= valence;
			else
				mesh.property(cogs,* v_it) = mesh.point(*v_it);
		}

		for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
			if (!mesh.is_boundary(*v_it))
				mesh.set_point(*v_it, mesh.property(cogs, *v_it));
	}
}

void MeshDeformation::Smooth()
{
	SmoothMesh(deformedMesh, 1);
}

void MeshDeformation::OptimizeMesh()
{

	CreateNoiseList();
	std::cout << "Create NoiseList done!" << std::endl;
	CreateDis2Boundary();
	std::cout << "Createdis2boundary done!" << std::endl;


	openMesh::VertexIter          v_it, v_end(deformedMesh.vertices_end());
	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  cur_pt = deformedMesh.point(*v_it);

		if (!deformedMesh.is_boundary(*v_it) && cur_pt[2] < basePlane)
		{
			cur_pt[2] = distance2Boundary[v_it->idx()] * (noiseList[v_it->idx()] + 1) / 2;
			deformedMesh.set_point(*v_it, cur_pt);

		}

	}

	//�����deformedmesh���һ�¿�����ʲô����
	//OpenMesh::IO::write_mesh(deformedMesh,"D:\\yuancodes\\MeshOptimization_2014_2_20\\MeshDeformation\\data\\deformedmesh.off");
	//getchar();
}

void MeshDeformation::CreateDis2Boundary()
{
	if (distance2Boundary == NULL)
		distance2Boundary = new float[deformedMesh.n_vertices()];
	openMesh::VertexIter          v_it, v_end(deformedMesh.vertices_end());
	openMesh::VertexIter          v_it2, v_end2(deformedMesh.vertices_end());

	//�Ȱ�boundaryvertex��������
	std::vector<openMesh::VertexHandle> boundaryVerList;
	openMesh::VertexHandle v0;
	openMesh::HalfedgeHandle heh, heh_init, heh_pre;

	//��boundaryvertex������
	for (v_it = constraintMesh.vertices_begin(); v_it != v_end; v_it++)
	{
		if (constraintMesh.is_boundary(*v_it))
		{
			v0 = *v_it;
			break;
		}
	}

	heh = heh_init = constraintMesh.halfedge_handle(v0);
	v0 = constraintMesh.to_vertex_handle(heh);
	boundaryVerList.push_back(v0);

	heh = constraintMesh.next_halfedge_handle(heh);
	heh_pre = mesh.prev_halfedge_handle(heh);

	while (heh != heh_init)
	{
		v0 = mesh.to_vertex_handle(heh);
		boundaryVerList.push_back(v0);
		heh = constraintMesh.next_halfedge_handle(heh);
	}
	//�������еı߽���Ѿ�����boundaryverlist����


	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		distance2Boundary[v_it->idx()] = 99999;
		openMesh::Point  cur_pt = deformedMesh.point(*v_it);
		if (deformedMesh.is_boundary(*v_it))
		{
			distance2Boundary[v_it->idx()] = 0.0;
		}
		else
		{
			//   for (v_it2=deformedMesh.vertices_begin(); v_it2!=v_end2; ++v_it2)
			   //{
			   //	if(deformedMesh.is_boundary(v_it2))
			   //	{
			   //		openMesh::Point  pt=deformedMesh.point( v_it2 );
			   //		float tempDis=sqrt(powf(cur_pt[0]-pt[0],2)+powf(cur_pt[1]-pt[1],2));
			   //		distance2Boundary[v_it.handle().idx()]=min(distance2Boundary[v_it.handle().idx()],tempDis);
	  //             }
			   //}
		   //ֱ����boundaryVerList�и��£�����һ��һ����
			for (int i = 0; i < boundaryVerList.size(); i++)
			{
				openMesh::Point pt = constraintMesh.point(boundaryVerList[i]);
				float tempDis = sqrt(powf(cur_pt[0] - pt[0], 2) + powf(cur_pt[1] - pt[1], 2));
				distance2Boundary[v_it->idx()] = min(distance2Boundary[v_it->idx()], tempDis);
			}
		}
	}

	float MaxDis = -99999;
	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		MaxDis = max(MaxDis, distance2Boundary[v_it->idx()]);
	}

	for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		distance2Boundary[v_it->idx()] /= MaxDis;
		//cout<<	distance2Boundary[v_it.handle().idx()]<<endl;
	}
}

//void MeshDeformation::DrawDistance()
//{
//	if (distance2Boundary == NULL)
//		return;
//	openMesh::VertexIter          v_it, v_end(mesh.vertices_end());
//	glDisable(GL_LIGHTING);
//	glBegin(GL_POINTS);
//	int i = 0;
//	for (v_it = mesh.vertices_begin(); i < mesh.n_vertices(), v_it != v_end; i++, ++v_it)
//	{
//		float cur_dis = distance2Boundary[v_it.handle().idx()];
//		openMesh::Point  cur_pt = mesh.point(v_it);
//
//		glColor3f(cur_dis, 1, 0);
//		glVertex2d(cur_pt[0], cur_pt[1]);
//	}
//	glEnd();
//}

void MeshDeformation::CreateNoiseList()
{
	int nVer = deformedMesh.n_vertices();
	Perlin   perlin = Perlin(4, 4, 1, 94);
	if (noiseList == NULL)
		noiseList = new float[nVer];

	float minNoise = 999999;
	float maxNoise = -999999;
	openMesh::VertexIter          v_it, v_end(deformedMesh.vertices_end());
	int i;
	for (i = 0, v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it, ++i)
	{
		openMesh::Point  pt = deformedMesh.point(*v_it);
		noiseList[i] = perlin.Get((pt[0] + 1) / 2, (pt[1] + 1) / 2);
		minNoise = min(minNoise, noiseList[i]);
		maxNoise = max(maxNoise, noiseList[i]);
	}

	for (int i = 0; i < nVer; i++)
	{
		noiseList[i] = (noiseList[i] - minNoise) / (maxNoise - minNoise);
	}
}

void MeshDeformation::CreateVetexType()
{
	if (vertexTypeList == NULL)
		vertexTypeList = new int[mesh.n_vertices()];
	openMesh::VertexIter          v_it, v_end(mesh.vertices_end());
	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  meshPt = mesh.point(*v_it);
		float minDis = 999999;
		for (int i = 0; i < selectedVertices.size(); i++)
		{
			openMesh::Point  selectedPt = mesh.point(*(selectedVertices[i]));
			Vector3  seleVec(selectedPt[0], selectedPt[1], selectedPt[2]);
			Vector3  meshPtVec(meshPt[0], meshPt[1], meshPt[2]);
			float  dis = Magnitude(seleVec - meshPtVec);
			minDis = min(minDis, dis);
		}
		if (minDis < 0.2)
		{
			vertexTypeList[v_it->idx()] = 1;
		}
		else
		{
			vertexTypeList[v_it->idx()] = 0;
		}

		for (int i = 0; i < selectedVertices.size(); i++)
		{
			vertexTypeList[selectedVertices[i]->idx()] = 2;

		}
	}
}

void MeshDeformation::AddInteriorConstrainst()
{

	//generate N< n_vertices() random numbers
	int N = N_cons;


	vector<Number>   initialRandList;
	vector<openMesh::VertexIter>  vertexIterList;
	vector<int>  randList;

	initialRandList.clear();
	int seed = rand();
	srand(seed);
	for (int i = 0; i < mesh.n_vertices(); i++)
	{
		Number  number(rand() % (5 * mesh.n_vertices()), i);
		initialRandList.push_back(number);

	}
std:sort(initialRandList.begin(), initialRandList.end());

	vertexIterList.clear();
	openMesh::VertexIter          v_it, v_end(mesh.vertices_end());

	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		vertexIterList.push_back(v_it);
	}

	randList.clear();

	for (int i = 0; i < initialRandList.size(); i++)
	{
		if (randList.size() <= N && !mesh.is_boundary(*vertexIterList[initialRandList[i].id]))
		{
			randList.push_back(initialRandList[i].id);
			//cout<<randList[randList.size()-1]<<endl;

		}

	}

	//add distance constraints

	float weight = interior_weight;
	int cur_col_count = diferentialCoordinateList.size();
	for (int i = 0; i < N; i++)
	{
		openMesh::Point  pt = mesh.point(*(vertexIterList[randList[i]]));
		constraintVerList.push_back(*(vertexIterList[randList[i]]));

		int row = i + cur_col_count;
		int col = randList[i];
		MatElement matEle(row, col, weight);
		laplaceMat.push_back(matEle);

		int constrain_id = vertexIterList[randList[i]]->idx();
		openMesh::Point  cons_pt = constraintMesh.point(constraintMeshVerHandleList[constrain_id]);

		Vector3 constrains_coor = Vector3(pt[0], pt[1], cons_pt[2])*weight;
		diferentialCoordinateList.push_back(constrains_coor);
	}
}

void MeshDeformation::CreateBoundaryList()
{

	//Find a boundary vertex as a starting 
	openMesh::VertexHandle v0;
	openMesh::VertexIter   v_it, v_end(mesh.vertices_end());
	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		if (mesh.is_boundary(*v_it))
		{
			v0 = *v_it;
			break;
		}
	}

	//boundaryVerList.push_back(v0);

	openMesh::HalfedgeHandle heh, heh_init;

	// Get the halfedge handle assigned to v0
	heh = heh_init = mesh.halfedge_handle(v0);
	v0 = mesh.to_vertex_handle(heh);
	boundaryVerList.push_back(v0);

	// heh now holds the handle to the initial halfedge.
	// We now get further on the boundary by requesting
	// the next halfedge adjacent to the vertex heh
	// points to...
	heh = mesh.next_halfedge_handle(heh);
	// We can do this as often as we want:
	while (heh != heh_init)
	{
		v0 = mesh.to_vertex_handle(heh);
		boundaryVerList.push_back(v0);
		heh = mesh.next_halfedge_handle(heh);

	}
}

void MeshDeformation::CreateDistance2D()
{
	float* ptList = new float[2 * boundaryVerList.size()];
	//ptList : x,y x,y x,y ...
	for (int i = 0; i < boundaryVerList.size(); i++)
	{
		openMesh::Point pt = mesh.point(boundaryVerList[i]);
		ptList[2 * i + 0] = pt[0];
		ptList[2 * i + 1] = pt[1];
	}


	//����df���verlist(�߽�����)
	df.CreateVerList(ptList, boundaryVerList.size());
	//
	df.CreateDisList();


	pixelTypeList = new int[PARTICLE_RES*PARTICLE_RES];

	for (int i = 0; i < PARTICLE_RES; i++)
		for (int j = 0; j < PARTICLE_RES; j++)
		{
			if (df.GetDistance(j, i) <= 0)
			{
				pixelTypeList[i*PARTICLE_RES + j] = 1;
			}
			else
				pixelTypeList[i*PARTICLE_RES + j] = 0;

		}



	delete[] ptList;
}

/*
void MeshDeformation::DrawDistance2D()
{
	df.DrawDistance();
}
*/

//void MeshDeformation::DrawHeightField()
//{
//	if (frontHF == NULL || behindHF == NULL)
//		return;
//
//	if (pixelTypeList == NULL)
//		return;
//
//	glEnable(GL_LIGHTING);
//
//	for (int i = 0; i < PARTICLE_RES - 1; i++)
//		for (int j = 0; j < PARTICLE_RES - 1; j++)
//		{
//			if (pixelTypeList[j*PARTICLE_RES + i] == 1 && pixelTypeList[j*PARTICLE_RES + i + 1] == 1 && pixelTypeList[(j + 1)*PARTICLE_RES + i] == 1 && pixelTypeList[(j + 1)*PARTICLE_RES + i + 1] == 1)
//			{
//				float normal0[3], normal1[3], PA[3], PB[3], PC[3], PD[3];
//				PA[0] = df.GetPos(i, j).x;
//				PA[1] = df.GetPos(i, j).y;
//				PA[2] = frontHF[j*PARTICLE_RES + i];
//				PB[0] = df.GetPos(i + 1, j).x;
//				PB[1] = df.GetPos(i + 1, j).y;
//				PB[2] = frontHF[j*PARTICLE_RES + i + 1];
//				PC[0] = df.GetPos(i + 1, j + 1).x;
//				PC[1] = df.GetPos(i + 1, j + 1).y;
//				PC[2] = frontHF[(j + 1)*PARTICLE_RES + i + 1];
//				PD[0] = df.GetPos(i, j + 1).x;
//				PD[1] = df.GetPos(i, j + 1).y;
//				PD[2] = frontHF[(j + 1)*PARTICLE_RES + i];
//				ComputeTriangleNormal(normal0, PA, PB, PC);
//				ComputeTriangleNormal(normal1, PA, PC, PD);
//
//				glNormal3fv(normal0);
//				glBegin(GL_TRIANGLES);
//				glVertex3fv(PA);
//				glVertex3fv(PB);
//				glVertex3fv(PC);
//				glEnd();
//
//				glNormal3fv(normal1);
//				glBegin(GL_TRIANGLES);
//				glVertex3fv(PA);
//				glVertex3fv(PC);
//				glVertex3fv(PD);
//				glEnd();
//			}
//
//		}
//
//	//behind height field
//
//	for (int i = 0; i < PARTICLE_RES - 1; i++)
//		for (int j = 0; j < PARTICLE_RES - 1; j++)
//		{
//			if (pixelTypeList[j*PARTICLE_RES + i] == 1 && pixelTypeList[j*PARTICLE_RES + i + 1] == 1 && pixelTypeList[(j + 1)*PARTICLE_RES + i] == 1 && pixelTypeList[(j + 1)*PARTICLE_RES + i + 1] == 1)
//			{
//				float normal0[3], normal1[3], PA[3], PB[3], PC[3], PD[3];
//				PA[0] = df.GetPos(i, j).x;
//				PA[1] = df.GetPos(i, j).y;
//				PA[2] = behindHF[j*PARTICLE_RES + i];
//				PB[0] = df.GetPos(i + 1, j).x;
//				PB[1] = df.GetPos(i + 1, j).y;
//				PB[2] = behindHF[j*PARTICLE_RES + i + 1];
//				PC[0] = df.GetPos(i + 1, j + 1).x;
//				PC[1] = df.GetPos(i + 1, j + 1).y;
//				PC[2] = behindHF[(j + 1)*PARTICLE_RES + i + 1];
//				PD[0] = df.GetPos(i, j + 1).x;
//				PD[1] = df.GetPos(i, j + 1).y;
//				PD[2] = behindHF[(j + 1)*PARTICLE_RES + i];
//				ComputeTriangleNormal(normal0, PA, PC, PB);
//				ComputeTriangleNormal(normal1, PA, PD, PC);
//
//				glNormal3fv(normal0);
//				glBegin(GL_TRIANGLES);
//				glVertex3fv(PA);
//				glVertex3fv(PC);
//				glVertex3fv(PB);
//
//				glEnd();
//
//				glNormal3fv(normal1);
//				glBegin(GL_TRIANGLES);
//				glVertex3fv(PA);
//				glVertex3fv(PD);
//				glVertex3fv(PC);
//				glEnd();
//			}
//
//		}
//}

void MeshDeformation::ComputeTriangleNormal(float normal[3], float PA[3], float PB[3], float PC[3])
{
	float vecAB[3], vecAC[3];
	for (int i = 0; i < 3; i++)
	{
		vecAB[i] = PB[i] - PA[i];
		vecAC[i] = PC[i] - PA[i];
	}

	normal[0] = vecAB[1] * vecAC[2] - vecAB[2] * vecAC[1];
	normal[1] = vecAB[2] * vecAC[0] - vecAB[0] * vecAC[2];
	normal[2] = vecAB[0] * vecAC[1] - vecAB[1] * vecAC[0];

	float  len = sqrtf(normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]);
	for (int i = 0; i < 3; i++)
	{
		normal[i] /= len;
	}
}

bool MeshDeformation::isProbabilityGreater(float threshold)
{
	int seed = rand();
	srand(seed);

	int ext = 1000;
	int smallNumber = int(ext*threshold);
	int count = 0;
	for (int i = 0; i < ext; i++)
	{

		if (rand() % ext > smallNumber)
		{
			count++;
		}

	}

	return   count > smallNumber;
}

float MeshDeformation::distanceToVolumeBoudary(float x, float y, float z)
{
	Vector3 pt(x, y, z);
	float  dis = MAXVAL;
	for (int i = 0; i < PARTICLE_RES; i++)
		for (int j = 0; j < PARTICLE_RES; j++)
		{
			if (pixelTypeList[i*PARTICLE_RES + j] == 1)
			{
				float cur_x = df.GetPos(j, i).x;
				float cur_y = df.GetPos(j, i).y;
				Vector3 pt_Front = Vector3(cur_x, cur_y, frontHF[i*PARTICLE_RES + j]);
				Vector3 pt_Back = Vector3(cur_x, cur_y, behindHF[i*PARTICLE_RES + j]);
				float tmpDis = Dist(pt_Front, pt);
				if (tmpDis < dis)
					dis = tmpDis;
				tmpDis = Dist(pt_Back, pt);
				if (tmpDis < dis)
					dis = tmpDis;

			}

		}

	return dis;
}

void MeshDeformation::ExportCloudModel(char* cloudfile)
{
	FILE* fp = NULL;
	fp = fopen(cloudfile, "wb");
	if (!fp) return;

	fwrite(&puffNumber, sizeof(int), 1, fp);

	for (int i = 0; i < puffNumber; i++)
	{
		fwrite(&puffPosVec[i], sizeof(Vector3), 1, fp);

	}
	for (int i = 0; i < puffNumber; i++)
	{
		fwrite(&puffSizeVec[i], sizeof(float), 1, fp);
	}
	for (int i = 0; i < puffNumber; i++)
	{
		fwrite(&puffColorVec[i], sizeof(Color4), 1, fp);
	}

	fclose(fp);
}

void MeshDeformation::CloudSampling()
{

	float smallValue = 1.2;
	float bigValue = 1.5;

	float interval = 2.0 / (PARTICLE_RES - 1);
	float scale = 40;

	puffNumber = 0;

	float* disField = new float[PARTICLE_RES*PARTICLE_RES*PARTICLE_RES];
	float maxDis = -9999;
	if (disField == NULL)
	{
		cout << "Memory failed" << endl;
		exit(1);
	}

	for (int i = 0; i < PARTICLE_RES; i++)
		for (int j = 0; j < PARTICLE_RES; j++)
			for (int k = 0; k < PARTICLE_RES; k++)
			{
				//sample center
				float x = -1 + k*interval;
				float y = -1 + j*interval;
				float z = -1 + i*interval;

				float height[2] = { frontHF[j*PARTICLE_RES + k],behindHF[j*PARTICLE_RES + k] };

				if (height[0] - height[1] > F_ZERO&& z >= height[1] && z <= height[0])
				{
					disField[i*PARTICLE_RES*PARTICLE_RES + j*PARTICLE_RES + k] = distanceToVolumeBoudary(x, y, z);
					if (disField[i*PARTICLE_RES*PARTICLE_RES + j*PARTICLE_RES + k] > maxDis)
						maxDis = disField[i*PARTICLE_RES*PARTICLE_RES + j*PARTICLE_RES + k];
				}
				else
				{

					disField[i*PARTICLE_RES*PARTICLE_RES + j*PARTICLE_RES + k] = 0.002;
				}

			}


	for (int i = 0; i < PARTICLE_RES; i++)
		for (int j = 0; j < PARTICLE_RES; j++)
			for (int k = 0; k < PARTICLE_RES; k++)
			{
				//sample center
				float x = -1 + k*interval;
				float y = -1 + j*interval;
				float z = -1 + i*interval;

				float height[2] = { frontHF[j*PARTICLE_RES + k],behindHF[j*PARTICLE_RES + k] };
				if (height[0] - height[1] > 0 && z >= height[1] && z <= height[0])
				{

					float normalDis = disField[i*PARTICLE_RES*PARTICLE_RES + j*PARTICLE_RES + k] / maxDis;
					if (isProbabilityGreater(normalDis))
					{
						Vector3 curPuf;
						//scale means  the extent of  x or y of cloud puff in [ -scale,scale ]

						int  seed = rand() % 99999;
						srand(seed);
						float disturb0 = ((rand() % 100) / 100.0 - 0.5) / 2; //[-1,  1]
						float disturb1 = ((rand() % 100) / 100.0 - 0.5) / 2; //[-1,  1]
						float disturb2 = ((rand() % 100) / 100.0 - 0.5) / 2; //[-1,  1]

					 //   disturb0=0;
						//disturb1=0;
						//disturb2=0;


						curPuf.x = scale*(x + disturb0*interval);
						curPuf.y = scale*(y + disturb1*interval);
						curPuf.z = scale*(z + disturb2*interval);

						puffPosVec.push_back(curPuf);

						float puffSize;
						puffSize = (smallValue + normalDis*(bigValue - smallValue))*interval;
						puffSize = max(puffSize, (float)0.002)*scale;

						puffSizeVec.push_back(puffSize);

						Color4  color(1.0, 0.0, 0.0, 1.0);

						puffColorVec.push_back(color);
					}
				}


				int id = PARTICLE_RES*PARTICLE_RES*PARTICLE_RES - i*PARTICLE_RES*PARTICLE_RES + j*PARTICLE_RES + k;
				if (id % 100 == 0)
					cout << "Left:  " << id / 100 << endl;


			}


	puffNumber = puffPosVec.size();

	cout << "Particle: " << puffNumber << endl;
	delete[] disField;

	cout << "cloud sampling---------------done!" << endl;
}

//void MeshDeformation::DrawConstraintMesh()
//{
//	glEnable(GL_LIGHTING);
//
//	constraintMesh.update_normals();
//
//	openMesh::FaceIter          f_it, f_end(constraintMesh.faces_end());
//	for (f_it = constraintMesh.faces_begin(); f_it != f_end; ++f_it)
//	{
//		openMesh::FaceVertexIter  fv_it, fv_end(constraintMesh.fv_end(f_it));
//
//		openMesh::Normal   normal = constraintMesh.normal(f_it);
//		Vector3  normalVec(normal[0], normal[1], normal[2]);
//		glNormal3fv(!normalVec);
//		glBegin(GL_POLYGON);
//		for (fv_it = constraintMesh.fv_begin(f_it); fv_it != fv_end; ++fv_it)
//		{
//			openMesh::Point  pt = constraintMesh.point(fv_it);
//			Vector3 ver(pt[0], pt[1], pt[2]);
//			glVertex3fv(!ver);
//
//		}
//		glEnd();
//
//
//	}
//
//
//
//	glPointSize(3.0);
//
//	glDisable(GL_LIGHTING);
//	glColor3f(1, 0, 0);
//	glBegin(GL_POINTS);
//	openMesh::VertexIter          v_it, v_end(constraintMesh.vertices_end());
//	for (v_it = constraintMesh.vertices_begin(); v_it != v_end; ++v_it)
//	{
//		if (constraintMesh.is_boundary(v_it))
//		{
//			openMesh::Point  pt = constraintMesh.point(v_it);
//			Vector3 ver(pt[0], pt[1], pt[2]);
//			glColor3f(1, 0, 0);
//			glVertex3fv(!ver);
//		}
//
//	}
//	glEnd();
//
//
//	glBegin(GL_POINTS);
//	for (int i = 0; i < constraintVerList.size(); i++)
//	{
//		openMesh::Point  cur_pt = constraintMesh.point(constraintMeshVerHandleList[constraintVerList[i].idx()]);
//		Vector3 ver(cur_pt[0], cur_pt[1], cur_pt[2]);
//		glColor3f(0, 1, 0);
//		glVertex3fv(!ver);
//
//	}
//	glEnd();
//}

void MeshDeformation::CreateOptimizedMesh(int loop)
{
	for (int i = 0; i < loop; i++)
	{
		cout << "Iterative step: " << i << endl;

		std::cout << "little step 1: update constraint mesh" << std::endl;
		UpdateConstraintMesh(i);
		std::cout << "little step 2: create laplace matrix" << std::endl;
		CreateLaplaceMatrix();
		std::cout << "little step 3: create difcoorlist" << std::endl;
		CreateDifCoorList();
		std::cout << "little step 4: add boundary constrainst" << std::endl;
		AddBoudaryContrainst();
		std::cout << "little step 5: add interior constrainst" << std::endl;
		AddInteriorConstrainst();
		std::cout << "little step 6: update mesh" << std::endl;
		bool upsuccess = UpdateMesh();
		std::cout << "little step 7: optimize mesh" << std::endl;
		OptimizeMesh();
		std::cout << "Iterative step " << i << " done!" << std::endl;

	}
}

void MeshDeformation::UpdateConstraintMesh(int loop)
{
	if (loop == 0)  //first, we use the assumption of [Dobahsi10] as the first shape constraint
	{
		std::cout << "loop == 0" << std::endl;
		int nv = constraintMesh.n_vertices();
		std::cout << "constraintMesh has " << nv << " vertices!" << std::endl;
		//compute the distance to the boundary for all interior vertices.
		//�Ȱ�boundaryvertex��������
		std::vector<openMesh::VertexHandle> boundaryVerList;
		openMesh::VertexHandle v0;
		openMesh::HalfedgeHandle heh, heh_init, heh_pre;

		float* distance2Boundary_ = new float[constraintMesh.n_vertices()];
		openMesh::VertexIter  v_it, v_end(constraintMesh.vertices_end());
		openMesh::VertexIter  v_it2, v_end2(constraintMesh.vertices_end());
		v_it = constraintMesh.vertices_begin();
		//cout << *v_it << endl;
		//cout << *v_end << endl;
		//��boundaryvertex������
		for (v_it = constraintMesh.vertices_begin(); v_it != v_end; v_it++)
		{
			if (!constraintMesh.is_boundary(*v_it))
			{
				//v0 = *v_it;
				//cout << *v_it << endl;
				v0 = *v_it;
				break;
			}
		}
		std::cout << "Found the first boundary ver v0!" << std::endl;

		heh = heh_init = constraintMesh.halfedge_handle(v0);
		v0 = constraintMesh.to_vertex_handle(heh);
		boundaryVerList.push_back(v0);

		heh = constraintMesh.next_halfedge_handle(heh);
		heh_pre = mesh.prev_halfedge_handle(heh);

		while (heh != heh_init)
		{
			v0 = mesh.to_vertex_handle(heh);
			boundaryVerList.push_back(v0);
			heh = constraintMesh.next_halfedge_handle(heh);
		}
		std::cout << "All " << boundaryVerList.size() << " boundary vertex have been found!" << std::endl;
		//�������еı߽���Ѿ�����boundaryverlist����

		int ncount = 0;

		for (v_it = constraintMesh.vertices_begin(); v_it != v_end; ++v_it, ncount++)
		{
			if (ncount % 1000 == 0)
			{
				std::cout << ncount << " of " << nv << " points dis done!" << std::endl;
			}
			distance2Boundary_[v_it->idx()] = 99999;
			openMesh::Point  cur_pt = constraintMesh.point(*v_it);
			if (constraintMesh.is_boundary(*v_it))
			{
				distance2Boundary_[v_it->idx()] = 0.0;
			}
			else
			{
				/*for (v_it2=constraintMesh.vertices_begin(); v_it2!=v_end2; ++v_it2)
				{
					if(constraintMesh.is_boundary(v_it2))
					{
						openMesh::Point  pt=constraintMesh.point( v_it2 );
						float tempDis=sqrt(powf(cur_pt[0]-pt[0],2)+powf(cur_pt[1]-pt[1],2));
						distance2Boundary_[v_it.handle().idx()]=min(distance2Boundary_[v_it.handle().idx()],tempDis);
					}
				}*/
				//ֱ����boundaryVerList�и��£�����һ��һ����
				for (int i = 0; i < boundaryVerList.size(); i++)
				{
					openMesh::Point pt = constraintMesh.point(boundaryVerList[i]);
					float tempDis = sqrt(powf(cur_pt[0] - pt[0], 2) + powf(cur_pt[1] - pt[1], 2));
					distance2Boundary_[v_it->idx()] = min(distance2Boundary_[v_it->idx()], tempDis);
				}
			}
		}
		std::cout << "first double for iter done!" << std::endl;

		float MaxDis = -99999;
		for (v_it = constraintMesh.vertices_begin(); v_it != v_end; ++v_it)
		{
			MaxDis = max(MaxDis, distance2Boundary_[v_it->idx()]);
		}
		for (v_it = constraintMesh.vertices_begin(); v_it != v_end; ++v_it)
		{
			distance2Boundary_[v_it->idx()] /= MaxDis;
			distance2Boundary_[v_it->idx()] *= dis_scale;
			float   dis = fabs(distance2Boundary_[v_it->idx()]);
			dis = sqrtf(dis_scale*dis_scale - (dis_scale - dis)*(dis_scale - dis));
			distance2Boundary_[v_it->idx()] = dis;
			//cout<<	distance2Boundary_[v_it.handle().idx()]<<endl;
		}
		for (v_it = constraintMesh.vertices_begin(); v_it != v_end; ++v_it)
		{
			openMesh::Point  cur_pt = constraintMesh.point(*v_it);
			cur_pt[2] = distance2Boundary_[v_it->idx()];
			constraintMesh.set_point(*v_it, cur_pt);
		}
		//�����constraintMesh���һ�¿�����ʲô����
		//OpenMesh::IO::write_mesh(constraintMesh,"D:\\yuancodes\\MeshOptimization_2014_2_20\\MeshDeformation\\data\\constraintmesh.off");
		//getchar();

		delete[] distance2Boundary_;
		std::cout << "loop == 0 done!" << std::endl;
	}


	if (loop > 0)
	{
		openMesh::VertexIter  v_it, v_end(deformedMesh.vertices_end());
		for (v_it = deformedMesh.vertices_begin(); v_it != v_end; ++v_it)
		{
			openMesh::Point  cur_pt = deformedMesh.point(*v_it);
			constraintMesh.set_point(constraintMeshVerHandleList[v_it->idx()], cur_pt);
		}

		//�����constraintMesh���һ�¿�����ʲô����
		//OpenMesh::IO::write_mesh(constraintMesh,"D:\\yuancodes\\MeshOptimization_2014_2_20\\MeshDeformation\\data\\constraintmesh.off");
		//getchar();


		//SmoothMesh(constraintMesh,3);

	}
}

void MeshDeformation::CloudSamplingSimulation(char* simulationData)
{
	int SIMULATION_RES = 128;

	float* disField = new float[SIMULATION_RES*SIMULATION_RES*SIMULATION_RES];
	float maxDis = -9999;
	if (disField == NULL)
	{
		std::cout << "Memory failed" << std::endl;
		exit(1);
	}
	FILE* pFilein = fopen(simulationData, "rb");
	fread(disField, sizeof(float), SIMULATION_RES*SIMULATION_RES*SIMULATION_RES, pFilein);
	fclose(pFilein);

	float smallValue = 1.2;
	float bigValue = 2;

	float interval = 2.0 / (SIMULATION_RES - 1);
	float scale = 40;

	puffNumber = 0;



	for (int i = 0; i < SIMULATION_RES; i++)
		for (int j = 0; j < SIMULATION_RES; j++)
			for (int k = 0; k < SIMULATION_RES; k++)
			{
				//sample center
				float x = -1 + k*interval;
				float y = -1 + j*interval;
				float z = -1 + i*interval;
				if (disField[i*SIMULATION_RES*SIMULATION_RES + j*SIMULATION_RES + k] > 0)
				{

					float normalDis = 0.5;

					Vector3 curPuf;
					//scale means  the extent of  x or y of cloud puff in [ -scale,scale ]
					int  seed = rand() % 99999;
					srand(seed);
					float disturb0 = ((rand() % 100) / 100.0 - 0.5) * 2; //[-1,  1]
					float disturb1 = ((rand() % 100) / 100.0 - 0.5) * 2; //[-1,  1]
					float disturb2 = ((rand() % 100) / 100.0 - 0.5) * 2; //[-1,  1]

					disturb0 = 0;
					disturb1 = 0;
					disturb2 = 0;

					curPuf.x = scale*(x + disturb0*interval);
					curPuf.y = scale*(z + disturb1*interval);
					curPuf.z = scale*(y + disturb2*interval);

					puffPosVec.push_back(curPuf);

					float puffSize;
					puffSize = (smallValue + normalDis*(bigValue - smallValue))*interval;
					puffSize = max(puffSize, (float)0.002)*scale;

					puffSizeVec.push_back(puffSize);

					Color4  color(1.0, 0.0, 0.0, 1.0);

					puffColorVec.push_back(color);
				}


				int id = SIMULATION_RES*SIMULATION_RES*SIMULATION_RES - i*SIMULATION_RES*SIMULATION_RES + j*SIMULATION_RES + k;
				if (id % 100 == 0)
					cout << "Left:  " << id / 100 << endl;


			}


	puffNumber = puffPosVec.size();

	cout << "Particle: " << puffNumber << endl;
	delete[] disField;

	cout << "cloud sampling---------------done!" << endl;



}


void MeshDeformation::ScaleMesh(openMesh& mesh, float x, float y, float z)
{

	openMesh::VertexIter v_it, v_end(mesh.vertices_end());
	for (v_it = mesh.vertices_begin(); v_it != v_end; ++v_it)
	{
		openMesh::Point  pt = mesh.point(*v_it);
		pt[0] *= x;
		pt[1] *= y;
		pt[2] *= z;
		mesh.set_point(*v_it, pt);
	}
}
