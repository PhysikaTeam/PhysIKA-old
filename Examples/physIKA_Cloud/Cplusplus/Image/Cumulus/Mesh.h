#pragma once
#include "Pixel.h"
#include "Image.h"
#include <opencv2/core.hpp>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/lloyd_optimize_mesh_2.h>
#include <CGAL/Polygon_2.h>
#include <CGAL/Polygon_mesh_processing/connected_components.h>
#include<sstream>
using namespace std;

class Mesh {
public:
	float* vertexList;
	int* edgeList;
	int* faceList;
	int ver_number;
	int edge_number;
	int face_number;
	stringstream fin;

	struct FaceInfo2
	{
		FaceInfo2(){}
		int nesting_level;
		bool in_domain(){
			return nesting_level % 2 == 1;
		}
	};

	typedef CGAL::Exact_predicates_inexact_constructions_kernel       K;
	typedef CGAL::Triangulation_vertex_base_2<K>                      Vb;
	typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, K>    Fbb;
	typedef CGAL::Constrained_triangulation_face_base_2<K, Fbb>        Fb;
	typedef CGAL::Triangulation_data_structure_2<Vb, Fb>               TDS;
	typedef CGAL::Exact_predicates_tag                                Itag;
	typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS, Itag>  CDT;
	typedef CDT::Point                                                Point;
	typedef CGAL::Polygon_2<K>                                        Polygon_2;
	// surface mesh
	typedef CGAL::Simple_cartesian<double>       Kernel;
	typedef Kernel::Point_2                      Point_2;
	typedef Kernel::Point_3                      Point_3;
	typedef Kernel::Vector_3                     Vector_3;
	typedef Kernel::Vector_2                     Vector_2;
	typedef Kernel::FT                           FT;
	typedef CGAL::Surface_mesh<Kernel::Point_3>  SurfaceMesh;
	typedef boost::graph_traits<SurfaceMesh>::halfedge_descriptor  halfedge_descriptor;
	typedef boost::graph_traits<SurfaceMesh>::vertex_descriptor    vertex_descriptor;
	typedef boost::graph_traits<SurfaceMesh>::vertex_iterator      vertex_iterator;
	typedef boost::graph_traits<SurfaceMesh>::face_descriptor      face_descriptor;
	typedef boost::graph_traits<SurfaceMesh>::face_iterator      face_iterator;

	vector<cv::Point> contour;
	void delaunay_triangulation(vector<cv::Point> contour);
	void mark_domains(CDT& ct, CDT::Face_handle start, int index, std::list<CDT::Edge>& border);
	void mark_domains(CDT& cdt);


	Mesh();
	void Initial();
	void ImportBaseMesh();
	void CreatBaseMesh();

    //*****************
	//for  image mesh
	float *  Cloud_vertexList;
	int*     Cloud_facelist;
	int      Cloud_vertexnumber;
	int      Cloud_facenumber;
	
};