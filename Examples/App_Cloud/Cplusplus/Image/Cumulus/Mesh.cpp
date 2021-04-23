#include "Mesh.h"

#include"delaunay.h"
#include"edge.h"
#include"numeric.h"
#include"vector2.h"


extern void RefineMesh_Rev(float** points, int n, string output_filename);

Mesh::Mesh()
{
	cout << "creat a mesh object" << endl;
}

void Mesh::Initial()
{
	vertexList = NULL;
	edgeList = NULL;
	faceList = NULL;
	ver_number = 0;
	edge_number = 0;
	face_number = 0;

	//------------

}

void Mesh::mark_domains(CDT& ct, CDT::Face_handle start, int index, std::list<CDT::Edge>& border)
{
	if (start->info().nesting_level != -1){
		return;
	}
	std::list<CDT::Face_handle> queue;
	queue.push_back(start);
	while (!queue.empty()){
		CDT::Face_handle fh = queue.front();
		queue.pop_front();
		if (fh->info().nesting_level == -1){
			fh->info().nesting_level = index;
			for (int i = 0; i < 3; i++){
				CDT::Edge e(fh, i);
				//CDT::Vertex v(fh, i);
				CDT::Face_handle n = fh->neighbor(i);
				if (n->info().nesting_level == -1){
					if (ct.is_constrained(e)) border.push_back(e);
					else queue.push_back(n);
				}
			}
		}
	}
}

void Mesh::mark_domains(CDT& cdt)
{
	for (CDT::All_faces_iterator it = cdt.all_faces_begin(); it != cdt.all_faces_end(); ++it){
		it->info().nesting_level = -1;
	}
	std::list<CDT::Edge> border;
	mark_domains(cdt, cdt.infinite_face(), 0, border);
	while (!border.empty()){
		CDT::Edge e = border.front();
		border.pop_front();
		CDT::Face_handle n = e.first->neighbor(e.second);
		if (n->info().nesting_level == -1){
			mark_domains(cdt, n, e.first->info().nesting_level + 1, border);
		}
	}
}

void Mesh::delaunay_triangulation(vector<cv::Point> contour)
{
	//construct two non-intersecting nested polygons
	Polygon_2 polygon1;
	for (int i = 0; i < contour.size(); i++) {
		cv::Point p = contour[i];
		polygon1.push_back(Point(p.x, p.y));
	}
	SurfaceMesh sm;

	//Insert the polygons into a constrained triangulation
	CDT cdt;
	cdt.insert_constraint(polygon1.vertices_begin(), polygon1.vertices_end(), true);
	//    cdt.insert_constraint(polygon2.vertices_begin(), polygon2.vertices_end(), true);

	//Mark facets that are inside the domain bounded by the polygon
	mark_domains(cdt);
	int face_count = 0;
	int count = 0;
	for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin();
		fit != cdt.finite_faces_end(); ++fit)
	{

		face_count++;
		Point pt = fit->vertex(1)->point();
		//        CDT::Face_handle fh = ;
		//        cout << fit->vertex(1)->info() << endl;
		if(fit->info().in_domain()){
			++count;
			Point   pt0 = fit->vertex(0)->point(),
				pt1 = fit->vertex(1)->point(),
				pt2 = fit->vertex(2)->point();

			vertex_descriptor vd0 = sm.add_vertex(Point_3(pt0.x(), pt0.y(), 0));
			vertex_descriptor vd1 = sm.add_vertex(Point_3(pt1.x(), pt1.y(), 0));
			vertex_descriptor vd2 = sm.add_vertex(Point_3(pt2.x(), pt2.y(), 0));

			sm.add_face(vd0, vd1, vd2);
		}
	}
	double target_edge_length = 4;
	CGAL::Polygon_mesh_processing::stitch_borders(sm);
	CGAL::Polygon_mesh_processing::isotropic_remeshing(CGAL::faces(sm), target_edge_length, sm);
	CGAL::Polygon_mesh_processing::keep_largest_connected_components(sm, 1);
	
	fin << sm;
	
	
	//ofstream os("../output/basemesh.off");
	//os << sm;
	//os.close();
	//cout << "face count is " << face_count << endl;
	//std::cout << "There are " << count << " facets in the domain." << std::endl;
	
}

void Mesh::ImportBaseMesh()
{
	
	//ifstream fin;
	//fin.open("../output/basemesh.off", std::ofstream::in);
	//if (fin.fail()){
	//	std::cerr << "打开新文件失败！" << std::endl;
	//}
	//else{
	//	string s;
	//	//空过第一行
	//	getline(fin, s);
	//	//读节点数、面片数、边数
	//	getline(fin, s);
	//	size_t pos = s.find(' ');
	//	ver_number = atoi(s.substr(0, pos).c_str());
	//	s = s.substr(pos + 1);
	//	pos = s.find(' ');
	//	face_number = atoi(s.substr(0, pos).c_str());
	//	edge_number = atoi(s.substr(pos + 1).c_str());
	//	vertexList = new float[ver_number * 3];
	//	faceList = new int[face_number * 4];
	//	for (int i = 0; i < ver_number; i++){
	//		float x, y, z;
	//		getline(fin, s);
	//		pos = s.find(' ');
	//		x = atof(s.substr(0, pos).c_str());
	//		s = s.substr(pos + 1);
	//		pos = s.find(' ');
	//		y = atof(s.substr(0, pos).c_str());
	//		z = atof(s.substr(pos + 1).c_str());
	//		vertexList[3 * i + 0] = x;
	//		vertexList[3 * i + 1] = y;
	//		vertexList[3 * i + 2] = z;
	//	}
	//	for (int i = 0; i < face_number; i++){
	//		int num, a, b, c;
	//		getline(fin, s);
	//		pos = s.find(' ');
	//		num = atoi(s.substr(0, pos).c_str());
	//		s = s.substr(pos + 1);
	//		pos = s.find(' ');
	//		a = atoi(s.substr(0, pos).c_str());
	//		s = s.substr(pos + 1);
	//		pos = s.find(' ');
	//		b = atoi(s.substr(0, pos).c_str());
	//		c = atoi(s.substr(pos + 1).c_str());
	//		faceList[4 * i + 0] = num;
	//		faceList[4 * i + 1] = a;
	//		faceList[4 * i + 2] = b;
	//		faceList[4 * i + 3] = c;
	//	}
	//}
	//fin.close();
	
	
	string s;
	//空过第一行
	getline(fin, s);
	//读节点数、面片数、边数
	getline(fin, s);
	size_t pos = s.find(' ');
	ver_number = atoi(s.substr(0, pos).c_str());
	s = s.substr(pos + 1);
	pos = s.find(' ');
	face_number = atoi(s.substr(0, pos).c_str());
	edge_number = atoi(s.substr(pos + 1).c_str());
	vertexList = new float[ver_number * 3];
	faceList = new int[face_number * 4];
	for (int i = 0; i < ver_number; i++) {
		float x, y, z;
		getline(fin, s);
		pos = s.find(' ');
		x = atof(s.substr(0, pos).c_str());
		s = s.substr(pos + 1);
		pos = s.find(' ');
		y = atof(s.substr(0, pos).c_str());
		z = atof(s.substr(pos + 1).c_str());
		vertexList[3 * i + 0] = x;
		vertexList[3 * i + 1] = y;
		vertexList[3 * i + 2] = z;
	}
	for (int i = 0; i < face_number; i++) {
		int num, a, b, c;
		getline(fin, s);
		pos = s.find(' ');
		num = atoi(s.substr(0, pos).c_str());
		s = s.substr(pos + 1);
		pos = s.find(' ');
		a = atoi(s.substr(0, pos).c_str());
		s = s.substr(pos + 1);
		pos = s.find(' ');
		b = atoi(s.substr(0, pos).c_str());
		c = atoi(s.substr(pos + 1).c_str());
		faceList[4 * i + 0] = num;
		faceList[4 * i + 1] = a;
		faceList[4 * i + 2] = b;
		faceList[4 * i + 3] = c;
	}
	
}

void Mesh::CreatBaseMesh()
{
	//这里开始！
	delaunay_triangulation(contour);
	ImportBaseMesh();
	//int n = contour.size();
	//float** points = new float*[2];
	//points[0] = new float[n];
	//points[1] = new float[n];
	//for (int i = 0; i < n; ++i) {
	//	points[0][i] = (float)contour[i].x;
	//	points[1][i] = (float)contour[i].y;
	//}
	//string output_filename = "../output/basemesh.off";
	//RefineMesh_Rev(points, n, output_filename);
	//delete[] points[0];
	//delete[] points[1];
	//delete[] points;
}

