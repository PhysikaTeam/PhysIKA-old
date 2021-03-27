#pragma once
#include <iostream>
#include <string>
#include <cmath>

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Constrained_Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_face_base_with_info_2.h>
#include <CGAL/Delaunay_mesher_2.h>
#include <CGAL/Delaunay_mesh_face_base_2.h>
#include <CGAL/Delaunay_mesh_vertex_base_2.h>
#include <CGAL/Delaunay_mesh_size_criteria_2.h>
#include <CGAL/lloyd_optimize_mesh_2.h>
#include <CGAL/Polygon_2.h>

using namespace std;

struct FaceInfo2
{
	FaceInfo2() {}
	int nesting_level;
	bool in_domain() {
		return nesting_level % 2 == 1;
	}
};
//refine函数需用
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Delaunay_mesh_vertex_base_2<K>                Vb;
typedef CGAL::Delaunay_mesh_face_base_2<K>                  Fb;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb>        TDS;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS>  CDT;
typedef CGAL::Delaunay_mesh_size_criteria_2<CDT>            Criteria;
typedef CGAL::Delaunay_mesher_2<CDT, Criteria>              Mesher;
typedef CDT::Vertex_handle                                  Vertex_handle;
typedef CDT::Face_handle                                    Face_handle;
typedef CDT::Point                                          Point;
typedef CGAL::Polygon_2<K>                                  Polygon_2;

//domain需用
typedef CGAL::Triangulation_face_base_with_info_2<FaceInfo2, K>    Fbb;
typedef CGAL::Constrained_triangulation_face_base_2<K, Fbb>        Fb2;
typedef CGAL::Triangulation_data_structure_2<Vb, Fb2>              TDS2;
typedef CGAL::Exact_predicates_tag                                 Itag;
typedef CGAL::Constrained_Delaunay_triangulation_2<K, TDS2, Itag>  CDT2;
typedef CDT::Point                                                 Point2;


void mark_domains(CDT2& ct, CDT2::Face_handle start, int index, std::list<CDT2::Edge>& border)
{
	if (start->info().nesting_level != -1) 
	{
		return;
	}
	std::list<CDT2::Face_handle> queue;
	queue.push_back(start);
	while (!queue.empty()) 
	{
		CDT2::Face_handle fh = queue.front();
		queue.pop_front();
		if (fh->info().nesting_level == -1) 
		{
			fh->info().nesting_level = index;
			for (int i = 0; i < 3; i++) 
			{
				CDT2::Edge e(fh, i);
				CDT2::Face_handle n = fh->neighbor(i);
				if (n->info().nesting_level == -1) 
				{
					if (ct.is_constrained(e)) border.push_back(e);
					else queue.push_back(n);
				}
			}
		}
	}
}

void mark_domains(CDT2& cdt)
{
	for (CDT2::All_faces_iterator it = cdt.all_faces_begin(); it != cdt.all_faces_end(); ++it) 
	{
		it->info().nesting_level = -1;
	}
	std::list<CDT2::Edge> border;
	mark_domains(cdt, cdt.infinite_face(), 0, border);
	while (!border.empty()) 
	{
		CDT2::Edge e = border.front();
		border.pop_front();
		CDT2::Face_handle n = e.first->neighbor(e.second);
		if (n->info().nesting_level == -1) 
		{
			mark_domains(cdt, n, e.first->info().nesting_level + 1, border);
		}
	}
}

struct Triple
{
	int v[3];
	Triple(int v1, int v2, int v3)
	{
		this->v[0] = v1;
		this->v[1] = v2;
		this->v[2] = v3;
	}
	Triple(int v[3])
	{
		this->v[0] = v[0];
		this->v[1] = v[1];
		this->v[2] = v[2];
	}
};

void PrintOff(const vector<Point>& memo, const vector<Triple>& note, string filename)
{
	ofstream out(filename);
	if (!out)
	{
		cout << "文件或文件夹不存在！";
		abort();
	}
	out << "OFF" << endl;

	int vertex_num = memo.size();
	int face_num = note.size();

	out << vertex_num << " " << face_num << " " << 0 << endl;

	for (int i = 0; i < memo.size();i++)
	{
		out << memo[i].x() << " " << memo[i].y() << " " << 0 << endl;
	}
	for (int i = 0; i < note.size(); i++)
	{
		out << 3;
		for (int j = 0; j < 3; j++)
		{
			out << " " << note.at(i).v[j];
		}
		out << endl;
	}
}

void RefineSegmentMesh(Point v1, Point v2, Point v3, map<Point, int>& ftv, vector<Point>& memo, vector<Triple>& note, float min_length)
{
	CDT cdt;
	cdt.insert_constraint(v1, v2);
	cdt.insert_constraint(v2, v3);
	cdt.insert_constraint(v3, v1);

	Mesher mesher(cdt);
	mesher.set_criteria(Criteria(0.005, min_length));
	mesher.refine_mesh();
	//CGAL::lloyd_optimize_mesh_2(cdt, CGAL::parameters::max_iteration_number = 5);

	for (CDT::Finite_faces_iterator fit = cdt.finite_faces_begin(); fit != cdt.finite_faces_end(); fit++)
	{
		int index[3];
		for (int i = 0; i < 3; i++)
		{
			Point p = fit->vertex(i)->point();

			map<Point, int>::const_iterator it=ftv.find(p);
			if (it != ftv.end())
			{
				index[i] = it->second;
			}
			else
			{
				index[i] = ftv.size();
				memo.push_back(p);
				ftv.insert((map<Point2, int>::value_type(p, ftv.size())));
			}
		}
		note.push_back(Triple(index));
	}
}

float ComputeDistance(Point p1, Point p2)
{
	return (sqrt(pow(p2.x() - p1.x(), 2) + pow(p2.y() - p1.y(), 2)));
}

void RefineMesh_Rev(float** points, int n, string output_filename)
{
	Polygon_2 polygon;
	for (int i = 0; i < n; i++)
	{
		polygon.push_back(Point(points[0][i], points[1][i]));
	};

	float min_length = INFINITY;
	for (Polygon_2::Edge_const_iterator eit = polygon.edges_begin(); eit != polygon.edges_end(); eit++)
	{
		double tmp = ComputeDistance(eit->vertex(0), eit->vertex(1));
		if (tmp < min_length)
		{
			min_length = tmp;
		}
	}
	min_length = min_length/5;
	min_length = min(min_length, (float)0.01);

	CDT2 cdt2;
	cdt2.insert_constraint(polygon.vertices_begin(), polygon.vertices_end(), true);
	mark_domains(cdt2);

    map<Point, int> ftv;
	vector<Point> memo;
	vector<Triple> note;
	for (CDT2::Finite_faces_iterator fit = cdt2.finite_faces_begin(); fit != cdt2.finite_faces_end(); ++fit)
	{
		if (fit->info().in_domain())
		{
			//std::cout << fit->vertex(0)->point() << "   " << fit->vertex(1)->point() << "   " << fit->vertex(2)->point() << std::endl;
			RefineSegmentMesh(fit->vertex(0)->point(), fit->vertex(1)->point(), fit->vertex(2)->point(), ftv, memo, note, min_length);
		}
	}

	assert(ftv.size() == memo.size());

	PrintOff(memo, note, output_filename);
}