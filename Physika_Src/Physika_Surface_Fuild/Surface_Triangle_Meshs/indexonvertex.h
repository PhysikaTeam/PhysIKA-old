#ifndef INDEXONVERTEX_H
#define INDEXONVERTEX_H

#include "mymesh.h"
#include <set>
#include <vector>
#include <functional>
#include <algorithm>

class IndexOnVertex {
public:
	IndexOnVertex(MyMesh const &mesh, MyMesh::VertexHandle vh, std::set<MyMesh::FaceHandle> const &set_fh);
	virtual ~IndexOnVertex();
	MyMesh::FaceHandle search(MyMesh const &mesh, MyMesh::Point p) const;
	MyMesh::VertexHandle nearest_vertex(MyMesh const &mesh, MyMesh::Point p, std::function<bool(MyMesh::VertexHandle)> condition) const;
	MyMesh::Point plane_map(MyMesh::Point const &p) const;
	MyMesh::Point to_nature_coord(MyMesh::Point const &p) const;
	float to_nature_coord(float x) const { return x; }
	MyMesh::Point from_nature_coord(MyMesh::Point const &p) const;
	MyMesh::Point vertical_offset_point(float offet) const;
	std::vector<MyMesh::VertexHandle> const &get_ring_1_ordered() const; // 1-ring���ڵ㣬����ڱ߽��ϣ���ôboundary�ĵ��Ȼ��ǰ2��Ԫ��
	size_t memory_cost() const;
	MyMesh::Point *GetRot(){return rot;}
public:
	static bool on_face_2d(MyMesh::Point const &p, MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c);
	inline static float index_conv(IndexOnVertex const *index_from, IndexOnVertex const *index_to, float value_from) { return value_from; }
	// vec_from����index_from����ϵ��XYƽ���ϵ�ʸ����ͨ������ϵ��ת����ת��index_to����ϵ�£�����ֵ����ת���ʸ����index_to����ϵ�µı�ʾ��
	static MyMesh::Point index_conv(IndexOnVertex const *index_from, IndexOnVertex const *index_to, MyMesh::Point const &vec_from);
	static MyMesh::Point coord_conv(MyMesh::Point const coord_from[3], MyMesh::Point const coord_to[3], MyMesh::Point const &vec_from);
private:
	IndexOnVertex(IndexOnVertex const &);
	IndexOnVertex &operator=(IndexOnVertex const &);
	void rebuild_from(MyMesh const &mesh, MyMesh::VertexHandle vh, std::set<MyMesh::FaceHandle> const &set_fh);
public: //private:
	static int n_instance;
	int unique_index_number;
	MyMesh::VertexHandle p_vh;
	MyMesh::Point p_pos;
	MyMesh::Point rot[3];
	float x0, y0;
	float dx, dy;
	int nx, ny;
	std::vector<MyMesh::VertexHandle> vec_vh;
	std::vector<MyMesh::FaceHandle> vec_fh;
	std::vector<MyMesh::VertexHandle> vec_ring_1_ordered;
	std::vector<std::vector<OpenMesh::FaceHandle> > contain;
	std::vector<MyMesh::FaceHandle> roundFh;
};

#endif
