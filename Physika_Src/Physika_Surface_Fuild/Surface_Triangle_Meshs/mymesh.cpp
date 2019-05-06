#include "mymesh.h"
#include <utility>
#include <queue>

namesapce Physiak{
CoordSystem gen_coord_system_by_z_axis(MyMesh::Point const &ez) {
	CoordSystem cs;
	cs[2] = ez.normalized();
	cs[0] = MyMesh::Point(0, 0, 1) % cs[2];
	if (cs[0].sqrnorm() < 1e-3)
		cs[0] = MyMesh::Point(0, 1, 0) % cs[2];
	cs[0].normalize();
	cs[1] = cs[2] % cs[0];
	return cs;
}

float calc_average_edge_length(MyMesh const &mesh) {
	int edge_cnt = 0;
	float avg_edge_length = 0;
	for (auto it = mesh.edges_begin(); it != mesh.edges_end(); ++it) {
		edge_cnt++;
		avg_edge_length += mesh.calc_edge_length(*it);
	}
	avg_edge_length /= edge_cnt;
	return avg_edge_length;
}

void insert_vertex_nring(MyMesh const &mesh, MyMesh::VertexHandle vh, int n, std::set<MyMesh::FaceHandle> &set_fh) {
	std::map<MyMesh::VertexHandle, int> map_vh;
	std::queue<MyMesh::VertexHandle> que_vh;
	que_vh.push(vh);
	map_vh[vh] = 0;
	while (!que_vh.empty()) {
		MyMesh::VertexHandle vh = que_vh.front();
		que_vh.pop();
		int d = map_vh[vh];
		if (d >= n)
			continue;
		for (auto vf_it = mesh.cvf_begin(vh); vf_it.is_valid(); ++vf_it) {
			auto f = *vf_it;
			if (set_fh.find(f) == set_fh.end())
				set_fh.insert(f);
		}
		if (d >= n - 1)
			continue;
		for (auto vv_it = mesh.cvv_begin(vh); vv_it.is_valid(); ++vv_it) {
			auto v = *vv_it;
			if (map_vh.find(v) == map_vh.end()) {
				que_vh.push(v);
				map_vh.insert(std::make_pair(v, d + 1));
			}
		}
	}
}

void calc_vertex_nring(MyMesh const &mesh, MyMesh::VertexHandle vh, int n, std::map<MyMesh::VertexHandle, int> &map_vh) {
	map_vh.clear();
	std::queue<MyMesh::VertexHandle> que_vh;
	que_vh.push(vh);
	map_vh[vh] = 0;
	while (!que_vh.empty()) {
		MyMesh::VertexHandle vh = que_vh.front();
		que_vh.pop();
		int d = map_vh[vh];
		if (d >= n)
			continue;
		for (auto vv_it = mesh.cvv_begin(vh); vv_it.is_valid(); ++vv_it) {
			auto v = *vv_it;
			if (map_vh.find(v) == map_vh.end()) {
				que_vh.push(v);
				map_vh.insert(std::make_pair(v, d + 1));
			}
		}
	}
}
}
