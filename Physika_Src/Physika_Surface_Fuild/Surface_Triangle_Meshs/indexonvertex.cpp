#include "indexonvertex.h"
#include "Physika_Surface_Fuild/Surface_Utilities/boundrecorder.h"
#include <iostream>
#include <cmath>
#include <queue>
#include <set>
#include <map>

using std::cout;
using std::endl;
namespace Physika{
IndexOnVertex::IndexOnVertex(MyMesh const &mesh, MyMesh::VertexHandle vh, std::set<MyMesh::FaceHandle> const &set_fh) {
	unique_index_number = ++n_instance;
	rebuild_from(mesh, vh, set_fh);
}

IndexOnVertex::~IndexOnVertex() { }

MyMesh::FaceHandle IndexOnVertex::search(MyMesh const &mesh, MyMesh::Point p) const {
	MyMesh::Point rot_p = p;
	if (rot_p[0] < x0 || rot_p[0] >= x0 + nx * dx ||
		rot_p[1] < y0 || rot_p[1] >= y0 + ny * dy) {
		return *mesh.faces_end();
	}
	else {
		int ind_x = (int)((rot_p[0] - x0) / dx);
		int ind_y = (int)((rot_p[1] - y0) / dy);
		// HACK: 为了避免与GPU结果不一致，去掉了索引
		//std::vector<OpenMesh::FaceHandle> const &contain = this->contain[ind_x * ny + ind_y];
		std::vector<OpenMesh::FaceHandle> const &contain = roundFh;
		for (auto it = contain.begin(); it != contain.end(); ++it) {
			auto fv_it = mesh.cfv_begin(*it);
			MyMesh::Point a(mesh.point(*fv_it++));
			MyMesh::Point b(mesh.point(*fv_it++));
			MyMesh::Point c(mesh.point(*fv_it));
			if (on_face_2d(rot_p, plane_map(a), plane_map(b), plane_map(c))) {
				return *it;
			}
		}
		return *mesh.faces_end();
	}
	return *mesh.faces_end();
}

MyMesh::VertexHandle IndexOnVertex::nearest_vertex(MyMesh const &mesh, MyMesh::Point p, std::function<bool(MyMesh::VertexHandle)> condition) const {
	// Todo: 加快搜索速度
	bool is_first = true;
	MyMesh::VertexHandle vh;
	float min_sqrnorm;
	for (auto it = vec_vh.begin(); it != vec_vh.end(); ++it) {
		if (condition(*it)) {
			MyMesh::Point p2(plane_map(mesh.point(*it)));
			float new_sqrnorm = (p2 - p).sqrnorm();
			if (is_first || new_sqrnorm < min_sqrnorm) {
				min_sqrnorm = new_sqrnorm;
				vh = *it;
				is_first = false;
			}
		}
	}
	return is_first ? *mesh.vertices_end() : vh;
}

MyMesh::Point IndexOnVertex::plane_map(MyMesh::Point const &p) const {
	MyMesh::Point a(p - p_pos);
	MyMesh::Point b(a | rot[0], a | rot[1], 0);
	auto b_norm = b.norm();
	return b_norm == 0 ? b : b * (a.norm() / b_norm);
}

MyMesh::Point IndexOnVertex::to_nature_coord(MyMesh::Point const &p) const {
	return p[0] * rot[0] + p[1] * rot[1] + p[2] * rot[2];
}

MyMesh::Point IndexOnVertex::from_nature_coord(MyMesh::Point const &p) const {
	return MyMesh::Point(p | rot[0], p | rot[1], p | rot[2]);
}

MyMesh::Point IndexOnVertex::vertical_offset_point(float offset) const {
	return p_pos + rot[2] * offset;
}

std::vector<MyMesh::VertexHandle> const &IndexOnVertex::get_ring_1_ordered() const {
	return vec_ring_1_ordered;
}

size_t IndexOnVertex::memory_cost() const {
	// Todo: update memory cost for vec_vh & vec_fh
	size_t sum = 0;
	sum += sizeof(*this);
	sum += contain.capacity() * sizeof(contain.front());
	for (auto it = contain.begin(); it != contain.end(); ++it)
		sum += it->capacity() * sizeof(it->front());
	return sum;
}

bool IndexOnVertex::on_face_2d(MyMesh::Point const &p, MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c) {
	auto triangle_area = [](MyMesh::Point const &a, MyMesh::Point const &b) {
		return a[0] * b[1] - a[1] * b[0];
	};
	MyMesh::Point pa = a - p;
	MyMesh::Point pb = b - p;
	MyMesh::Point pc = c - p;
	MyMesh::Point ab = b - a;
	MyMesh::Point ac = c - a;
	auto abc = triangle_area(ab, ac);
	auto pab = triangle_area(pa, pb);
	auto pbc = triangle_area(pb, pc);
	auto pca = triangle_area(pc, pa);
	if (abc < 0) {
		abc = -abc;
		pab = -pab;
		pbc = -pbc;
		pca = -pca;
	}
	float eps = -abc * 1e-3f;
	return pab > eps && pbc > eps && pca > eps;
}

MyMesh::Point IndexOnVertex::index_conv(IndexOnVertex const *index_from, IndexOnVertex const *index_to, MyMesh::Point const &value_from) {
	return coord_conv(index_from->rot, index_to->rot, value_from);
}

MyMesh::Point IndexOnVertex::coord_conv(MyMesh::Point const coord_from[3], MyMesh::Point const coord_to[3], MyMesh::Point const &vec_from) {
	MyMesh::Point axis_z(coord_from[2] % coord_to[2]);
	auto axis_z_norm = axis_z.norm();
	if (axis_z_norm < 2e-4f) {
		MyMesh::Point vec(vec_from[0] * coord_from[0] + vec_from[1] * coord_from[1] + vec_from[2] * coord_from[2]);
		return MyMesh::Point(vec | coord_to[0], vec | coord_to[1], vec | coord_to[2]);
	}
	else {
		axis_z /= axis_z_norm;
		MyMesh::Point axis_from_x(coord_from[2] % axis_z);
		MyMesh::Point axis_to_x(coord_to[2] % axis_z);
		MyMesh::Point vec(vec_from[0] * coord_from[0] + vec_from[1] * coord_from[1] + vec_from[2] * coord_from[2]);
		vec = MyMesh::Point(vec | axis_from_x, vec | coord_from[2], vec | axis_z);
		vec = vec[0] * axis_to_x + vec[1] * coord_to[2] + vec[2] * axis_z;
		return MyMesh::Point(vec | coord_to[0], vec | coord_to[1], vec | coord_to[2]);
	}
}

static bool _have_intersection(float x0, float x1, float y0, float y1, MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c) {
	auto divide = [](float ax, float ay, float bx, float by, float cx, float cy, float dx, float dy) {
		// 直线AB是否分隔点C、点D
		auto lx = bx - ax;
		auto ly = by - ay;
		auto dist = [&](float x, float y) {
			return ly * (x - ax) - lx * (y - ay);
		};
		auto t0 = dist(cx, cy);
		auto t1 = dist(dx, dy);
		return (t0 > 0 && t1 < 0) || (t0 < 0 && t1 > 0);
	};
	MyMesh::Point sqr[] = {
		MyMesh::Point(x0, y0, 0),
		MyMesh::Point(x0, y1, 0),
		MyMesh::Point(x1, y1, 0),
		MyMesh::Point(x1, y0, 0)
	};
	MyMesh::Point tri[] = { a, b, c };
	for (int i = 0; i < 3; i++) {
		int i1 = (i + 1) % 3;
		int i2 = (i + 2) % 3;
		int j;
		for (j = 0; j < 4; j++) {
			if (!divide(tri[i][0], tri[i][1], tri[i1][0], tri[i1][1], sqr[j][0], sqr[j][1], tri[i2][0], tri[i2][1]))
				break;
		}
		if (j == 4)
			return false;
	}
	return true;
}

void IndexOnVertex::rebuild_from(MyMesh const &mesh, MyMesh::VertexHandle vh, std::set<MyMesh::FaceHandle> const &set_fh) {
	std::set<MyMesh::VertexHandle> set_vh;
	vec_vh.clear();
	vec_fh.clear();
	vec_ring_1_ordered.clear();
	contain.clear();
	p_vh = vh;
	p_pos = mesh.point(vh);
	rot[2] = mesh.normal(vh);
#if 0
	rot[0] = MyMesh::Point(0, 0, -1) % rot[2];
	if (rot[0].norm() < 1e-3)
		rot[0] = MyMesh::Point(0, 1, 0) % rot[2];
	rot[0].normalize();
#else
	// 随机坐标轴，用于暴露未坐标转换的问题
	do {
		rot[0] = MyMesh::Point((float)rand(), (float)rand(), (float)rand()).normalized() % rot[2];
	} while (rot[0].norm() < 1e-3);
	rot[0].normalize();
#endif
	rot[1] = rot[2] % rot[0];
	if (set_fh.empty())
		return;
	BoundRecorder<float> x_bound, y_bound;
	int edge_cnt = 0;
	float edge_avg_length = 0;
	for (auto it = set_fh.begin(); it != set_fh.end(); ++it) {
		vec_fh.push_back(*it);
		for (auto fv_it = mesh.cfv_begin(*it); fv_it.is_valid(); ++fv_it) {
			if (set_vh.find(*fv_it) == set_vh.end())
				set_vh.insert(*fv_it);
			MyMesh::Point p(mesh.point(*fv_it));
			p = plane_map(p);
			x_bound.insert(p[0]);
			y_bound.insert(p[1]);
		}
		for (auto fe_it = mesh.cfe_begin(*it); fe_it.is_valid(); ++fe_it) {
			edge_cnt++;
			edge_avg_length += mesh.calc_edge_length(*fe_it);
		}
	}
	for (auto it = set_vh.begin(); it != set_vh.end(); ++it)
		vec_vh.push_back(*it);
	edge_avg_length /= edge_cnt;
	dx = dy = edge_avg_length * 0.5f;
	float eps = edge_avg_length * 1e-3f;
	x0 = x_bound.get_min() - eps;
	y0 = y_bound.get_min() - eps;
	auto x1 = x_bound.get_max() + eps;
	auto y1 = y_bound.get_max() + eps;
	nx = (int)((x1 - x0) / dx + 1);
	ny = (int)((y1 - y0) / dy + 1);
	contain.resize(nx * ny);
	for (auto it = set_fh.begin(); it != set_fh.end(); ++it) {
		BoundRecorder<float> x_bound, y_bound;
		for (auto fv_it = mesh.cfv_iter(*it); fv_it.is_valid(); ++fv_it) {
			MyMesh::Point p(mesh.point(*fv_it));
			p = plane_map(p);
			x_bound.insert(p[0]);
			y_bound.insert(p[1]);
		}
		int ind_x0 = (int)((x_bound.get_min() - x0) / dx);
		int ind_x1 = (int)((x_bound.get_max() - x0) / dx);
		int ind_y0 = (int)((y_bound.get_min() - y0) / dy);
		int ind_y1 = (int)((y_bound.get_max() - y0) / dy);
		auto fv_it = mesh.cfv_begin(*it);
		MyMesh::Point a(plane_map(mesh.point(*fv_it++)));
		MyMesh::Point b(plane_map(mesh.point(*fv_it++)));
		MyMesh::Point c(plane_map(mesh.point(*fv_it)));
		for (int i = ind_x0; i <= ind_x1; i++) {
			for (int j = ind_y0; j <= ind_y1; j++) {
				std::vector<OpenMesh::FaceHandle> &contain = this->contain[i * ny + j];
				if (_have_intersection(x0 + i * dx - eps, x0 + (i + 1) * dx + eps, y0 + j * dy - eps, y0 + (j + 1) * dy + eps, a, b, c))
					contain.push_back(*it);
			}
		}
	}
	// 构建vec_ring_1_ordered，在边界时，保证边界点在前2个元素
	std::queue<MyMesh::HalfedgeHandle> que_hh;
	for (auto voh_it = mesh.cvoh_ccwiter(vh); voh_it.is_valid(); ++voh_it)
		que_hh.push(*voh_it);
	if (mesh.is_boundary(vh)) {
		while (!mesh.is_boundary(que_hh.front())) {
			que_hh.push(que_hh.front());
			que_hh.pop();
		}
	}
	while (!que_hh.empty()) {
		vec_ring_1_ordered.push_back(mesh.to_vertex_handle(que_hh.front()));
		que_hh.pop();
	}

	std::set<MyMesh::FaceHandle> s;
	for (auto it = contain.begin(); it != contain.end(); ++it) {
		for (auto f_it = it->begin(); f_it != it->end(); ++f_it) {
			if (s.find(*f_it) == s.end()) {
				s.insert(*f_it);
				roundFh.push_back(*f_it);
			}
		}
	}
}

int IndexOnVertex::n_instance = 0;
}
