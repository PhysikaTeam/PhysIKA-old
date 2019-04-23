#include "indexofmesh.h"
#include "indexonvertex.h"
#include "boundrecorder.h"
#include <set>
#include <iostream>
using namespace std;

IndexOfMesh::IndexOfMesh(MyMesh &mesh) {
	create_index(mesh);
}

IndexOfMesh::~IndexOfMesh() {
	release_index();
}

void IndexOfMesh::nearest_intersection(MyMesh::Point base, MyMesh::Point normal, MyMesh::FaceHandle &fh, float &dist) {
	static int cnt = 0;
	cnt++;
	float around_len = 3 * avg_edge_len;
	MyMesh::Point leftbound = base - MyMesh::Point(1, 1, 1) * around_len;
	MyMesh::Point rightbound = base + MyMesh::Point(1, 1, 1) * around_len;
	int i0 = (int)((leftbound[0] - this->base[0]) / delta);
	int i1 = (int)((rightbound[0] - this->base[0]) / delta);
	int j0 = (int)((leftbound[1] - this->base[1]) / delta);
	int j1 = (int)((rightbound[1] - this->base[1]) / delta);
	int k0 = (int)((leftbound[2] - this->base[2]) / delta);
	int k1 = (int)((rightbound[2] - this->base[2]) / delta);
	std::set<MyMesh::FaceHandle> set_fh;
	for (int i = i0; i <= i1; i++)
	for (int j = j0; j <= j1; j++)
	for (int k = k0; k <= k1; k++) {
		int mapind = toMapIndex(i, j, k);
		auto it = div.find(mapind);
		if (it == div.end())
			continue;
		for (auto vec_it = it->second.begin(); vec_it != it->second.end(); ++vec_it) {
			set_fh.insert(*vec_it);
		}
	}
	bool is_first = true;
	float min_dist;
	MyMesh::FaceHandle min_fh;
	for (auto f_it = set_fh.begin(); f_it != set_fh.end(); ++f_it) {
		bool have_intersection;
		float dist;
		calc_intersection(base, normal, *f_it, have_intersection, dist);
		if (have_intersection) {
			if (is_first || fabs(dist) < fabs(min_dist)) {
				min_dist = dist;
				min_fh = *f_it;
				is_first = false;
			}
		}
	}
	if (is_first || fabs(min_dist) > around_len) {
		for (auto f_it = m_mesh->faces_begin(); f_it != m_mesh->faces_end(); ++f_it) {
			bool have_intersection;
			float dist;
			calc_intersection(base, normal, *f_it, have_intersection, dist);
			if (have_intersection) {
				if (is_first || fabs(dist) < fabs(min_dist)) {
					min_dist = dist;
					min_fh = *f_it;
					is_first = false;
				}
			}
		}
	}
	if (is_first) {
		fh = *m_mesh->faces_end();
		dist = 0;
	}
	else {
		fh = min_fh;
		dist = min_dist;
	}
}

void IndexOfMesh::create_index(MyMesh &mesh) {
	m_mesh = &mesh;
	BoundRecorder<float> xbound, ybound, zbound;
	for (auto v_it = m_mesh->vertices_begin(); v_it != m_mesh->vertices_end(); ++v_it) {
		MyMesh::Point p(m_mesh->point(*v_it));
		xbound.insert(p[0]);
		ybound.insert(p[1]);
		zbound.insert(p[2]);
	}
	avg_edge_len = calc_average_edge_length(*m_mesh);
	delta = 2.0f * avg_edge_len;
	base = MyMesh::Point(xbound.get_min(), ybound.get_min(), zbound.get_min());
	base -= MyMesh::Point(1, 1, 1) * delta * 1e-3f;
	MyMesh::Point upbound(xbound.get_max(), ybound.get_max(), zbound.get_max());
	upbound += MyMesh::Point(1, 1, 1) * delta * 1e-3f;
	nx = (int)((upbound[0] - base[0]) / delta) + 1;
	ny = (int)((upbound[1] - base[1]) / delta) + 1;
	nz = (int)((upbound[2] - base[2]) / delta) + 1;
	for (auto f_it = m_mesh->faces_begin(); f_it != m_mesh->faces_end(); ++f_it) {
		BoundRecorder<float> xbound, ybound, zbound;
		for (auto fv_it = m_mesh->fv_iter(*f_it); fv_it.is_valid(); ++fv_it) {
			MyMesh::Point p(m_mesh->point(*fv_it));
			xbound.insert(p[0]);
			ybound.insert(p[1]);
			zbound.insert(p[2]);
		}
		int i0 = (int)((xbound.get_min() - base[0]) / delta);
		int i1 = (int)((xbound.get_max() - base[0]) / delta);
		int j0 = (int)((ybound.get_min() - base[1]) / delta);
		int j1 = (int)((ybound.get_max() - base[1]) / delta);
		int k0 = (int)((zbound.get_min() - base[2]) / delta);
		int k1 = (int)((zbound.get_max() - base[2]) / delta);
		for (int i = i0; i <= i1; i++)
		for (int j = j0; j <= j1; j++)
		for (int k = k0; k <= k1; k++) {
			int mapind = toMapIndex(i, j, k);
			div[mapind].push_back(*f_it);
		}
	}
}

void IndexOfMesh::release_index() {

}

int IndexOfMesh::toMapIndex(int i, int j, int k) const {
	return i * ny * nz + j * nz + k;
}

void IndexOfMesh::calc_intersection(MyMesh::Point base, MyMesh::Point direct, MyMesh::FaceHandle fh, bool &have_intersection, float &dist) {
	have_intersection = true;
	dist = 0;
	auto fv_it = m_mesh->fv_iter(fh);
	MyMesh::Point a = m_mesh->point(*fv_it++);
	MyMesh::Point b = m_mesh->point(*fv_it++);
	MyMesh::Point c = m_mesh->point(*fv_it++);
	MyMesh::Point n = ((a - b) % (c - b)).normalized();
	float per_unit = direct | n;
	float total_dist = (a - base) | n;
	if (per_unit == 0) {
		if (total_dist == 0) {
			// Todo: 需要正确处理共面情况
			std::cout << "IndexOfMesh::calc_intersection Todo: 需要正确处理共面情况" << std::endl;
			have_intersection = true;
			dist = 0;
		}
		else {
			have_intersection = false;
		}
	}
	else {
		dist = total_dist / per_unit;
		MyMesh::Point x = (b - a).normalized();
		MyMesh::Point y = n % x;
		MyMesh::Point na(0, 0, 0);
		MyMesh::Point nb = MyMesh::Point((b - a).norm(), 0, 0);
		MyMesh::Point nc = c - a;
		nc = MyMesh::Point(nc | x, nc | y, 0);
		MyMesh::Point p = base + direct * dist;
		MyMesh::Point np = p - a;
		np = MyMesh::Point(np | x, np | y, 0);
		have_intersection = IndexOnVertex::on_face_2d(np, na, nb, nc);
	}
}
