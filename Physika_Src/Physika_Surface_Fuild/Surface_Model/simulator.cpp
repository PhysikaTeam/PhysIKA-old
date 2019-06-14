#include "simulator.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/indexonvertex.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/indexofmesh.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/cvfem.h"
#include "Physika_Surface_Fuild/Surface_Smooth/smooth.h"
#include "Physika_Surface_Fuild/Surface_Utilities/boundrecorder.h"
#include "Physika_Surface_Fuild/Surface_Utilities/windowstimer.h"
#include <vector>
#include <queue>
#include <deque>

#pragma warning(disable: 4258)

using namespace std;

namespace Physika{
Simulator::Simulator() { }

Simulator::~Simulator() { }
void Simulator::getheight() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		//height[x_cells*m_mesh.point(*v_it)[0] + m_mesh.point(*v_it)[2]] = m_mesh.property(m_height, *v_it);
		if (m_mesh.property(m_depth, *v_it) > 0) {
			height[m_mesh.point(*v_it)[0] * z_cells + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
		}
		else {
			height[m_mesh.point(*v_it)[0] * z_cells + m_mesh.point(*v_it)[2]] = 0;
		}
	}
}
void Simulator::init(int argc, char** argv) {
	m_situation = 7;
	set_initial_constants();
	generate_origin();
	generate_mesh();
	add_properties();
	m_depth_threshold = (avg_edge_len = calc_average_edge_length(m_mesh)) * 1e-4f;
	m_depth_threshold = 1e-5f;
	add_index_to_vertex(3);
	match_bottom_height();
	calculate_tensor();
	set_initial_conditions();

}
void Simulator::init() {
	add_properties();
	m_depth_threshold = (avg_edge_len = calc_average_edge_length(m_mesh)) * 1e-4f;
	m_depth_threshold = 1e-5f;
	add_index_to_vertex(3);
	match_bottom_height();
	calculate_tensor();
}
void Simulator::belowwatermodel() {
	MyMesh::Point m_rotate_center(0, 0, 0);
	CoordSystem g_system = gen_coord_system_by_z_axis(m_g);
	CoordSystem screen_system = gen_coord_system_by_z_axis(MyMesh::Point(0, -1, 0));
	auto rotate_by_g = [&](MyMesh::Point p) {
			return p;
	};

	// HACK: 避免水在模型下方的问题
	float const h_gain = 0.00f;
	float const cond4_y_min = -20.8f;
	int *vertex_to_idx = new int[m_mesh.n_vertices()];
	int *edge_to_idx = new int[m_mesh.n_edges()];
	memset(vertex_to_idx, 0, m_mesh.n_vertices() * sizeof(int));
	memset(edge_to_idx, 0, m_mesh.n_edges() * sizeof(int));
	for (size_t i = 0; i < m_origin.n_vertices(); i++) {
		MyMesh::VertexHandle vh((int)i);
		MyMesh::Point p(rotate_by_g(m_origin.point(vh)));
	}
	auto have_water = [&](MyMesh::VertexHandle vh) {
		if (m_situation == 4) {
			MyMesh::Point p(m_mesh.point(vh));
			if (p[1] < cond4_y_min)
				return false;
		}
		return m_mesh.property(m_extrapolate_depth, vh) > m_depth_threshold;
	};
	auto water_point = [&](MyMesh::VertexHandle vh) {
		float h = m_mesh.property(m_bottom, vh) + m_mesh.property(m_extrapolate_depth, vh) + h_gain;
		return rotate_by_g(m_mesh.data(vh).index->vertical_offset_point(h));
	};
	int vnum = (int)m_origin.n_vertices();
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		if (have_water(*v_it)) {
			MyMesh::Point p(water_point(*v_it));
			vnum++;
			vertex_to_idx[v_it->idx()] = vnum;
		}
	}
	for (auto e_it = m_mesh.edges_begin(); e_it != m_mesh.edges_end(); ++e_it) {
		auto heh = m_mesh.halfedge_handle(*e_it, 0);
		auto from_v = m_mesh.from_vertex_handle(heh);
		auto to_v = m_mesh.to_vertex_handle(heh);
		if ((have_water(from_v) ? 1 : 0) + (have_water(to_v) ? 1 : 0) == 1) {
			float from_d = m_mesh.property(m_extrapolate_depth, from_v);
			float to_d = m_mesh.property(m_extrapolate_depth, to_v);
			MyMesh::Point from_p = water_point(from_v);
			MyMesh::Point to_p = water_point(to_v);
			float t = (m_depth_threshold - from_d) / (to_d - from_d);
			t = fmin(fmax(t, 0.0f), 1.0f);
			MyMesh::Point p = t * to_p + (1 - t) * from_p;
			vnum++;
			edge_to_idx[e_it->idx()] = vnum;
		}
	}
	for (size_t i = 0; i < m_origin.n_vertices(); i++) {
		MyMesh::VertexHandle vh((int)i);
		MyMesh::Point p(rotate_by_g(m_origin.point(vh)));
		float pic_size = 20.0f;
		float tx = 0;
		float ty = p[2] / 100.0f;
		MyMesh::TexCoord2D tc(tx, ty);
		if (m_origin.has_vertex_texcoords2D())
			tc = m_origin.texcoord2D(vh);
	}
	for (auto f_it = m_origin.faces_begin(); f_it != m_origin.faces_end(); ++f_it) {
		auto fv_it = m_origin.cfv_ccwbegin(*f_it);
		MyMesh::VertexHandle v1 = *fv_it++;
		MyMesh::VertexHandle v2 = *fv_it++;
		MyMesh::VertexHandle v3 = *fv_it++;
		if (m_mesh.point(v1)[0] > 60.1f || m_mesh.point(v2)[0] > 60.1f || m_mesh.point(v3)[0] > 60.1f)
			continue;
		if (m_mesh.point(v1)[2] < -0.1f || m_mesh.point(v2)[2] < -0.1f || m_mesh.point(v3)[2] < -0.1f)
			continue;
		if (m_mesh.point(v1)[2] > 50.1f || m_mesh.point(v2)[2] > 50.1f || m_mesh.point(v3)[2] > 50.1f)
			continue;
	}
	for (auto f_it = m_mesh.faces_begin(); f_it != m_mesh.faces_end(); ++f_it) {
		auto fh_it = m_mesh.cfh_ccwiter(*f_it);
		MyMesh::HalfedgeHandle he1 = *fh_it++;
		MyMesh::HalfedgeHandle he2 = *fh_it++;
		MyMesh::HalfedgeHandle he3 = *fh_it++;
		MyMesh::EdgeHandle e1(m_mesh.edge_handle(he1));
		MyMesh::EdgeHandle e2(m_mesh.edge_handle(he2));
		MyMesh::EdgeHandle e3(m_mesh.edge_handle(he3));
		MyMesh::VertexHandle v1 = m_mesh.from_vertex_handle(he1);
		MyMesh::VertexHandle v2 = m_mesh.from_vertex_handle(he2);
		MyMesh::VertexHandle v3 = m_mesh.from_vertex_handle(he3);
		if (m_mesh.point(v1)[0] > 60.1f || m_mesh.point(v2)[0] > 60.1f || m_mesh.point(v3)[0] > 60.1f)
			continue;
		if (m_mesh.point(v1)[2] < -0.1f || m_mesh.point(v2)[2] < -0.1f || m_mesh.point(v3)[2] < -0.1f)
			continue;
		if (m_mesh.point(v1)[2] > 50.1f || m_mesh.point(v2)[2] > 50.1f || m_mesh.point(v3)[2] > 50.1f)
			continue;
		int have_water_cnt = 0;
		if (have_water(v1)) have_water_cnt++;
		if (have_water(v2)) have_water_cnt++;
		if (have_water(v3)) have_water_cnt++;
		if (have_water_cnt == 3) {
		}
		else if (have_water_cnt == 2) {
			for (int i = 0; i < 2; i++) {
				if (have_water(v3)) {
					swap(e2, e3);
					swap(v2, v3);
					swap(e1, e2);
					swap(v1, v2);
				}
			}
		}
		else if (have_water_cnt == 1) {
			for (int i = 0; i < 2; i++) {
				if (!have_water(v1)) {
					swap(e2, e3);
					swap(v2, v3);
					swap(e1, e2);
					swap(v1, v2);
				}
			}
		}
		else {
			// do nothing
		}
	}
	delete[] vertex_to_idx;
	delete[] edge_to_idx;
}
void Simulator::runoneframe() {
	belowwatermodel();
	update_midvels();
	advect_filed_values();
	extrapolate_depth();
	force_boundary_depth();
	calculate_pressure();
	update_velocity();
	force_boundary_velocity();
	velocity_fast_march();
	update_depth();
}
void Simulator::clear() {
	release_index();
	release_properties();
}

void Simulator::set_initial_constants(bool m_have_tensor,float m_fric_coef,float m_gamma,float m_dt,float g) {
	m_g = MyMesh::Point(0, -1, 0).normalized() * g;
	this->m_have_tensor = m_have_tensor;
	this->m_fric_coef = m_fric_coef;
	this->m_gamma = m_gamma;
	this->m_dt = m_dt;
}
void Simulator::set_initial_constants() {
	m_g = MyMesh::Point(0, -1, 0).normalized() * 9.80f;

	// 摩擦力系数
	m_have_tensor = true;
	m_fric_coef = 1.3f;

	// 表面张力的系数
	m_gamma = 1.000f;
	// gamma is surface tension coefficient divided by density, theoretical value for 25 degree(C) water is 7.2*10^-5 (N/m)/(kg/m^3)
	// note: surface tension force not negligible under length scale less than about 4mm, so gamma value should be set with careful considering mesh edge length. Try using mm or cm instead of m
	m_water_boundary_theta = (float)M_PI / 180 * 30.0f;
	m_water_boundary_tension_multiplier = 1.0f;
	m_max_p_bs = 10.0f;

	// 模拟帧设置
	m_dt = 0.033f;

	switch (m_situation) {
	case 7:
	case 10:
		m_dt = 0.01f;
		break;
	case 9:
		m_dt = 0.02f;
		break;
	default:
		break;
	}
}

void Simulator::generate_origin(int MRES_X, int MRES_Z, std::vector<float> bt,float grid_size) {
	bottom.resize(MRES_X*MRES_Z);
	height.resize(MRES_X*MRES_Z);
	x_cells = MRES_X;
	z_cells = MRES_Z;
	MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES_X];
	for (int i = 0; i < MRES_X; i++)
		vhandle[i] = new MyMesh::VertexHandle[MRES_Z];
	std::vector<MyMesh::VertexHandle>  face_vhandles;
	for (int i = 0; i < MRES_X; i++)
		for (int j = 0; j < MRES_Z; j++) {
			float b = bt[i*MRES_X+j];
			bottom.push_back(b);
			vhandle[i][j] = m_origin.add_vertex(MyMesh::Point(i * grid_size, b, j * grid_size));
		}
	for (int i = 0; i < MRES_X - 1; i++)
		for (int j = 0; j < MRES_Z - 1; j++) {
			face_vhandles.clear();
			face_vhandles.push_back(vhandle[i][j]);
			face_vhandles.push_back(vhandle[i + 1][j + 1]);
			face_vhandles.push_back(vhandle[i + 1][j]);
			m_origin.add_face(face_vhandles);
			face_vhandles.clear();
			face_vhandles.push_back(vhandle[i][j]);
			face_vhandles.push_back(vhandle[i][j + 1]);
			face_vhandles.push_back(vhandle[i + 1][j + 1]);
			m_origin.add_face(face_vhandles);
		}
	for (int i = 0; i < MRES_X; i++)
		delete[] vhandle[i];
	delete[] vhandle;
}
void Simulator::generate_origin() {
	m_origin.clear();
	switch (m_situation) {
	case 1: case 2: case 3: case 8:
	{
		enum { MRES = 101 };
		float const grid_size = 0.5f;
		MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES];
		for (int i = 0; i < MRES; i++)
			vhandle[i] = new MyMesh::VertexHandle[MRES];
		for (int i = 0; i < MRES; i++)
			for (int j = 0; j < MRES; j++)
				vhandle[i][j] = m_origin.add_vertex(MyMesh::Point(i * grid_size, 0, j * grid_size));
		std::vector<MyMesh::VertexHandle>  face_vhandles;
		for (int i = 0; i < MRES - 1; i++)
			for (int j = 0; j < MRES - 1; j++) {
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j]);
				m_origin.add_face(face_vhandles);
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				m_origin.add_face(face_vhandles);
			}
		for (int i = 0; i < MRES; i++)
			delete[] vhandle[i];
		delete[] vhandle;
	}
	break;
	case 4:
	{
		char const input_filename[] = "obj_models/Arma.obj";
		OpenMesh::IO::read_mesh(m_origin, input_filename);
		// jacobi_laplace_smooth(m_origin, 50);
		BoundRecorder<float> xbound, ybound, zbound;
		for (auto v_it = m_origin.vertices_begin(); v_it != m_origin.vertices_end(); ++v_it) {
			MyMesh::Point p(m_origin.point(*v_it));
			xbound.insert(p[0]);
			ybound.insert(p[1]);
			zbound.insert(p[2]);
		}
		auto max = [](double a, double b) { return a > b ? a : b; };
		float size = max(max(xbound.get_max() - xbound.get_min(), ybound.get_max() - ybound.get_min()), zbound.get_max() - zbound.get_min());
		float ratio = 50.0f / size;
		float xoff = -xbound.get_min();
		float yoff = -ybound.get_min() - 25.0f / ratio;
		float zoff = -zbound.get_min();
		for (auto v_it = m_origin.vertices_begin(); v_it != m_origin.vertices_end(); ++v_it) {
			MyMesh::Point p(m_origin.point(*v_it));
			MyMesh::Point np((p[0] + xoff) * ratio, (p[1] + yoff) * ratio, (p[2] + zoff) * ratio);
			m_origin.point(*v_it) = np;
		}
	}
	break;
	case 5: case 6:
	{
		enum { MRES = 101 };
		float const grid_size = 0.5f;
		bottom.resize(MRES*MRES);
		height.resize(MRES*MRES);
		x_cells = 101;
		z_cells = 101;
		MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES];
		for (int i = 0; i < MRES; i++)
			vhandle[i] = new MyMesh::VertexHandle[MRES];
		for (int i = 0; i < MRES; i++)
			for (int j = 0; j < MRES; j++) {
				float r = (MyMesh::Point(i * grid_size, 0, j * grid_size) - MyMesh::Point(25.0f, 0, 25.0f)).norm();
				bottom.push_back(r*r / 500);
				vhandle[i][j] = m_origin.add_vertex(MyMesh::Point(i * grid_size, r * r / 500, j * grid_size));
			}
		std::vector<MyMesh::VertexHandle>  face_vhandles;
		for (int i = 0; i < MRES - 1; i++)
			for (int j = 0; j < MRES - 1; j++) {
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j]);
				m_origin.add_face(face_vhandles);
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				m_origin.add_face(face_vhandles);
			}
		for (int i = 0; i < MRES; i++)
			delete[] vhandle[i];
		delete[] vhandle;
	}
	break;
	case 7:
	{
		enum { MRES = 101 };
		bottom.resize(MRES*MRES);
		height.resize(MRES*MRES);
		x_cells = 101;
		z_cells = 101;
		float const grid_size = 0.5f;
		MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES];
		for (int i = 0; i < MRES; i++)
			vhandle[i] = new MyMesh::VertexHandle[MRES];
		std::vector<MyMesh::VertexHandle>  face_vhandles;
		for (int i = 0; i < MRES; i++)
			for (int j = 0; j < MRES; j++) {
				float b = 0.1f * (sin(2.0f * 0.5f * (float)i + 2.0f * 0.5f * (float)j) + 1.01f);
				bottom[i*MRES + j] = b;
				vhandle[i][j] = m_origin.add_vertex(MyMesh::Point(i * grid_size, b, j * grid_size));
			}
		for (int i = 0; i < MRES - 1; i++)
			for (int j = 0; j < MRES - 1; j++) {
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j]);
				m_origin.add_face(face_vhandles);
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				m_origin.add_face(face_vhandles);
			}
		for (int i = 0; i < MRES; i++)
			delete[] vhandle[i];
		delete[] vhandle;
	}
	break;
	case 9:
	{
#if 1
		char const input_filename[] = "obj_models/trunk.obj";
		m_origin.request_vertex_texcoords2D();
		OpenMesh::IO::Options opt(OpenMesh::IO::Options::VertexTexCoord);
		OpenMesh::IO::read_mesh(m_origin, input_filename, opt);
		//decimater(m_origin, 5e-2);
		for (auto v_it = m_origin.vertices_begin(); v_it != m_origin.vertices_end(); ++v_it) {
			MyMesh::Point p(m_origin.point(*v_it));
			if ((p[0] + p[1] / 2) > 60.0f || (p[0] - p[1] / 2) > -15.0f) {
				m_origin.delete_vertex(*v_it);
			}
			else {
				m_origin.point(*v_it) = MyMesh::Point(p[1], -p[0] - 23.0f, p[2] + 80) * 0.4f;
			}
		}
		m_origin.garbage_collection();
		OpenMesh::IO::write_mesh(m_origin, "obj_models/trunk_part.obj", opt);

#else
		char const input_filename[] = "obj_models/trunk_part.obj";
		m_origin.request_vertex_texcoords2D();
		OpenMesh::IO::Options opt(OpenMesh::IO::Options::VertexTexCoord);
		OpenMesh::IO::read_mesh(m_origin, input_filename, opt);
#endif
	}
	break;
	case 10:
	{
		enum { MRES_X = 131, MRES_Z = 121 };
		bottom.resize(MRES_X*MRES_Z);
		height.resize(MRES_X*MRES_Z);
		x_cells = 131;
		z_cells = 121;
		float const grid_size = 0.5f;
		MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES_X];
		for (int i = 0; i < MRES_X; i++)
			vhandle[i] = new MyMesh::VertexHandle[MRES_Z];
		std::vector<MyMesh::VertexHandle>  face_vhandles;
		for (int i = 0; i < MRES_X; i++)
			for (int j = 0; j < MRES_Z; j++) {
				float b = 0.05f * (cos(2.0f * 0.5f * (float)i + 2.0f * 0.5f * (float)j) + 1.01f);
				bottom.push_back(b);
				vhandle[i][j] = m_origin.add_vertex(MyMesh::Point(i * grid_size, b, (j - 10) * grid_size));
			}
		for (int i = 0; i < MRES_X - 1; i++)
			for (int j = 0; j < MRES_Z - 1; j++) {
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j]);
				m_origin.add_face(face_vhandles);
				face_vhandles.clear();
				face_vhandles.push_back(vhandle[i][j]);
				face_vhandles.push_back(vhandle[i][j + 1]);
				face_vhandles.push_back(vhandle[i + 1][j + 1]);
				m_origin.add_face(face_vhandles);
			}
		for (int i = 0; i < MRES_X; i++)
			delete[] vhandle[i];
		delete[] vhandle;
	}
	break;
	default:
		break;
	}
}
void Simulator::generate_mesh(int situation,int times) {
	m_mesh.clear();
	switch (situation) {
	case 1:
		m_mesh = m_origin;
		break;
	case 2:
		m_mesh = m_origin; jacobi_laplace_smooth_and_expand(m_mesh, times);
		break;
	case 3:
		m_mesh = m_origin;
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); v_it++) {
			m_mesh.point(*v_it)[1] = 0;
		}
		break;
	default:
		break;
	}
}
void Simulator::generate_mesh() {

	// 令m_mesh为m_origin的平滑网格
	m_mesh.clear();
	switch (m_situation) {
	case 1: case 2: case 3: case 5: case 6: case 8:
		m_mesh = m_origin;
		break;
	case 4:
		// m_mesh = potential_field_mesh(m_origin);
		m_mesh = m_origin; jacobi_laplace_smooth_and_expand(m_mesh, 50);
		break;
	case 7: case 10:
		m_mesh = m_origin;
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); v_it++) {
			m_mesh.point(*v_it)[1] = 0;
		}
		break;
	case 9:
		m_mesh = m_origin;
		//decimater(m_mesh, 1e-2);
		jacobi_laplace_smooth_and_expand(m_mesh, 200);
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point p(m_mesh.point(*v_it));
			if (((-23.0f - p[1] / 0.4f) + p[0] / 0.4f / 2) > 50.0f || ((-23.0f - p[1] / 0.4f) - p[0] / 0.4f / 2) > -25.0f) {
				m_mesh.delete_vertex(*v_it);
			}
		}
		m_mesh.garbage_collection();
	default:
		break;
	}
}

void Simulator::add_properties() {
	m_mesh.request_face_normals();
	m_mesh.request_vertex_normals();
	m_mesh.update_normals();
	m_mesh.add_property(m_normal);
	m_mesh.add_property(m_bottom);
	m_mesh.add_property(m_depth);
	m_mesh.add_property(m_height);
	m_mesh.add_property(m_on_water_boundary);
	m_mesh.add_property(m_extrapolate_depth);
	m_mesh.add_property(m_pressure_gravity);
	m_mesh.add_property(m_pressure_surface);
	m_mesh.add_property(m_float_temp);
	m_mesh.add_property(m_velocity);
	m_mesh.add_property(m_midvel);
	m_mesh.add_property(m_vector_temp);
	m_mesh.add_property(m_origin_face);
	m_mesh.add_property(m_tensor);
	m_mesh.add_property(m_boundary);
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it)
		if (m_mesh.is_boundary(*v_it))
			m_mesh.property(m_boundary, *v_it) = new BoundaryCondition;
		else
			m_mesh.property(m_boundary, *v_it) = 0;
}

void Simulator::add_index_to_vertex(int ring) {
	size_t memory_cost = 0;
	float avg_faces_per_vertex = 0;
	WindowsTimer timer;
	timer.restart();
	m_mesh.request_face_normals();
	m_mesh.request_vertex_normals();
	m_mesh.update_face_normals();
	m_mesh.update_vertex_normals();
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		std::set<MyMesh::FaceHandle> set_fh;
		insert_vertex_nring(m_mesh, *v_it, ring, set_fh);
		m_mesh.data(*v_it).index = new IndexOnVertex(m_mesh, *v_it, set_fh);
		memory_cost += m_mesh.data(*v_it).index->memory_cost();
		avg_faces_per_vertex += (float)set_fh.size();
	}
	m_mesh.release_vertex_normals();
	m_mesh.release_face_normals();
	timer.stop();
	avg_faces_per_vertex /= (float)m_mesh.n_vertices();
	//cout << "create " << ring << "-ring index for " << timer.get() << " sec, " << ((float)memory_cost / 1024 / 1024) << " MB" << endl;
	//cout << "average: " << avg_faces_per_vertex << " faces per vertex (in " << ring << "-ring)" << endl;
}

void Simulator::match_bottom_height() {
	// Todo: 用类似光线跟踪的kd-tree方法加速
	cout << "从原网格向平滑网格映射" << endl;
	IndexOfMesh origin_index(m_origin);
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
#if 0
		// 计算2-ring平均法向
		map<MyMesh::VertexHandle, int> m;
		calc_vertex_nring(m_mesh, *v_it, 2, m);
		MyMesh::Point n(0, 0, 0);
		for (auto it = m.begin(); it != m.end(); ++it) {
			MyMesh::VertexHandle vh = it->first;
			n += m_mesh.normal(vh);
		}
		n.normalize();
		m_mesh.property(m_normal, *v_it) = n;
#else
		m_mesh.property(m_normal, *v_it) = m_mesh.normal(*v_it);
#endif
	}
	int cnt = 0;
	int n_not_matched = 0;
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		if (++cnt % 1000 == 0)
			cout << cnt << " / " << m_mesh.n_vertices() << endl;
		MyMesh::Point base = m_mesh.point(*v_it);
		MyMesh::Point normal = m_mesh.property(m_normal, *v_it); // m_mesh.normal(*v_it);
		MyMesh::FaceHandle fh;
		float dist;
		origin_index.nearest_intersection(base, normal, fh, dist);
		// Todo: 判断沿法向的交点是否正确（现在认为距离在2以内则正确，否则可能映射到了其它面上）
		if (fabs(dist) > 2)
			fh = *m_origin.faces_end();
		m_mesh.property(m_origin_face, *v_it) = fh;
		if (fh == *m_origin.faces_end()) {
			// 将从附近的点外插
			n_not_matched++;
			m_mesh.property(m_bottom, *v_it) = 999.0; // 当未能成功外插时，让错误更明显
			m_mesh.property(m_height, *v_it) = 999.0;
		}
		else {
			m_mesh.property(m_bottom, *v_it) = dist;
			m_mesh.property(m_height, *v_it) = dist;
		}
	}
	cout << n_not_matched << " 个平滑网格顶点沿法向与原网格没有交点" << endl;
	// 法向未匹配的面片的点从附近的点取bottom
	std::queue<MyMesh::VertexHandle> q;
	std::set<MyMesh::VertexHandle> set_not_matched;
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		if (m_mesh.property(m_origin_face, *v_it) == *m_origin.faces_end())
			continue;
		for (auto vv_it = m_mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
			if (m_mesh.property(m_origin_face, *vv_it) == *m_origin.faces_end() &&
				set_not_matched.find(*vv_it) == set_not_matched.end()) {
				m_mesh.property(m_bottom, *vv_it) = m_mesh.property(m_bottom, *v_it);
				m_mesh.property(m_height, *vv_it) = m_mesh.property(m_height, *v_it);
				q.push(*vv_it);
				set_not_matched.insert(*vv_it);
			}
		}
	}
	while (!q.empty()) {
		MyMesh::VertexHandle vh = q.front();
		q.pop();
		for (auto vv_it = m_mesh.vv_iter(vh); vv_it.is_valid(); ++vv_it) {
			if (m_mesh.property(m_origin_face, *vv_it) == *m_origin.faces_end() &&
				set_not_matched.find(*vv_it) == set_not_matched.end()) {
				m_mesh.property(m_bottom, *vv_it) = m_mesh.property(m_bottom, vh);
				m_mesh.property(m_height, *vv_it) = m_mesh.property(m_height, vh);
				q.push(*vv_it);
				set_not_matched.insert(*vv_it);
			}
		}
	}
	if (n_not_matched != set_not_matched.size())
		cout << "错误：有些法向不与原网格相交的点未能从附近点外插" << endl;
}

void Simulator::calculate_tensor() {
	if (!m_have_tensor)
		return;
	cout << "计算摩擦力张量" << endl;
	int cnt = 0;
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		cnt++;
		if (cnt % 1000 == 0)
			cout << cnt << " / " << m_mesh.n_vertices() << endl;
		if (m_mesh.is_boundary(*v_it) || m_mesh.property(m_origin_face, *v_it) == *m_origin.faces_end()) {
			m_mesh.property(m_tensor, *v_it) = Tensor22(0, 0, 0, 0);
			continue;
		}
		IndexOnVertex *index = m_mesh.data(*v_it).index;
		float bottom = m_mesh.property(m_bottom, *v_it);
		std::map<MyMesh::VertexHandle, MyMesh::Point> rotate_map;
		auto rotate_to_local_point = [&](MyMesh::VertexHandle vh_on_origin) {
			auto it = rotate_map.find(vh_on_origin);
			if (it != rotate_map.end())
				return it->second;
			MyMesh::Point p(m_origin.point(vh_on_origin));
			MyMesh::Point line = index->from_nature_coord(p - index->vertical_offset_point(bottom));
			MyMesh::Point to_search(line[0], line[1], 0);
			MyMesh::FaceHandle fh = index->search(m_mesh, to_search);
			while (fh == *m_mesh.faces_end()) {
				// Todo: 正确查找最近的面片
				to_search *= 0.67f;
				fh = index->search(m_mesh, to_search);
			}
			MyMesh::Point fcoord[3];
			fcoord[2] = index->from_nature_coord(m_mesh.normal(fh));
			fcoord[0] = MyMesh::Point(0, 0, 1) % fcoord[2];
			if (fcoord[0].norm() < 0.1)
				fcoord[0] = MyMesh::Point(0, 1, 0) % fcoord[2];
			fcoord[0].normalize();
			fcoord[1] = fcoord[2] % fcoord[0];
			MyMesh::Point line_in_f_coord(line | fcoord[0], line | fcoord[1], line | fcoord[2]);
			static CoordSystem const standard_system(MyMesh::Point(1, 0, 0), MyMesh::Point(0, 1, 0), MyMesh::Point(0, 0, 1));
			MyMesh::Point line_in_index_coord = IndexOnVertex::coord_conv(fcoord, standard_system.data(), line_in_f_coord);
			//cout << line << "\t| " << line_in_index_coord << endl;
			rotate_map.insert(it, std::make_pair(vh_on_origin, line_in_index_coord));
			return line_in_index_coord;
		};
		auto in_control_volumn = [&](MyMesh::Point p) {
			return false;
		};
		auto have_crossed_volumn = [&](MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c) {
			auto have_crossed_volumn = [](MyMesh::Point const tri0[3], MyMesh::Point const tri1[3]) {
				auto line_cross = [](MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c, MyMesh::Point const &d) {
					return ((c - a) % (b - a))[2] * ((b - a) % (d - a))[2] > 0 && ((a - c) % (d - c))[2] * ((d - c) % (b - c))[2] > 0;
				};
				auto in_triangle = [](MyMesh::Point const &p, MyMesh::Point const tri[3]) {
					return IndexOnVertex::on_face_2d(p, tri[0], tri[1], tri[2]);
				};
				int crossed_line = 0;
				for (int i = 0; i < 3; i++) {
					if (in_triangle(tri0[i], tri1) || in_triangle(tri1[i], tri0))
						return true;
					for (int j = 0; j < 3; j++)
						if (line_cross(tri0[i], tri0[(i + 1) % 3], tri1[j], tri1[(j + 1) % 3]))
							crossed_line++;
				}
				return crossed_line > 3;
			};
			MyMesh::Point tri0[3] = { a, b, c };
			for (int i = 0; i < 3; i++)
				tri0[i][2] = 0;
			std::vector<MyMesh::Point> vec_points;
			for (auto vv_it = m_mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
				vec_points.push_back(index->plane_map(m_mesh.point(*vv_it)));
			}
			for (size_t i = 0; i < vec_points.size(); i++) {
				size_t j = i + 1;
				if (j == vec_points.size())
					j = 0;
				static MyMesh::Point const o(0, 0, 0);
				MyMesh::Point a(vec_points[i]);
				MyMesh::Point b(vec_points[j]);
				MyMesh::Point tri1[3] = { o, (o + a) / 2, (o + a + b) / 3 };
				if (have_crossed_volumn(tri0, tri1))
					return true;
				MyMesh::Point tri2[3] = { o, (o + a + b) / 3, (o + b) / 2 };
				if (have_crossed_volumn(tri0, tri2))
					return true;
			}
			return false;
		};
		auto crossed_control_volumn = [&](MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c) {
			if (have_crossed_volumn(a, b, c))
				return 1.0f;
			else
				return (float)0;
		};

		MyMesh local_mesh;
		std::map<MyMesh::VertexHandle, MyMesh::VertexHandle> local_vertex_map; // m_origin.vertex -> local_mesh.vertex
		std::map<MyMesh::FaceHandle, MyMesh::FaceHandle> local_face_map; // m_origin.face -> local_mesh.face
		std::map<MyMesh::FaceHandle, bool> origin_face_in_volumn; // m_origin.face;
		std::set<MyMesh::VertexHandle> traversed_vertex; // m_origin.vertex
		std::queue<MyMesh::VertexHandle> candidate_vertex; // m_origin.vertex
		std::vector<MyMesh::VertexHandle> face_vhandles;
		auto fv_it = m_origin.fv_iter(m_mesh.property(m_origin_face, *v_it));
		for (int i = 0; i < 3; i++) {
			candidate_vertex.push(*fv_it);
			traversed_vertex.insert(*fv_it);
			++fv_it;
		}
		while (!candidate_vertex.empty()) {
			MyMesh::VertexHandle vh = candidate_vertex.front();
			candidate_vertex.pop();
			for (auto vf_it = m_origin.vf_iter(vh); vf_it.is_valid(); ++vf_it) {
				if (origin_face_in_volumn.find(*vf_it) == origin_face_in_volumn.end()) {
					auto fv_it = m_origin.fv_iter(*vf_it);
					MyMesh::Point a(rotate_to_local_point(*fv_it++));
					MyMesh::Point b(rotate_to_local_point(*fv_it++));
					MyMesh::Point c(rotate_to_local_point(*fv_it++));
					origin_face_in_volumn[*vf_it] = have_crossed_volumn(a, b, c);
					if (origin_face_in_volumn[*vf_it] == true) {
						for (auto fv_it = m_origin.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it)
							if (traversed_vertex.find(*fv_it) == traversed_vertex.end()) {
								candidate_vertex.push(*fv_it);
								traversed_vertex.insert(*fv_it);
							}
					}
				}
			}
		}
		for (auto v_it = traversed_vertex.begin(); v_it != traversed_vertex.end(); ++v_it) {
			if (local_vertex_map.find(*v_it) == local_vertex_map.end()) {
				MyMesh::Point p_local = rotate_to_local_point(*v_it);
				local_vertex_map[*v_it] = local_mesh.add_vertex(p_local);
			}
			for (auto vv_it = m_origin.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
				if (local_vertex_map.find(*vv_it) == local_vertex_map.end()) {
					MyMesh::Point p_local = rotate_to_local_point(*vv_it);
					local_vertex_map[*vv_it] = local_mesh.add_vertex(p_local);
				}
			}
			for (auto vf_it = m_origin.vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
				if (local_face_map.find(*vf_it) == local_face_map.end()) {
					auto fv_it = m_origin.fv_ccwiter(*vf_it);
					face_vhandles.clear();
					for (int i = 0; i < 3; i++) {
						MyMesh::VertexHandle vh = *fv_it;
						MyMesh::VertexHandle vh_local = local_vertex_map[vh];
						face_vhandles.push_back(vh_local);
						++fv_it;
					}
					local_face_map[*vf_it] = local_mesh.add_face(face_vhandles);
				}
			}
		}
		OpenMesh::FPropHandleT<float> prop_cross;
		OpenMesh::FPropHandleT<CoordSystem> prop_axis;
		OpenMesh::FPropHandleT<Tensor22> prop_tensor;
		OpenMesh::VPropHandleT<MyMesh::Point> prop_normali;
		local_mesh.add_property(prop_cross);
		local_mesh.add_property(prop_axis);
		local_mesh.add_property(prop_tensor);
		local_mesh.add_property(prop_normali);
		local_mesh.request_face_normals();
		local_mesh.request_vertex_normals();
		local_mesh.update_normals();
		std::queue<MyMesh::HalfedgeHandle> que_hh;
		std::vector<MyMesh::Point> ring_points;
		// 计算顶点法向时，以面片占的角度为权重
		for (auto v_it = local_mesh.vertices_begin(); v_it != local_mesh.vertices_end(); ++v_it) {
			ring_points.clear();
			for (auto voh_it = local_mesh.cvoh_ccwiter(*v_it); voh_it.is_valid(); ++voh_it)
				que_hh.push(*voh_it);
			if (local_mesh.is_boundary(*v_it)) {
				while (!local_mesh.is_boundary(que_hh.front())) {
					que_hh.push(que_hh.front());
					que_hh.pop();
				}
			}
			while (!que_hh.empty()) {
				ring_points.push_back(local_mesh.point(local_mesh.to_vertex_handle(que_hh.front())));
				que_hh.pop();
			}
			MyMesh::Point normali = computeNormali(ring_points, local_mesh.point(*v_it), local_mesh.is_boundary(*v_it));
			local_mesh.property(prop_normali, *v_it) = normali;
		}
		for (auto f_it = local_mesh.faces_begin(); f_it != local_mesh.faces_end(); ++f_it) {
			MyMesh::FaceHandle fh_local = *f_it;
			// 求解fh_local处的摩擦力张量
			bool have_boundary_vertex = false;
			for (auto fv_it = local_mesh.fv_iter(fh_local); fv_it.is_valid(); ++fv_it)
				if (local_mesh.is_boundary(*fv_it))
					have_boundary_vertex = true;
			if (have_boundary_vertex) {
				local_mesh.property(prop_cross, fh_local) = 0;
			}
			else {
				{
					auto fv_it = local_mesh.fv_iter(*f_it);
					MyMesh::Point a(local_mesh.point(*fv_it++));
					MyMesh::Point b(local_mesh.point(*fv_it++));
					MyMesh::Point c(local_mesh.point(*fv_it++));
					local_mesh.property(prop_cross, fh_local) = crossed_control_volumn(a, b, c);
				}

				// fh_local处的局部坐标系ex, ey, n
				MyMesh::Point n_local(local_mesh.normal(fh_local));
				MyMesh::Point ex_local = MyMesh::Point(1, 0, 0) % n_local;
				if (ex_local.sqrnorm() < 0.3f)
					ex_local = MyMesh::Point(0, 1, 0) % n_local;
				ex_local.normalize();
				MyMesh::Point ey_local = n_local % ex_local;
				static auto const to_local_coord = [&](MyMesh::Point const &v) {
					return MyMesh::Point(v | ex_local, v | ey_local, v | n_local);
				};
				static auto const _gradient = [](MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c, float fa, float fb, float fc) {
					auto dx = (b[1] - c[1]) * fa + (c[1] - a[1]) * fb + (a[1] - b[1]) * fc;
					auto dy = (c[0] - b[0]) * fa + (a[0] - c[0]) * fb + (b[0] - a[0]) * fc;
					auto area = ((b - a) % (c - a))[2] / 2;
					return MyMesh::Point(dx / 2, dy / 2, 0) / area;
				};
				auto fv_it = local_mesh.fv_iter(*f_it);
				MyMesh::Point a(local_mesh.point(*fv_it));
				//MyMesh::Point na(local_mesh.normal(*fv_it));
				MyMesh::Point na(local_mesh.property(prop_normali, *fv_it));
				++fv_it;
				MyMesh::Point b(local_mesh.point(*fv_it));
				//MyMesh::Point nb(local_mesh.normal(*fv_it));
				MyMesh::Point nb(local_mesh.property(prop_normali, *fv_it));
				++fv_it;
				MyMesh::Point c(local_mesh.point(*fv_it));
				//MyMesh::Point nc(local_mesh.normal(*fv_it));
				MyMesh::Point nc(local_mesh.property(prop_normali, *fv_it));
				//cout << (index->to_nature_coord((a + b + c) / 3) + m_mesh.point(*v_it)) << endl;
				b = to_local_coord(b - a);
				c = to_local_coord(c - a);
				a = MyMesh::Point(0, 0, 0);
				na = to_local_coord(na);
				nb = to_local_coord(nb);
				nc = to_local_coord(nc);
				MyMesh::Point grad_x = _gradient(a, b, c, na[0], nb[0], nc[0]);
				MyMesh::Point grad_y = _gradient(a, b, c, na[1], nb[1], nc[1]);

				Tensor22 tensor_local(-grad_x[0], -(grad_x[1] + grad_y[0]) / 2, -(grad_x[1] + grad_y[0]) / 2, -grad_y[1]);
				local_mesh.property(prop_axis, fh_local) = CoordSystem(ex_local, ey_local, n_local);
				local_mesh.property(prop_tensor, fh_local) = tensor_local;
			}
		}
		std::vector<MyMesh::FaceHandle> effective_faces;
		for (auto f_it = local_mesh.faces_begin(); f_it != local_mesh.faces_end(); ++f_it)
			if (local_mesh.property(prop_cross, *f_it) != 0)
				effective_faces.push_back(*f_it);
		static CoordSystem const standard_system(MyMesh::Point(1, 0, 0), MyMesh::Point(0, 1, 0), MyMesh::Point(0, 0, 1));
		float max_tao;
		float result_theta;
		bool is_first = true;
		for (float theta = 0; theta < 3.1415926f; theta += 0.05f) {
			float sum_mult = 0;
			float sum_area = 0;
			for (auto f_it = effective_faces.begin(); f_it != effective_faces.end(); ++f_it) {
				float area = local_mesh.property(prop_cross, *f_it);
				MyMesh::Point x(cos(theta), sin(theta), 0);
				x = IndexOnVertex::coord_conv(standard_system.data(), local_mesh.property(prop_axis, *f_it).data(), x);
				Tensor22 const &t(local_mesh.property(prop_tensor, *f_it));
				float mult = area * fabs(t[0] * x[0] * x[0] + t[1] * x[0] * x[1] + t[2] * x[0] * x[1] + t[3] * x[1] * x[1]);
				sum_mult += mult;
				sum_area += area;
			}
			if (sum_area == 0)
				continue;
			float tao = sum_mult / sum_area;
			if (is_first || tao > max_tao) {
				max_tao = tao;
				result_theta = theta;
				is_first = false;
			}
		}
		if (is_first)
			cout << "错误：计算摩擦力张量时，没有与control volumn相交的面片，或都在原mesh的边缘上，无法完成计算" << endl;
		float min_tao;
		{
			float theta = result_theta + 3.1415926f / 2;
			float sum_mult = 0;
			float sum_area = 0;
			for (auto f_it = local_mesh.faces_begin(); f_it != local_mesh.faces_end(); ++f_it) {
				float area = local_mesh.property(prop_cross, *f_it);
				if (area == 0)
					continue;
				MyMesh::Point x(cos(theta), sin(theta), 0);
				x = IndexOnVertex::coord_conv(standard_system.data(), local_mesh.property(prop_axis, *f_it).data(), x);
				Tensor22 const &t(local_mesh.property(prop_tensor, *f_it));
				float mult = area * fabs(t[0] * x[0] * x[0] + t[1] * x[0] * x[1] + t[2] * x[0] * x[1] + t[3] * x[1] * x[1]);
				sum_mult += mult;
				sum_area += area;
			}
			float tao = sum_mult / sum_area;
			min_tao = tao;
		}
		float x = cos(result_theta);
		float y = sin(result_theta);
		// 设置主曲率的上限，使得update_velocity的时候摩擦力不至于使速度反向
		float max_tao_bound = 1.0f / (m_dt * m_fric_coef) * 0.5f;
		if (max_tao > max_tao_bound) {
			min_tao = min_tao / max_tao * max_tao_bound;
			max_tao = max_tao_bound;
		}
		Tensor22 tensor(max_tao * x * x + min_tao * y * y, (max_tao - min_tao) * x * y, (max_tao - min_tao) * x * y, max_tao * y * y + min_tao * x * x);
		m_mesh.property(m_tensor, *v_it) = tensor;
		//if (v_it->idx() % 285 == 0)
		//	cout << v_it->idx() << '\t' << result_theta << "\t" << max_tao << "\t" << min_tao << "\t" << index->to_nature_coord(MyMesh::Point(cos(result_theta), sin(result_theta), 0)) << "\t" << m_mesh.point(*v_it) << endl;
	}
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		if (m_mesh.is_boundary(*v_it)) {
			// Todo: 规定边界情况
			MyMesh::VertexHandle vh = m_mesh.data(*v_it).index->nearest_vertex(m_mesh, MyMesh::Point(0, 0, 0), [&](MyMesh::VertexHandle vh) { return !m_mesh.is_boundary(vh); });
			if (vh == *m_mesh.vertices_end())
				cout << "错误：计算摩擦力张量时，边界点使用临近的非边界点的张量，但找不到临近的非边界点" << endl;
			m_mesh.property(m_tensor, *v_it) = m_mesh.property(m_tensor, vh);
		}
	}
}

void Simulator::set_initial_conditions(std::vector<float> hi, std::vector<std::vector<float>> v) {
	int num = 0;
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		m_mesh.property(m_depth, *v_it) = hi[num] - m_mesh.property(m_bottom, *v_it);
		m_mesh.property(m_velocity, *v_it) = MyMesh::Point(v[num][0], v[num][1], v[num][2]);
		height[x_cells*m_mesh.point(*v_it)[0] + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
		if (m_mesh.is_boundary(*v_it)) {
			m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
			m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
		}
		if (m_mesh.property(m_depth, *v_it) > m_depth_threshold) {
			m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
			m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
		}
		num++;
	}
}
void Simulator::set_initial_conditions(std::vector<float> hi, std::vector<float> v) {
	int num = 0;
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		m_mesh.property(m_depth, *v_it) = hi[num] - m_mesh.property(m_bottom, *v_it);
		m_mesh.property(m_velocity, *v_it) = MyMesh::Point(v[0], v[1], v[2]);
		height[x_cells*m_mesh.point(*v_it)[0] + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
		if (m_mesh.is_boundary(*v_it)) {
			m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
			m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
		}
		if (m_mesh.property(m_depth, *v_it) > m_depth_threshold) {
			m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
			m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
		}
		num++;
	}
}
void Simulator::set_initial_conditions() {
	switch (m_situation) {
	case 1:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point midpos(25.0f, 0, 25.0f);
			MyMesh::Point p(m_mesh.point(*v_it));
			float dis2 = (p - midpos).sqrnorm();
			m_mesh.property(m_depth, *v_it) = (p - midpos).norm() < 9.01f ? 5.0f + 5.0f * exp(-dis2) : 5.0f;
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
		}
		break;
	case 2:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point midpos(25.0f, 0, 25.0f);
			MyMesh::Point p(m_mesh.point(*v_it));
			m_mesh.property(m_depth, *v_it) = (p - midpos).norm() < 1.01f ? 10.0f : 5.0f;
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
		}
		break;
	case 3:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::Point p(m_mesh.point(*v_it));
			m_mesh.property(m_depth, *v_it) = p[0] < 0.1 ? 0.5f : 0;
			m_mesh.property(m_velocity, *v_it) = index->from_nature_coord(MyMesh::Point(7, 0, 0));
			if (m_mesh.is_boundary(*v_it)) {
				if (p[0] < 0.1)
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
				else
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
		}
		break;
	case 4:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point p(m_mesh.point(*v_it));
			m_mesh.property(m_depth, *v_it) = p[1] > 15 && p[2] > 10 ? 0.1f : 0;
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
		}
		break;
	case 5:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point midpos(25.0f, 0, 25.0f);
			MyMesh::Point p(m_mesh.point(*v_it));
			float dis2 = (p - midpos).sqrnorm();
			m_mesh.property(m_depth, *v_it) = (p - midpos).norm() < 9.01f ? 4.0f + 2.0f * exp(-dis2) : 4.0f;
			height.push_back(m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it));
			height[x_cells*m_mesh.point(*v_it)[0] + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
		}
		break;
	case 6:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point midpos(25.0f, 0, 25.0f);
			MyMesh::Point p(m_mesh.point(*v_it));
			float dis2 = (p - midpos).sqrnorm();
			float baseh = (4.5f - p[1]) / m_mesh.normal(*v_it)[1];
			m_mesh.property(m_depth, *v_it) = (p - midpos).norm() < 9.01f ? baseh + 2.0f * exp(-dis2) : baseh;
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
			height[x_cells*m_mesh.point(*v_it)[0] + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
		}
		break;
	case 7:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::Point p(m_mesh.point(*v_it));
			m_mesh.property(m_depth, *v_it) = (p[0] < 0.1f && p[2] > 25.0f) ? 1.0f - m_mesh.property(m_bottom, *v_it) : 0;
			//m_mesh.property(m_depth, *v_it) = 0;
			m_mesh.property(m_velocity, *v_it) = index->from_nature_coord(MyMesh::Point(7, 0, 0));
			if (m_mesh.property(m_depth, *v_it) > 0) {
				height[m_mesh.point(*v_it)[0] * z_cells + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
			}
			else {
				height[m_mesh.point(*v_it)[0] * z_cells + m_mesh.point(*v_it)[2]] = 0;
			}
			//height.push_back(m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it));
			/*if (m_mesh.is_boundary(*v_it)) {
				if (m_mesh.property(m_depth, *v_it) > m_depth_threshold) {
					m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
				}
				else {
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
				}
			}*/
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
			if (m_mesh.property(m_depth, *v_it) > m_depth_threshold) {
				m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
			}
		}
		break;
	case 8:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point midpos(25.0f, 0, 25.0f);
			MyMesh::Point p(m_mesh.point(*v_it));
			m_mesh.property(m_depth, *v_it) = (p - midpos).norm() < 1.01f ? 0.1f : 0.0f;
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
		}
		break;
	case 9:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point p(m_mesh.point(*v_it));
			m_mesh.property(m_depth, *v_it) = (p[0] < 10.0f && p[2] < 44.0f && p[2] > 20.0f) ? 0.9f : 0.0f;
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_NOACTION, this, *v_it);
			}
		}
		break;
	case 10:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::Point p(m_mesh.point(*v_it));
			float b = m_mesh.property(m_bottom, *v_it);
			float h = min(0.5f, min(p[2] - 20, 40 - p[2]) * 0.4f);
			h = min(h, h - p[0] * 0.25f);
			m_mesh.property(m_depth, *v_it) = h > b ? h - b : 0;
			height[x_cells*m_mesh.point(*v_it)[0] + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
			m_mesh.property(m_velocity, *v_it) = index->from_nature_coord(MyMesh::Point(7, 0, 0));
			if (m_mesh.is_boundary(*v_it)) {
				if (m_mesh.property(m_depth, *v_it) > m_depth_threshold) {
					m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
				}
				else {
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
				}
			}
		}
		break;
	default:
		break;
	}
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		m_mesh.property(m_height, *v_it) = m_mesh.property(m_bottom, *v_it) + m_mesh.property(m_depth, *v_it);
	}
	extrapolate_depth();
}

void Simulator::update_midvels() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		MyMesh::Point vel = m_mesh.property(m_velocity, *v_it);
		MyMesh::Point midvel;
		if (vel.sqrnorm() == 0) {
			midvel = MyMesh::Point(0, 0, 0);
		}
		else {
			MyMesh::Point p(vel * (m_dt / -2));
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::FaceHandle fh = index->search(m_mesh, p);
			if (fh == *m_mesh.faces_end()) { // 在边界外
				midvel = vel;
			}
			else {
				midvel = point_interpolation(m_mesh, m_velocity, p, *v_it, fh);
			}
		}
		m_mesh.property(m_midvel, *v_it) = midvel;
	}
}

void Simulator::advect_filed_values() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		MyMesh::Point p(m_mesh.property(m_midvel, *v_it) * -m_dt);
		IndexOnVertex *index = m_mesh.data(*v_it).index;
		MyMesh::FaceHandle fh = index->search(m_mesh, p);

		float depth_new;
		MyMesh::Point velocity_new;
		if (fh == *m_mesh.faces_end()) {
			// Todo: 正确处理边界情况
			MyMesh::VertexHandle vh = index->nearest_vertex(m_mesh, p, [](MyMesh::VertexHandle) { return true; });
			if (vh == *m_mesh.vertices_end()) {
				cout << "ERROR: impossible case in Simulator::advect_filed_values" << endl;
			}
			else {
				depth_new = m_mesh.property(m_depth, vh);
				velocity_new = m_mesh.property(m_velocity, vh);
				if (m_mesh.is_boundary(*v_it)) {
					auto bound = m_mesh.property(m_boundary, *v_it);
					bound->apply_depth(depth_new);
					bound->apply_velocity(velocity_new);
				}
				velocity_new = IndexOnVertex::index_conv(m_mesh.data(vh).index, index, velocity_new);
			}
		}
		else {
			depth_new = point_interpolation(m_mesh, m_extrapolate_depth, p, *v_it, fh);
			if (depth_new <= m_depth_threshold)
				depth_new = 0;
			velocity_new = point_interpolation(m_mesh, m_velocity, p, *v_it, fh);
		}
		m_mesh.property(m_float_temp, *v_it) = depth_new;
		m_mesh.property(m_vector_temp, *v_it) = velocity_new;
	}
	swap(m_depth, m_float_temp);
	swap(m_velocity, m_vector_temp);
}

void Simulator::extrapolate_depth() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		m_mesh.property(m_on_water_boundary, *v_it) = false;
		float d = m_mesh.property(m_depth, *v_it);
		m_mesh.property(m_extrapolate_depth, *v_it) = d;
		if (m_mesh.property(m_depth, *v_it) <= m_depth_threshold) {
			bool close_to_water = false;
			for (auto vv_it = m_mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
				if (m_mesh.property(m_depth, *vv_it) > m_depth_threshold) {
					close_to_water = true;
					break;
				}
			}
			m_mesh.property(m_on_water_boundary, *v_it) = close_to_water;
			if (close_to_water) {
				int cnt = 0;
				float ex_depth = 0;
				for (auto voh_it = m_mesh.voh_iter(*v_it); voh_it.is_valid(); ++voh_it) {
					if (m_mesh.is_boundary(*voh_it))
						continue; // 不存在此三角形
					MyMesh::HalfedgeHandle hh = m_mesh.next_halfedge_handle(*voh_it);
					if (m_mesh.is_boundary(hh))
						cout << "错误：非法的拓扑结构" << endl;
					if (m_mesh.property(m_depth, m_mesh.from_vertex_handle(hh)) <= m_depth_threshold ||
						m_mesh.property(m_depth, m_mesh.to_vertex_handle(hh)) <= m_depth_threshold)
						continue; // 不是有水的面
					hh = m_mesh.opposite_halfedge_handle(hh);
					if (m_mesh.is_boundary(hh))
						continue; // 此边不存在相对的面
					MyMesh::FaceHandle fh = m_mesh.face_handle(hh);
					float this_ex_depth = point_interpolation(m_mesh, m_depth, MyMesh::Point(0, 0, 0), *v_it, fh);
					cnt++;
					ex_depth += this_ex_depth;
				}
				if (cnt == 0) {
					for (auto vv_it = m_mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
						if (m_mesh.property(m_depth, *vv_it) <= m_depth_threshold)
							continue;
						for (auto vf_it = m_mesh.vf_iter(*vv_it); vf_it.is_valid(); ++vf_it) {
							bool face_have_water = true;
							for (auto fv_it = m_mesh.fv_iter(*vf_it); fv_it.is_valid(); ++fv_it) {
								if (m_mesh.property(m_depth, *fv_it) <= m_depth_threshold) {
									face_have_water = false;
									break;
								}
							}
							if (face_have_water) {
								float this_ex_depth = point_interpolation(m_mesh, m_depth, MyMesh::Point(0, 0, 0), *v_it, *vf_it);
								cnt++;
								ex_depth += this_ex_depth;
							}
						}
					}
				}
				if (cnt == 0) {
					m_mesh.property(m_extrapolate_depth, *v_it) = 0;
				}
				else {
					float extrapolation = ex_depth / cnt;
					if (extrapolation < 0)
						m_mesh.property(m_extrapolate_depth, *v_it) = extrapolation;
					else
						m_mesh.property(m_extrapolate_depth, *v_it) = 0;
				}
			}
			else {
				// 此点无水且周围也无水
				// 设置一个负值，防止插值时因浮点数误差而出错
				m_mesh.property(m_extrapolate_depth, *v_it) = -1.0f;
			}
		}
	}
}

void Simulator::force_boundary_depth() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it)
		if (m_mesh.is_boundary(*v_it)) {
			BoundaryCondition *bc = m_mesh.property(m_boundary, *v_it);
			bc->apply_depth(m_mesh.property(m_depth, *v_it));
		}
}

void Simulator::calculate_pressure() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		float b = m_mesh.property(m_bottom, *v_it);
		float d = m_mesh.property(m_depth, *v_it);
		m_mesh.property(m_height, *v_it) = b + d;
	}
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		IndexOnVertex *index_i = m_mesh.data(*v_it).index;
		float b = m_mesh.property(m_bottom, *v_it);
		float d = m_mesh.property(m_depth, *v_it);
		if (d <= m_depth_threshold) {
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::Point g_ind = index->from_nature_coord(m_g);
			float pg = -g_ind[2] * b;
			m_mesh.property(m_pressure_gravity, *v_it) = pg;
			if (m_mesh.is_boundary(*v_it)) {
				// 模型边界且是水的边界的点，从附近水的边界但非模型边界的点外插
				// 但也可能无法外插，需要初始化，以免得出不可预料的值
				m_mesh.property(m_pressure_surface, *v_it) = 0;
				continue;
			}
			bool close_to_water = m_mesh.property(m_on_water_boundary, *v_it);
			if (close_to_water) {
				float coef_LL = m_gamma * m_water_boundary_tension_multiplier;
				float coef_SO = coef_LL * (1 + cosf(m_water_boundary_theta)) / 2;
				float coef_LS = coef_LL * (1 - cosf(m_water_boundary_theta)) / 2;
				float extrapolate_depth = m_mesh.property(m_extrapolate_depth, *v_it);
				MyMesh::Point bottom_center = index->vertical_offset_point(b);
				MyMesh::Point normal_face = m_mesh.normal(*v_it);
				MyMesh::Point ex = MyMesh::Point(0, 0, 1) % normal_face;
				if (ex.norm() < 0.1)
					ex = MyMesh::Point(0, 1, 0) % normal_face;
				ex.normalize();
				MyMesh::Point ey = normal_face % ex;
				struct VVertex {
					MyMesh::VertexHandle vh;
					float depth;
					bool have_water;
					MyMesh::Point bottom_point;
					MyMesh::Point water_point;
				};
				std::vector<VVertex> ring;
				std::vector<MyMesh::VertexHandle> const &ring_vh(index->get_ring_1_ordered());
				for (auto vv_it = ring_vh.begin(); vv_it != ring_vh.end(); ++vv_it) {
					VVertex vertex;
					vertex.vh = *vv_it;
					float b = m_mesh.property(m_bottom, *vv_it);
					float d = m_mesh.property(m_depth, *vv_it);
					vertex.depth = d;
					vertex.have_water = (d > m_depth_threshold);
					MyMesh::Point bottom_point = m_mesh.data(*vv_it).index->vertical_offset_point(b);
					MyMesh::Point planed_bottom_point = bottom_point - bottom_center;
					float len = planed_bottom_point.norm();
					planed_bottom_point = MyMesh::Point(planed_bottom_point | ex, planed_bottom_point | ey, 0).normalized() * len;
					vertex.bottom_point = planed_bottom_point;
					vertex.water_point = MyMesh::Point(vertex.bottom_point[0], vertex.bottom_point[1], m_mesh.property(m_extrapolate_depth, *vv_it)); // 高度为外插的负高度（为了避免边缘面片因为采样问题而过薄）
					ring.push_back(vertex);
				}
				MyMesh::Point n(0, 0, 0);
				int num_water_boundary = 0;
				for (size_t i = 0; i < ring.size(); i++) {
					size_t prev = (i == 0) ? ring.size() - 1 : i - 1;
					size_t succ = (i == ring.size() - 1) ? 0 : i + 1;
					if (!ring[prev].have_water && !ring[i].have_water && ring[succ].have_water) {
						n += MyMesh::Point(ring[i].bottom_point[1], -ring[i].bottom_point[0], 0).normalized();
						num_water_boundary++;
					}
					else if (ring[prev].have_water && !ring[i].have_water && !ring[succ].have_water) {
						n += MyMesh::Point(-ring[i].bottom_point[1], ring[i].bottom_point[0], 0).normalized();
						num_water_boundary++;
					}
				}
				n.normalize();
				if (num_water_boundary == 2) {// && m_mesh.property(m_once_have_water, *v_it) == false) {
					auto partial_area = [](MyMesh::Point const &center, MyMesh::Point const &curr, MyMesh::Point const &succ) {
						MyMesh::Point b = center - succ;
						MyMesh::Point c = center - curr;
						MyMesh::Point area = (c % b) / 2;
						MyMesh::Point norm = area.normalized();
						float cosxy = norm[2]; // dot(norm, Vec3f(0,0,1.0f));
						float cosyz = norm[0]; // dot(norm, Vec3f(1.0f,0,0));
						float coszx = norm[1]; // dot(norm, Vec3f(0,1.0f,0));
						MyMesh::Point a = curr - succ;
						float par_x = 0.5f * (a[1]) * cosxy + 0.5f * (-a[2]) * coszx;
						float par_y = 0.5f * (a[2]) * cosyz + 0.5f * (-a[0]) * cosxy;
						float par_z = 0.5f * (a[0]) * coszx + 0.5f * (-a[1]) * cosyz;
						return MyMesh::Point(par_x, par_y, par_z);
					};
					static MyMesh::Point const o(0, 0, 0);
					MyMesh::Point F(0, 0, 0);
					float area_from_direct_n = 0;
					for (size_t i = 0; i < ring.size(); i++) {
						size_t succ = (i == ring.size() - 1) ? 0 : i + 1;
						if (ring[i].have_water || ring[succ].have_water) {
							F += partial_area(MyMesh::Point(0, 0, extrapolate_depth), ring[i].water_point, ring[succ].water_point) * coef_LL; // 高度为外插的负高度（为了避免边缘面片因为采样问题而过薄）
							F += partial_area(o, ring[i].bottom_point, ring[succ].bottom_point) * coef_LS;
							area_from_direct_n += (((ring[i].water_point - MyMesh::Point(0, 0, extrapolate_depth)) % (ring[succ].water_point - MyMesh::Point(0, 0, extrapolate_depth))) | n) / 6;
						}
						else {
							F += partial_area(o, ring[i].bottom_point, ring[succ].bottom_point) * coef_SO;
						}
					}
					float ps = (F | n) / area_from_direct_n;
					if (ps < -m_max_p_bs)
						ps = -m_max_p_bs;
					else if (ps > m_max_p_bs)
						ps = m_max_p_bs;
					//debug_out << "frame " << m_stepnum << ": p_g  " << pg << endl;
					//debug_out << "frame " << m_stepnum << ": p_bs " << ps << endl;
					m_mesh.property(m_pressure_surface, *v_it) = ps;
					continue;
				}
				else {
					// 周围水的情况混乱，按此点有水来计算压强ps，即不continue
				}
			}
			else {
				// 周围无水，不需要计算压强
				m_mesh.property(m_pressure_surface, *v_it) = 0;
				continue;
			}
		}
		float h = m_mesh.property(m_height, *v_it);
		MyMesh::Point g_ind = m_mesh.data(*v_it).index->from_nature_coord(m_g);
		float pg = -g_ind[2] * h;

		std::vector<MyMesh::Point> real1ringP;
		std::vector<MyMesh::VertexHandle> const &ringvh(index_i->get_ring_1_ordered());

		bool boundaryFlag = m_mesh.is_boundary(*v_it);

		for (auto vv_it = ringvh.begin(); vv_it != ringvh.end(); ++vv_it) {
			IndexOnVertex *index = m_mesh.data(*vv_it).index;
			float h = m_mesh.property(m_height, *vv_it);
			if (m_mesh.property(m_on_water_boundary, *vv_it))
				h = m_mesh.property(m_bottom, *vv_it) + m_mesh.property(m_extrapolate_depth, *vv_it);
			real1ringP.push_back(MyMesh::Point(index->vertical_offset_point(h)));
		}
		float h_i = m_mesh.property(m_height, *v_it);
		MyMesh::Point center_i(index_i->vertical_offset_point(h_i));

		MyMesh::Point normal_i = computeNormali(real1ringP, center_i, boundaryFlag);
		float ps = areaGradP(real1ringP, center_i, normal_i, boundaryFlag) * m_gamma;
		//debug_out << "frame " << m_stepnum << ": p_g  " << pg << endl;
		//debug_out << "frame " << m_stepnum << ": p_s  " << ps << endl;
		m_mesh.property(m_pressure_gravity, *v_it) = pg;
		m_mesh.property(m_pressure_surface, *v_it) = ps;
		// Surface tension pressure
	}

	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		if (m_mesh.is_boundary(*v_it) && m_mesh.property(m_depth, *v_it) <= m_depth_threshold) {
			bool is_first = true;
			float min_sqrlen;
			for (auto vv_it = m_mesh.vv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
				if (m_mesh.is_boundary(*vv_it) || m_mesh.property(m_depth, *vv_it) > m_depth_threshold)
					continue;
				MyMesh::Point delta = m_mesh.point(*vv_it) - m_mesh.point(*v_it);
				float len2 = dot(delta, delta);
				if (is_first || len2 < min_sqrlen) {
					min_sqrlen = len2;
					m_mesh.property(m_pressure_surface, *v_it) = m_mesh.property(m_pressure_surface, *vv_it);
					is_first = false;
				}
			}
		}
	}
}

void Simulator::update_velocity() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		float b = m_mesh.property(m_bottom, *v_it);
		float d = m_mesh.property(m_depth, *v_it);
		if (d <= m_depth_threshold) {
			m_mesh.property(m_vector_temp, *v_it) = m_mesh.property(m_velocity, *v_it);
			continue;
		}
		float h = m_mesh.property(m_height, *v_it);
		float pg = m_mesh.property(m_pressure_gravity, *v_it);
		float ps = m_mesh.property(m_pressure_surface, *v_it);
		float p = pg + ps;
		MyMesh::Point g_ind = m_mesh.data(*v_it).index->from_nature_coord(m_g);

		for (auto vv_it = m_mesh.cvv_iter(*v_it); vv_it.is_valid(); ++vv_it) {
			float vb = m_mesh.property(m_bottom, *vv_it);
			float vd = m_mesh.property(m_depth, *vv_it);
			float vpg = m_mesh.property(m_pressure_gravity, *vv_it);
			float vps = m_mesh.property(m_pressure_surface, *vv_it);
			float vp = vpg + vps;
			if (vd <= m_depth_threshold) {
#if 0
				// HACK: 强行外插
				vp = pg + vps;
#else
				if (g_ind[2] <= 0.0f)
					vp = (vb > h) ? pg + vps : vpg + vps;
				else
					vp = (vb > h) ? vpg + vps : pg + vps;
#endif
			}
			m_mesh.property(m_float_temp, *vv_it) = vp;
		}
		m_mesh.property(m_float_temp, *v_it) = p;
		MyMesh::Point grad = vertex_gradient(m_mesh, m_float_temp, *v_it, *v_it);
		MyMesh::Point vel = m_mesh.property(m_velocity, *v_it);
		if (m_have_tensor) {
			Tensor22 tensor(m_mesh.property(m_tensor, *v_it));
			vel += m_dt * -m_fric_coef * MyMesh::Point(tensor[0] * vel[0] + tensor[1] * vel[1], tensor[2] * vel[0] + tensor[3] * vel[1], 0);
		}
		vel += m_dt * (-grad + MyMesh::Point(g_ind[0], g_ind[1], 0));
		m_mesh.property(m_vector_temp, *v_it) = vel;
	}
	swap(m_velocity, m_vector_temp);
}

void Simulator::force_boundary_velocity() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it)
		if (m_mesh.is_boundary(*v_it))
			m_mesh.property(m_boundary, *v_it)->apply_velocity(m_mesh.property(m_velocity, *v_it));
}

void Simulator::velocity_fast_march() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		auto have_water = [&](MyMesh::VertexHandle vh) {
			return m_mesh.property(m_depth, vh) > m_depth_threshold;
		};
		if (m_mesh.property(m_depth, *v_it) <= m_depth_threshold) {
			MyMesh::Point vel;
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::VertexHandle vh = index->nearest_vertex(m_mesh, MyMesh::Point(0, 0, 0), have_water);
			if (vh == *m_mesh.vertices_end()) {
				vel = MyMesh::Point(0, 0, 0); // 此vertex周围都无水
			}
			else {
				vel = index->index_conv(m_mesh.data(vh).index, index, m_mesh.property(m_velocity, vh));
			}
			m_mesh.property(m_velocity, *v_it) = vel;
		}
	}
}

void Simulator::update_depth() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		float b = m_mesh.property(m_bottom, *v_it);
		float d = m_mesh.property(m_depth, *v_it);
		if (d <= m_depth_threshold)
			continue;
		float div = vertex_divergence(m_mesh, m_velocity, *v_it, *v_it);
		float term = 1 - m_dt * div;
		d *= term;
		if (d < 0.0f)
			d = 0.0f;
		// HACK: 强制限制水面高度
		// if (d > 2.0f) d = 1.8f;
		float h = b + d;
		//height[x_cells*m_mesh.point(*v_it)[0] + m_mesh.point(*v_it)[2]] = h;
		if (m_mesh.property(m_depth, *v_it) > 0) {
			height[m_mesh.point(*v_it)[0] * z_cells + m_mesh.point(*v_it)[2]] = m_mesh.property(m_depth, *v_it) + m_mesh.property(m_bottom, *v_it);
		}
		else {
			height[m_mesh.point(*v_it)[0] * z_cells + m_mesh.point(*v_it)[2]] = 0;
		}
		m_mesh.property(m_extrapolate_depth, *v_it) = m_mesh.property(m_depth, *v_it) = d;
		m_mesh.property(m_height, *v_it) = h;
	}
}

void Simulator::release_index() {
	m_mesh.release_face_normals();
	m_mesh.release_vertex_normals();
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		delete m_mesh.data(*v_it).index;
	}
}

void Simulator::release_properties() {
	m_mesh.remove_property(m_bottom);
	m_mesh.remove_property(m_depth);
	m_mesh.remove_property(m_height);
	m_mesh.remove_property(m_float_temp);
	m_mesh.remove_property(m_velocity);
	m_mesh.remove_property(m_midvel);
	m_mesh.remove_property(m_vector_temp);
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it)
		if (m_mesh.is_boundary(*v_it))
			delete m_mesh.property(m_boundary, *v_it);
	m_mesh.remove_property(m_boundary);
}

Simulator::BoundaryCondition::BoundaryCondition()
	:dtype(DEP_NOACTION), vtype(VEL_NOACTION) { }

void Simulator::BoundaryCondition::set_depth(DepthType type, Simulator *sim, MyMesh::VertexHandle vh) {
	this->dtype = type;
	switch (type) {
	case DEP_FIXED:
		dvalue0 = sim->m_mesh.property(sim->m_depth, vh);
		break;
	case DEP_NOACTION:
		break;
	default:
		break;
	};
}

void Simulator::BoundaryCondition::set_velocity(VelocityType type, Simulator *sim, MyMesh::VertexHandle vh) {
	this->vtype = type;
	switch (type) {
	case VEL_BOUND:
	{
		IndexOnVertex *index = sim->m_mesh.data(vh).index;
		MyMesh::Point in_direct(0, 0, 0);
		MyMesh::Point out_direct(0, 0, 0);
		for (auto vih_it = sim->m_mesh.cvih_iter(vh); vih_it.is_valid(); ++vih_it) {
			if (sim->m_mesh.is_boundary(*vih_it)) {
				MyMesh::Point b(sim->m_mesh.point(sim->m_mesh.from_vertex_handle(*vih_it)));
				in_direct += -index->plane_map(b);
			}
		}
		for (auto voh_it = sim->m_mesh.cvoh_iter(vh); voh_it.is_valid(); ++voh_it) {
			if (sim->m_mesh.is_boundary(*voh_it)) {
				MyMesh::Point c(sim->m_mesh.point(sim->m_mesh.to_vertex_handle(*voh_it)));
				out_direct += index->plane_map(c);
			}
		}
		vvalue0 = (in_direct + out_direct).normalized();
		// Todo: 处理介于直角与平边之间的情况
		if ((in_direct.normalized() | out_direct.normalized()) < 0.5f)
			vvalue0 = MyMesh::Point(0, 0, 0);
	}
	break;
	case VEL_FIXED:
		vvalue0 = sim->m_mesh.property(sim->m_velocity, vh);
		break;
	case VEL_NOACTION:
		break;
	default:
		break;
	};
}

void Simulator::BoundaryCondition::apply_depth(float &depth) {
	switch (dtype) {
	case DEP_FIXED:
		depth = dvalue0;
		break;
	case DEP_NOACTION:
		break;
	default:
		break;
	};
}

void Simulator::BoundaryCondition::apply_velocity(MyMesh::Point &velocity) {
	switch (vtype) {
	case VEL_BOUND:
		velocity = (velocity | vvalue0) * vvalue0;
		break;
	case VEL_FIXED:
		velocity = vvalue0;
		break;
	case VEL_NOACTION:
		break;
	default:
		break;
	};
}
}
