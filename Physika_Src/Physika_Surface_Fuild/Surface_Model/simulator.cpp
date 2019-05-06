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
#include "swe.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#pragma warning(disable: 4258)

using namespace std;

#define COMPUTE_DIV_METHOD 2
namespace Physika{
Simulator::Simulator() { }
Simulator::~Simulator() { }
void Simulator::setsituation(int situation){
	m_situation=situation;
}
void Simulator::setinitcon(float threshold,bool havetensor,float fric_coef,float gamma,float boundary_theta,float boundary_tension_multiplier,float max_p_bs){
	m_depth_threshold=threshold;
	m_have_tensor=havetensor;
	m_fric_coef=fric_coef;
	m_gamma=gamma;
	m_water_boundary_theta=boundary_theta;
	m_water_boundary_tension_multiplier=boundary_tension_multiplier;
	m_max_p_bs=max_p_bs;
}
void Simulator::setframecon(int stepnum,int totalstep,int outstep,flaot dt){
	m_stepnum=stepnum;
	m_total_steps=totalstep;
	m_output_step=outstep;
	m_dt=dt;
}
void Simulator::init(int argc, char** argv) {
	cout << "seed: " << rand() << endl;
	m_situation = 9;
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

	// cuda
	prepareData ();
	cudaInit ( argc, argv );
	setupCuda(*this);
	copyToCuda(*this);

	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: FSInitialAdd0: %s\n", cudaGetErrorString(error));
	}
}

void Simulator::run_cuda ( int frame )
{
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: FSInitialAdd10: %s\n", cudaGetErrorString(error));
	}
	WindowsTimer timer;
	if (frame <= m_total_steps) {
		timer.restart ();
		processCuda(*this); // cuda process
		timer.stop ();
		printf ( "calculate frame %d used %f seconds\n", m_stepnum, timer.get () );
		copyFromCuda(*this); // copy back to cpu for render
		post_data ();
		m_stepnum++;
	}
}

void Simulator::output_obj_cuda(int frame) {
	CoordSystem g_system = gen_coord_system_by_z_axis(m_g);
	CoordSystem screen_system = gen_coord_system_by_z_axis(MyMesh::Point(0, -1, 0));
	auto rotate_by_g = [&](MyMesh::Point p) {
		if (m_situation == 9 || m_situation == 13) {
			MyMesh::Point delta(p - m_rotate_center);
			MyMesh::Point coord(delta | g_system[0], delta | g_system[1], delta | g_system[2]);
			coord = IndexOnVertex::coord_conv(g_system.data(), screen_system.data(), coord);
			delta = coord[0] * screen_system[0] + coord[1] * screen_system[1] + coord[2] * screen_system[2];
			return delta + m_rotate_center;
		}
		else {
			return p;
		}
	};

	std::stringstream objbuffer;
	objbuffer << "result/obj/frame_" << frame << ".obj";
	std::string objfn;
	objbuffer >> objfn;
	std::ofstream obj(objfn);
	if (!obj.is_open()) {
		perror("output error");
		exit(-1);
	}
	// HACK: 避免水在模型下方的问题
	float const h_gain = 0.10f;
	int num_v_start = 1;
	for (size_t i = 0; i < m_origin.n_vertices(); i++) {
		MyMesh::VertexHandle vh((int)i);
		MyMesh::Point p(rotate_by_g(m_origin.point(vh)));
		float pic_size = 20.0f;
		float tx = 0;
		float ty = p[2] / 100.0f;
		MyMesh::TexCoord2D tc(tx, ty);
		if (m_origin.has_vertex_texcoords2D())
			tc = m_origin.texcoord2D(vh);
		obj << "v" << "  " << p[0] << " " << p[1] << " " << p[2] << endl;
		obj << "vt" << "  " << tc[0] << " " << tc[1] << endl;
	}
	for (size_t i = 0; i < m_mesh.n_vertices(); i++) {
		MyMesh::VertexHandle vh((int)i);
		IndexOnVertex *index = m_mesh.data(vh).index;
		auto b = m_mesh.property(m_bottom, vh);
		auto h = b + m_mesh.property(m_depth, vh);
		h += h_gain;
		MyMesh::Point p(rotate_by_g(index->vertical_offset_point(h)));
		obj << "v" << "  " << p[0] << " " << p[1] << " " << p[2] << endl;
	}
	obj << "usemtl" << "  " << "Bottom" << endl;
	for (auto f_it = m_origin.faces_begin(); f_it != m_origin.faces_end(); ++f_it) {
		auto fv_it = m_origin.cfv_ccwbegin(*f_it);
		MyMesh::VertexHandle v1 = *fv_it++;
		MyMesh::VertexHandle v2 = *fv_it++;
		MyMesh::VertexHandle v3 = *fv_it++;
		obj << "f" << "  " << (v1.idx() + num_v_start) << "/" << (v1.idx() + num_v_start) << " " <<
			(v2.idx() + num_v_start) << "/" << (v2.idx() + num_v_start) << " " <<
			(v3.idx() + num_v_start) << "/" << (v3.idx() + num_v_start) << endl;
	}
	num_v_start += (int)m_origin.n_vertices();
	int maxvnum = num_v_start + (int)m_mesh.n_vertices() - 1;
	obj << "usemtl" << "  " << "Water" << endl;
	for (auto f_it = m_mesh.faces_begin(); f_it != m_mesh.faces_end(); ++f_it) {
		auto fv_it = m_mesh.cfv_ccwbegin(*f_it);
		MyMesh::VertexHandle vh = *fv_it;
		MyMesh::VertexHandle v1 = *fv_it++;
		MyMesh::VertexHandle v2 = *fv_it++;
		MyMesh::VertexHandle v3 = *fv_it++;
		static auto const have_water = [&](MyMesh::VertexHandle vh) {
			return m_mesh.property(m_depth, vh) > m_depth_threshold;
		};
		int have_water_cnt = 0;
		if (have_water(v1)) have_water_cnt++;
		if (have_water(v2)) have_water_cnt++;
		if (have_water(v3)) have_water_cnt++;
		if (have_water_cnt == 3) {
			obj << "f" << "  " << (v1.idx() + num_v_start) << " " << (v2.idx() + num_v_start) << " " << (v3.idx() + num_v_start) << endl;
		}
		else if (have_water_cnt == 2) {
			for (int i = 0; i < 2; i++) {
				if (have_water(v3)) {
					swap(v2, v3);
					swap(v1, v2);
				}
			}
			float d1 = m_mesh.property(m_depth, v1);
			float d2 = m_mesh.property(m_depth, v2);
			float d3 = m_mesh.property(m_depth, v3);
			if (d3 > 0) d3 = 0;
			MyMesh::Point p1 = rotate_by_g(m_mesh.data(v1).index->vertical_offset_point(m_mesh.property(m_bottom, v1) + d1 + h_gain));
			MyMesh::Point p2 = rotate_by_g(m_mesh.data(v2).index->vertical_offset_point(m_mesh.property(m_bottom, v2) + d2 + h_gain));
			MyMesh::Point p3 = rotate_by_g(m_mesh.data(v3).index->vertical_offset_point(m_mesh.property(m_bottom, v3) + d3 + h_gain));
			MyMesh::Point p4 = p1 + (d1 / (d1 - d3)) * (p3 - p1);
			MyMesh::Point p5 = p2 + (d2 / (d2 - d3)) * (p3 - p2);
			obj << "v" << "  " << p4[0] << " " << p4[1] << " " << p4[2] << endl;
			obj << "v" << "  " << p5[0] << " " << p5[1] << " " << p5[2] << endl;
			obj << "f" << "  " << (v1.idx() + num_v_start) << " " << (v2.idx() + num_v_start) << " " << (maxvnum + 2) << endl;
			obj << "f" << "  " << (v1.idx() + num_v_start) << " " << (maxvnum + 2) << " " << (maxvnum + 1) << endl;
			maxvnum += 2;
		}
		else if (have_water_cnt == 1) {
			for (int i = 0; i < 2; i++) {
				if (!have_water(v1)) {
					swap(v2, v3);
					swap(v1, v2);
				}
			}
			float d1 = m_mesh.property(m_depth, v1);
			float d2 = m_mesh.property(m_depth, v2);
			if (d2 > 0) d2 = 0;
			float d3 = m_mesh.property(m_depth, v3);
			if (d3 > 0) d3 = 0;
			MyMesh::Point p1 = rotate_by_g(m_mesh.data(v1).index->vertical_offset_point(m_mesh.property(m_bottom, v1) + d1 + h_gain));
			MyMesh::Point p2 = rotate_by_g(m_mesh.data(v2).index->vertical_offset_point(m_mesh.property(m_bottom, v2) + d2 + h_gain));
			MyMesh::Point p3 = rotate_by_g(m_mesh.data(v3).index->vertical_offset_point(m_mesh.property(m_bottom, v3) + d3 + h_gain));
			MyMesh::Point p4 = p1 + (d1 / (d1 - d2)) * (p2 - p1);
			MyMesh::Point p5 = p1 + (d1 / (d1 - d3)) * (p3 - p1);
			obj << "v" << "  " << p4[0] << " " << p4[1] << " " << p4[2] << endl;
			obj << "v" << "  " << p5[0] << " " << p5[1] << " " << p5[2] << endl;
			obj << "f" << "  " << (v1.idx() + num_v_start) << " " << (maxvnum + 1) << " " << (maxvnum + 2) << endl;
			maxvnum += 2;
		}
		else {
			// do nothing
		}
	}
	obj.close();
}

void Simulator::run() {
	WindowsTimer timer;
	auto f = [&]() {
		timer.record();
		return timer.get();
	};
	WindowsTimer::time_t t[6] = {0, 0, 0, 0, 0, 0};
	while (m_stepnum <= m_total_steps) {
		if (m_stepnum % 10 == 0) {
			timer.restart();
			output_obj();
			timer.stop();
			printf("output frame %d used %f seconds\n", m_stepnum, timer.get());
		}

		timer.restart();
		update_midvels();
		t[0] += f();
		advect_filed_values();
		extrapolate_depth();
		t[1] += f();
		force_boundary_depth();
		calculate_pressure();
		update_velocity();
		t[2] += f();
		force_boundary_velocity();
		t[3] += f();
		velocity_fast_march();
		t[4] += f();
		update_depth();
		t[5] += f();
		timer.stop();
		printf("calculate frame %d used %f seconds\n", m_stepnum + 1, timer.get());
		printf("%f %f %f %f %f %f\n", t[0], t[1] - t[0], t[2] - t[1], t[3] - t[2], t[4] - t[3], t[5] - t[4]);
		m_stepnum++;
	}
}

void Simulator::clear() {
	release_index();
	release_properties();
}

void Simulator::set_initial_constants() {
	m_g = MyMesh::Point(0, -1, 0).normalized() * 9.80f;
	m_rotate_center = MyMesh::Point(0, 0, 0);

	// 摩擦力系数
	m_have_tensor = false;
	m_fric_coef = 1.3f;

	// 表面张力的系数
	m_gamma = 1.000f;
	// gamma is surface tension coefficient divided by density, theoretical value for 25 degree(C) water is 7.2*10^-5 (N/m)/(kg/m^3)
	// note: surface tension force not negligible under length scale less than about 4mm, so gamma value should be set with careful considering mesh edge length. Try using mm or cm instead of m
	m_water_boundary_theta = (float)M_PI / 180 * 30.0f;
	m_water_boundary_tension_multiplier = 1.0f;
	m_max_p_bs = 10.0f;

	// 风力
	m_wind_coef = 0;

	// 模拟帧设置
	m_stepnum = 0;
	m_total_steps = 1000;
	m_output_step = 10;
	m_dt = 0.033f;

	switch (m_situation) {
	case 4:
		m_have_tensor = true;
		m_fric_coef = 6.667f;
		m_total_steps = 1500;
		m_dt = 0.0165f;
		break;
	case 7:
	case 10:
		m_have_tensor = true;
		m_total_steps = 1000;
		m_output_step = 20;
		m_dt = 0.01f;
		break;
	case 9:
		m_g = MyMesh::Point(1, -2, 0).normalized() * 9.80f;
		m_rotate_center = MyMesh::Point(25, 0, 25);
		m_have_tensor = true;
		m_fric_coef = 2.5f;
		m_gamma = 5.000f;
		m_total_steps = 1600;
		m_dt = 0.02f;
		break;
	case 16:
		m_have_tensor = false;
		m_fric_coef = 2.5f;
		m_gamma = 0.500f;
		m_wind_coef = 1.0f;
		m_total_steps = 1000;
		m_dt = 0.02f;
		break;
	default:
		break;
	}
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
			float xoff = (-xbound.get_min() - xbound.get_max()) / 2 + 26.0f / ratio;
			float yoff = -ybound.get_min() - 25.0f / ratio;
			float zoff = (-zbound.get_min() - zbound.get_max()) / 2 + 25.0f / ratio;
			for (auto v_it = m_origin.vertices_begin(); v_it != m_origin.vertices_end(); ++v_it) {
				MyMesh::Point p(m_origin.point(*v_it));
				MyMesh::Point np((p[0] + xoff) * ratio, (p[1] + yoff) * ratio - 0.5f, (p[2] + zoff) * ratio);
				m_origin.point(*v_it) = np;
			}
		}
		break;
	case 5: case 6:
		{
			enum { MRES = 101 };
			float const grid_size = 0.5f;
			MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES];
			for (int i = 0; i < MRES; i++)
				vhandle[i] = new MyMesh::VertexHandle[MRES];
			for (int i = 0; i < MRES; i++)
				for (int j = 0; j < MRES; j++) {
					float r = (MyMesh::Point(i * grid_size, 0, j * grid_size) - MyMesh::Point(25.0f, 0, 25.0f)).norm();
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
			float const grid_size = 0.5f;
			MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES];
			for (int i = 0; i < MRES; i++)
				vhandle[i] = new MyMesh::VertexHandle[MRES];
			std::vector<MyMesh::VertexHandle>  face_vhandles;
			for (int i = 0; i < MRES; i++)
				for (int j = 0; j < MRES; j++) {
					float b = 0.05f * (cos(2.0f * 0.5f * (float)i + 2.0f * 0.5f * (float)j) + 1.01f);
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
				} else {
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
			  enum { MRES_X = 121, MRES_Z = 101 };
			  float const grid_size = 0.5f;
			  MyMesh::VertexHandle **vhandle = new MyMesh::VertexHandle *[MRES_X];
			  for (int i = 0; i < MRES_X; i++)
				  vhandle[i] = new MyMesh::VertexHandle[MRES_Z];
			  std::vector<MyMesh::VertexHandle>  face_vhandles;
			  for (int i = 0; i < MRES_X; i++)
			  for (int j = 0; j < MRES_Z; j++) {
				  float b = 0.05f * (cos(2.0f * 0.5f * (float)i + 2.0f * 0.5f * (float)j) + 1.01f);
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
		break;
	case 16:
	{
			   char const input_filename[] = "obj_models/Creature.simple.obj";
			   OpenMesh::IO::read_mesh(m_origin, input_filename);
			   float rotate_yz = (float)M_PI * (160.0f / 180.0f);
			   for (auto v_it = m_origin.vertices_begin(); v_it != m_origin.vertices_end(); ++v_it) {
				   MyMesh::Point p(m_origin.point(*v_it));
				   p = MyMesh::Point(p[0], p[1] * cos(rotate_yz) + p[2] * sin(rotate_yz), -p[1] * sin(rotate_yz) + p[2] * cos(rotate_yz));
				   m_origin.point(*v_it) = p;
			   }
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
		break;
	case 16:
		m_mesh = m_origin;
		break;
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
	m_mesh.add_property(m_on_water_boundary);
	m_mesh.add_property(m_pressure_gravity);
	m_mesh.add_property(m_pressure_surface);
	m_mesh.add_property(m_float_temp);
	m_mesh.add_property(m_velocity);
	m_mesh.add_property(m_midvel);
	m_mesh.add_property(m_vector_temp);
	m_mesh.add_property(m_origin_face);
	m_mesh.add_property(m_tensor);
	m_mesh.add_property(m_wind_velocity);
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
	cout << "create " << ring << "-ring index for " << timer.get() << " sec, " << ((float)memory_cost / 1024 / 1024) << " MB" << endl;
	cout << "average: " << avg_faces_per_vertex << " faces per vertex (in " << ring << "-ring)" << endl;
}

void Simulator::match_bottom_height() {
	// Todo: 用类似光线跟踪的kd-tree方法加速
	std::map<int, std::string> cache_filename;
	cache_filename[9] = "obj_models/trunk.bottom.txt";
	cache_filename[16] = "obj_models/Creature.bottom.txt";
	if (cache_filename.find(m_situation) != cache_filename.end()) {
		std::string filename(cache_filename[m_situation]);
		ifstream in;
		in.open(filename);
		if (!!in) {
			cout << "从 " << filename << " 导入平滑网格到原网格的映射" << endl;
			for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
				float bottom;
				int fidx;
				in >> bottom >> fidx;
				m_mesh.property(m_bottom, *v_it) = bottom;
				m_mesh.property(m_origin_face, *v_it) = MyMesh::FaceHandle(fidx);
			}
			in.close();
			return;
		}
	}
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
		} else {
			m_mesh.property(m_bottom, *v_it) = dist;
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
				q.push(*vv_it);
				set_not_matched.insert(*vv_it);
			}
		}
	}
	if (n_not_matched != set_not_matched.size())
		cout << "错误：有些法向不与原网格相交的点未能从附近点外插" << endl;
	if (cache_filename.find(m_situation) != cache_filename.end()) {
		std::string filename(cache_filename[m_situation]);
		ofstream out(filename);
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			out << m_mesh.property(m_bottom, *v_it) << ' ' << m_mesh.property(m_origin_face, *v_it).idx() << endl;
		}
		out.close();
	}
}

void Simulator::calculate_tensor() {
	std::map<int, std::string> cache_filename;
	cache_filename[9] = "obj_models/trunk.tensor.txt";
	cache_filename[16] = "obj_models/Creature.tensor.txt";
	if (!m_have_tensor)
		return;
	if (cache_filename.find(m_situation) != cache_filename.end()) {
		std::string filename(cache_filename[m_situation]);
		ifstream in;
		in.open(filename);
		if (!!in) {
			cout << "从 " << filename << " 导入摩擦力张量" << endl;
			for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
				Tensor22 tensor;
				in >> tensor[0] >> tensor[1] >> tensor[2] >> tensor[3];
				m_mesh.property(m_tensor, *v_it) = tensor;
			}
			in.close();
			return;
		}
	}
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
			MyMesh::Point tri0[3] = {a, b, c};
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
				MyMesh::Point tri1[3] = {o, (o + a) / 2, (o + a + b) / 3};
				if (have_crossed_volumn(tri0, tri1))
					return true;
				MyMesh::Point tri2[3] = {o, (o + a + b) / 3, (o + b) / 2};
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
			} else {
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
	if (cache_filename.find(m_situation) != cache_filename.end()) {
		std::string filename(cache_filename[m_situation]);
		ofstream out(filename);
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			Tensor22 tensor(m_mesh.property(m_tensor, *v_it));
			out << tensor[0] << ' ' << tensor[1] << ' ' << tensor[2] << ' ' << tensor[3] << endl;
		}
		out.close();
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
			m_mesh.property(m_depth, *v_it) = p[1] > 15 && p[2] > 10 ? 0.19f : 0;
			m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
		}
		break;
	case 5:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			MyMesh::Point midpos(25.0f, 0, 25.0f);
			MyMesh::Point p(m_mesh.point(*v_it));
			float dis2 = (p - midpos).sqrnorm();
			m_mesh.property(m_depth, *v_it) = (p - midpos).norm() < 9.01f ? 4.0f + 2.0f * exp(-dis2) : 4.0f;
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
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
			}
		}
		break;
	case 7:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::Point p(m_mesh.point(*v_it));
			m_mesh.property(m_depth, *v_it) = (p[0] < 0.1f && p[2] > 24.9f) ? 0.5f - m_mesh.property(m_bottom, *v_it) : 0;
			m_mesh.property(m_velocity, *v_it) = index->from_nature_coord(MyMesh::Point(7, 0, 0));
			if (m_mesh.is_boundary(*v_it)) {
				if (m_mesh.property(m_depth, *v_it) > m_depth_threshold) {
					m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
				} else {
					m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_BOUND, this, *v_it);
				}
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
			m_mesh.property(m_depth, *v_it) = (p[0] < 10.0f && p[2] < 44.0f && p[2] > 20.0f) ? 1.5f : 0.0f;
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
	case 16:
		for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
			IndexOnVertex *index = m_mesh.data(*v_it).index;
			MyMesh::Point p(m_mesh.point(*v_it));
			float b = m_mesh.property(m_bottom, *v_it);
			float d = p[2] > -10 && p[2] < -5 && p[0] > -5 && p[0] < 6.6f && p[1] > 2 ? 0.60f : 0;
			m_mesh.property(m_depth, *v_it) = d;
			m_mesh.property(m_velocity, *v_it) = index->from_nature_coord(MyMesh::Point(0, 0, 0));
			if (m_mesh.is_boundary(*v_it)) {
				m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
				m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_FIXED, this, *v_it);
			}
			MyMesh::Point wind_vel = index->from_nature_coord(MyMesh::Point(1, 0, 0));
			wind_vel[2] = 0;
			if (wind_vel.sqrnorm() > 0) {
				wind_vel = wind_vel.normalized() * 1.00f;
			}
			m_mesh.property(m_wind_velocity, *v_it) = wind_vel;
		}
		break;
	default:
		break;
	}
	extrapolate_depth();
}

void Simulator::output_obj() {
	output_obj_cuda(m_stepnum);
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
			} else {
				depth_new = m_mesh.property(m_depth, vh);
				velocity_new = m_mesh.property(m_velocity, vh);
				velocity_new = IndexOnVertex::index_conv(m_mesh.data(vh).index, index, velocity_new);
				if (m_mesh.is_boundary(*v_it)) {
					auto bound = m_mesh.property(m_boundary, *v_it);
					bound->apply_depth(depth_new);
					bound->apply_velocity(velocity_new);
				}
			}
		}
		else {
			depth_new = point_interpolation(m_mesh, m_depth, p, *v_it, fh);
			auto fv_it = m_mesh.fv_iter(fh);
			MyMesh::VertexHandle vh1 = *fv_it++;
			MyMesh::VertexHandle vh2 = *fv_it++;
			MyMesh::VertexHandle vh3 = *fv_it;
			float face_depth_max = fmax(m_mesh.property(m_depth, vh1), fmax(m_mesh.property(m_depth, vh2), m_mesh.property(m_depth, vh3)));
			float face_depth_min = fmin(m_mesh.property(m_depth, vh1), fmin(m_mesh.property(m_depth, vh2), m_mesh.property(m_depth, vh3)));
			depth_new = fmin(fmax(depth_new, face_depth_min), face_depth_max);
			velocity_new = point_interpolation(m_mesh, m_velocity, p, *v_it, fh);
		}
		m_mesh.property(m_float_temp, *v_it) = depth_new;
		m_mesh.property(m_vector_temp, *v_it) = velocity_new;
	}
	if (COMPUTE_DIV_METHOD <= 2)
		swap(m_depth, m_float_temp);
	swap(m_velocity, m_vector_temp);
}

void Simulator::extrapolate_depth() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		m_mesh.property(m_on_water_boundary, *v_it) = false;
		float d = m_mesh.property(m_depth, *v_it);
		m_mesh.property(m_float_temp, *v_it) = d;
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
					m_mesh.property(m_float_temp, *v_it) = 0;
				}
				else {
					float extrapolation = ex_depth / cnt;
					if (extrapolation < 0)
						m_mesh.property(m_float_temp, *v_it) = extrapolation;
					else
						m_mesh.property(m_float_temp, *v_it) = 0;
				}
			}
			else {
				// 此点无水且周围也无水
				// 设置一个负值，防止插值时因浮点数误差而出错
				m_mesh.property(m_float_temp, *v_it) = -1e-4f;
			}
		}
	}
	swap(m_depth, m_float_temp);
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
				float extrapolate_depth = m_mesh.property(m_depth, *v_it);
				MyMesh::Point bottom_center = index->vertical_offset_point(b);
				MyMesh::Point normal_face = m_mesh.normal(*v_it);
				MyMesh::Point ex = MyMesh::Point(0, 0, 1) % normal_face;
				if (ex.norm() < 0.1)
					ex = MyMesh::Point(0, 1, 0) % normal_face;
				ex.normalize();
				MyMesh::Point ey = normal_face % ex;
				struct VVertex {
					MyMesh::VertexHandle vh;
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
					vertex.have_water = (d > m_depth_threshold);
					MyMesh::Point bottom_point = m_mesh.data(*vv_it).index->vertical_offset_point(b);
					MyMesh::Point planed_bottom_point = bottom_point - bottom_center;
					float len = planed_bottom_point.norm();
					planed_bottom_point = MyMesh::Point(planed_bottom_point | ex, planed_bottom_point | ey, 0).normalized() * len;
					vertex.bottom_point = planed_bottom_point;
					vertex.water_point = MyMesh::Point(vertex.bottom_point[0], vertex.bottom_point[1], m_mesh.property(m_depth, *vv_it)); // 高度为外插的负高度（为了避免边缘面片因为采样问题而过薄）
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
		float h = b + d;
		MyMesh::Point g_ind = m_mesh.data(*v_it).index->from_nature_coord(m_g);
		float pg = -g_ind[2] * h;

		std::vector<MyMesh::Point> real1ringP;
		std::vector<MyMesh::VertexHandle> const &ringvh(index_i->get_ring_1_ordered());

		bool boundaryFlag = m_mesh.is_boundary(*v_it);

		for (auto vv_it = ringvh.begin(); vv_it != ringvh.end(); ++vv_it){
			IndexOnVertex *index = m_mesh.data(*vv_it).index;
			float h = m_mesh.property(m_bottom, *vv_it) + m_mesh.property(m_depth, *vv_it);
			real1ringP.push_back(MyMesh::Point(index->vertical_offset_point(h)));
		}
		float h_i = b + d;
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
		// HACK: 多计算一圈v
		if (d <= m_depth_threshold && !m_mesh.property(m_on_water_boundary, *v_it)) {
			m_mesh.property(m_vector_temp, *v_it) = m_mesh.property(m_velocity, *v_it);
			continue;
		}
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
#if 1
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
			// vel += vel * m_dt * -10.0f; ////Added some viscosicty
		}
		vel += m_dt * (-grad + MyMesh::Point(g_ind[0], g_ind[1], 0));
		// 风力
		if (m_wind_coef != 0) {
			MyMesh::Point wind_vel = m_mesh.property(m_wind_velocity, *v_it);
			wind_vel[2] = 0;
			vel += m_dt * m_wind_coef * wind_vel * wind_vel.norm();
		}
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
			// HACK: 多计算一圈v，从外围开始外插
			return m_mesh.property(m_depth, vh) > m_depth_threshold || m_mesh.property(m_on_water_boundary, vh);
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
#if COMPUTE_DIV_METHOD == 1
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		float b = m_mesh.property(m_bottom, *v_it);
		float d = m_mesh.property(m_depth, *v_it);
		if (d <= m_depth_threshold)
			continue;
		float div = vertex_divergence(m_mesh, m_velocity, *v_it, *v_it);
		float term = 1 - m_dt * div;
		d *= term;
		if (d > 0 && d <= m_depth_threshold)
			d = 0;
		// HACK: 强制限制水面高度
		if (m_situation == 9 && d > 2.0f)
			d = 1.8f;
		m_mesh.property(m_depth, *v_it) = d;
	}
#elif COMPUTE_DIV_METHOD == 2
	auto _divergence_by_area = [](MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c, MyMesh::Point fa, MyMesh::Point fb, MyMesh::Point fc) {
		auto dx = (b[1] - c[1]) * fa[0] + (c[1] - a[1]) * fb[0] + (a[1] - b[1]) * fc[0];
		auto dy = (c[0] - b[0]) * fa[1] + (a[0] - c[0]) * fb[1] + (b[0] - a[0]) * fc[1];
		return dx / 2 + dy / 2;
	};
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		float b = m_mesh.property(m_bottom, *v_it);
		float d = m_mesh.property(m_depth, *v_it);
		if (d <= m_depth_threshold)
			continue;
		IndexOnVertex *index = m_mesh.data(*v_it).index;
		float sum = 0;
		float sum_area = 0;
		for (auto vf_it = m_mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
			auto fv_it = m_mesh.fv_ccwiter(*vf_it);
			MyMesh::VertexHandle vh1 = *fv_it++;
			MyMesh::VertexHandle vh2 = *fv_it++;
			MyMesh::VertexHandle vh3 = *fv_it;
			for (int i = 0; i < 2; i++) {
				if (vh1 != *v_it) {
					swap(vh1, vh2);
					swap(vh1, vh3);
				}
			}
			MyMesh::Point p1 = index->plane_map(m_mesh.point(vh1));
			float d1 = fmax(m_mesh.property(m_depth, vh1), 0.0f);
			MyMesh::Point u1 = IndexOnVertex::index_conv(m_mesh.data(vh1).index, index, m_mesh.property(m_velocity, vh1));
			MyMesh::Point p2 = index->plane_map(m_mesh.point(vh2));
			float d2 = fmax(m_mesh.property(m_depth, vh2), 0.0f);
			MyMesh::Point u2 = IndexOnVertex::index_conv(m_mesh.data(vh2).index, index, m_mesh.property(m_velocity, vh2));
			MyMesh::Point p3 = index->plane_map(m_mesh.point(vh3));
			float d3 = fmax(m_mesh.property(m_depth, vh3), 0.0f);
			MyMesh::Point u3 = IndexOnVertex::index_conv(m_mesh.data(vh3).index, index, m_mesh.property(m_velocity, vh3));
			float area = ((p2 - p1) % (p3 - p1))[2] / 2;
			float divv = _divergence_by_area(p1, p2, p3, u1, u2, u3) / area;
			sum_area += area;
			sum += area * (d1 + d2 + d3) / 3 * divv;
		}
		m_mesh.property(m_float_temp, *v_it) = -m_dt * sum / sum_area;
	}
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		float b = m_mesh.property(m_bottom, *v_it);
		float d = m_mesh.property(m_depth, *v_it);
		if (d <= m_depth_threshold)
			continue;
		float delta_d = m_mesh.property(m_float_temp, *v_it);
		d += delta_d;
		// HACK: 强制限制水面高度
		if (m_situation == 9 && d > 2.0f)
			d = 1.8f;
		m_mesh.property(m_depth, *v_it) = d;
	}
#elif COMPUTE_DIV_METHOD == 3
	auto out_of_S1 = [&](MyMesh::VertexHandle vh) {
		return m_mesh.property(m_depth, vh) <= m_depth_threshold && !m_mesh.property(m_on_water_boundary, vh);
	};
	auto in_S = [&](MyMesh::VertexHandle vh) {
		return m_mesh.property(m_depth, vh) > m_depth_threshold;
	};
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		if (out_of_S1(*v_it))
			continue;
		IndexOnVertex *index = m_mesh.data(*v_it).index;
		float sum = 0;
		float sum_area = 0;
		for (auto vf_it = m_mesh.vf_iter(*v_it); vf_it.is_valid(); ++vf_it) {
			auto fv_it = m_mesh.fv_ccwiter(*vf_it);
			MyMesh::VertexHandle vh1 = *fv_it;
			++fv_it;
			MyMesh::VertexHandle vh2 = *fv_it;
			++fv_it;
			MyMesh::VertexHandle vh3 = *fv_it;
			for (int i = 0; i < 2; i++) {
				if (vh1 != *v_it) {
					swap(vh1, vh2);
					swap(vh1, vh3);
				}
			}
			MyMesh::Point p1 = index->plane_map(m_mesh.point(vh1));
			float d1 = m_mesh.property(m_depth, vh1);
			MyMesh::Point u1 = IndexOnVertex::index_conv(m_mesh.data(vh1).index, index, m_mesh.property(m_velocity, vh1));
			MyMesh::Point p2 = index->plane_map(m_mesh.point(vh2));
			float d2 = m_mesh.property(m_depth, vh2);
			MyMesh::Point u2 = IndexOnVertex::index_conv(m_mesh.data(vh2).index, index, m_mesh.property(m_velocity, vh2));
			MyMesh::Point p3 = index->plane_map(m_mesh.point(vh3));
			float d3 = m_mesh.property(m_depth, vh3);
			MyMesh::Point u3 = IndexOnVertex::index_conv(m_mesh.data(vh3).index, index, m_mesh.property(m_velocity, vh3));
			bool flag1 = in_S(vh1);
			bool flag2 = in_S(vh2);
			bool flag3 = in_S(vh3);
			if (!flag1 && !flag2 && !flag3)
				continue;
			if (!flag1 && !flag2) {
				float d = d1 * 3 / 4 + d2 * 1 / 4;
				MyMesh::Point u = u1 * 3 / 4 + u2 * 1 / 4;
				MyMesh::Point nA(p2[1] / 2 - p1[1] / 2, -p2[0] / 2 + p1[0] / 2, 0);
				sum += d * (u | nA);
			}
			if (!flag1 && !flag3) {
				float d = d1 * 3 / 4 + d3 * 1 / 4;
				MyMesh::Point u = u1 * 3 / 4 + u3 * 1 / 4;
				MyMesh::Point nA(-p3[1] / 2 + p1[1] / 2, p3[0] / 2 - p1[0] / 2, 0);
				sum += d * (u | nA);
			}
			float df1 = d1 * 5 / 12 + d2 * 5 / 12 + d3 * 2 / 12;
			MyMesh::Point uf1 = u1 * 5 / 12 + u2 * 5 / 12 + u3 * 2 / 12;
			MyMesh::Point nAf1(p3[1] / 3 - p2[1] / 6 - p1[1] / 6, -p3[0] / 3 + p2[0] / 6 + p1[0] / 6, 0);
			float df2 = d1 * 5 / 12 + d2 * 2 / 12 + d3 * 5 / 12;
			MyMesh::Point uf2 = u1 * 5 / 12 + u2 * 2 / 12 + u3 * 5 / 12;
			MyMesh::Point nAf2(-p2[1] / 3 + p3[1] / 6 + p1[1] / 6, p2[0] / 3 - p3[0] / 6 - p1[0] / 6, 0);
			sum += df1 * (uf1 | nAf1) + df2 * (uf2 | nAf2);
			sum_area += ((p2 - p1) % (p3 - p1))[2] / 6;
		}
		if (sum_area == 0)
			cout << "Impossible case" << endl;
		float delta_d = -m_dt * sum / sum_area;
		m_mesh.property(m_float_temp, *v_it) = delta_d;
		//if (m_mesh.point(*v_it)[2] == 30)
		//	cout << m_mesh.point(*v_it) << "   \t" << delta_d << '\t' << m_mesh.property(m_depth, *v_it) << '\t' << index->to_nature_coord(m_mesh.property(m_velocity, *v_it)) << endl;
	}
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		float d = m_mesh.property(m_depth, *v_it);
		if (out_of_S1(*v_it))
			continue;
		float b = m_mesh.property(m_bottom, *v_it);
		float delta_d = m_mesh.property(m_float_temp, *v_it);
		d += delta_d;
		if (m_mesh.property(m_boundary, *v_it))
			m_mesh.property(m_boundary, *v_it)->apply_depth(d);
		if (d > 0 && d <= m_depth_threshold)
			d = 0;
		m_mesh.property(m_depth, *v_it) = d;
	}
#endif
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

void Simulator::prepareData ()
{
	int vnum = (int)m_mesh.n_vertices();

	// malloc
	c_vel = new float3[vnum];
	c_mvel = new float3[vnum];
	c_point = new float3[vnum];
	c_value0 = new float3[vnum];
	c_bottom = new float[vnum];
	c_depth = new float[vnum];
	c_height = new float[vnum];
	c_boundary = new int[vnum];

	// pull#1:增加
	c_once_have_water = new bool[vnum];
	c_on_water_boundary = new bool[vnum];
	c_extrapolate_depth = new float[vnum];
	c_tensor = new float4[vnum];
	c_wind_velocity = new float3[vnum];
	c_depth_boundary_type = new int[vnum];
	c_depth_boundary_value = new float[vnum];

	c_vertex = new MyVertex[vnum];
	c_vertex_rot = new float3[vnum][3];
	c_vertex_oneRing = new int[vnum][MAX_VERTEX];
	c_vertex_nearVert = new int[vnum][MAX_NEAR_V];
	c_vertex_nerbFace = new int3[vnum][MAX_FACES];
	c_vertex_planeMap = new float3[vnum][MAX_FACES * 3];
	c_vertex_opph = new VertexOppositeHalfedge[vnum][MAX_VERTEX];
	
	// initialize
	for (int i = 0; i < vnum; i++) {
		for (int j = 0; j < MAX_VERTEX; j++) {
			c_vertex_oneRing[i][j] = -1;
			c_vertex_opph[i][j].is_valid = false;
		}
		//for (int j = 0; j < MAX_GRID; j++)
		//for (int k = 0; k < MAX_FACE; k++)
		//{
			////c_vertex[i].nerbFaceTest[j*MAX_FACE + k].x = c_vertex[i].nerbFaceTest[j*MAX_FACE + k].y = c_vertex[i].nerbFaceTest[j*MAX_FACE + k].z = -1;
			//c_vertex[i].nerbFace[j][k].x = c_vertex[i].nerbFace[j][k].y = c_vertex[i].nerbFace[j][k].z = -1;
		//}
		for (int j = 0; j < MAX_FACES; j++)
		{
			c_vertex_nerbFace[i][j].x = c_vertex_nerbFace[i][j].y = c_vertex_nerbFace[i][j].z = -1;
		}
		for (int j = 0; j < MAX_NEAR_V; j++) {
			c_vertex_nearVert[i][j] = -1;
		}
	}
	
	//copy
	int idx = 0;
	for (auto v_it = m_mesh.vertices_begin (); v_it != m_mesh.vertices_end (); ++v_it, ++idx) {
		MyMesh::Point tmp;
		tmp = m_mesh.property ( m_velocity, *v_it );
		c_vel[idx].x = tmp[0]; c_vel[idx].y = tmp[1]; c_vel[idx].z = tmp[2];
		tmp = m_mesh.property ( m_midvel, *v_it );
		c_mvel[idx].x = tmp[0]; c_mvel[idx].y = tmp[1]; c_mvel[idx].z = tmp[2];
		tmp = m_mesh.point ( *v_it );
		c_point[idx].x = tmp[0]; c_point[idx].y = tmp[1]; c_point[idx].z = tmp[2];
		c_bottom[idx] = m_mesh.property ( m_bottom, *v_it );
		c_depth[idx] = m_mesh.property ( m_depth, *v_it );
		
		if (m_mesh.is_boundary ( *v_it )) {
			switch (m_mesh.property ( m_boundary, *v_it )->vtype) {
			case BoundaryCondition::VEL_BOUND: c_boundary[idx] = 2; break;
			case BoundaryCondition::VEL_FIXED: c_boundary[idx] = 3; break;
			case BoundaryCondition::VEL_NOACTION:default: c_boundary[idx] = 1; break;
			}
			tmp = m_mesh.property(m_boundary, *v_it)->vvalue0;
			switch (m_mesh.property(m_boundary, *v_it)->dtype) {
			case BoundaryCondition::DEP_FIXED: c_depth_boundary_type[idx] = 2; break;
			case BoundaryCondition::DEP_NOACTION: default: c_depth_boundary_type[idx] = 1; break;
			}
			c_depth_boundary_value[idx] = m_mesh.property(m_boundary, *v_it)->dvalue0;
		}
		else {
			c_boundary[idx] = 0;
			tmp[0] = tmp[1] = tmp[2] = 0;
			c_depth_boundary_type[idx] = 0;
			c_depth_boundary_value[idx] = 0;
		}
		
		c_value0[idx].x = tmp[0]; c_value0[idx].y = tmp[1]; c_value0[idx].z = tmp[2];
		
		
		// myVertex
		// pull#1: 约定1-ring的顺序
		IndexOnVertex *index = m_mesh.data(*v_it).index;
		int k = 0;
		std::vector<MyMesh::VertexHandle> const &ring_vh(index->get_ring_1_ordered());
		for (auto vv_it = ring_vh.begin(); vv_it != ring_vh.end(); ++vv_it) {
			c_vertex_oneRing[idx][k++] = vv_it->idx();
			if (k >= MAX_VERTEX) printf("%d error1\n", k);
		}
		std::queue<MyMesh::HalfedgeHandle> que_hh;
		for (auto voh_it = m_mesh.cvoh_ccwiter(*v_it); voh_it.is_valid(); ++voh_it)
			que_hh.push(*voh_it);
		if (m_mesh.is_boundary(*v_it)) {
			while (!m_mesh.is_boundary(que_hh.front())) {
				que_hh.push(que_hh.front());
				que_hh.pop();
			}
		}
		k = 0;
		while (!que_hh.empty()) { //edge
			auto hh = que_hh.front();
			que_hh.pop();
			VertexOppositeHalfedge &opph(c_vertex_opph[idx][k]);
			hh = m_mesh.next_halfedge_handle(hh);
			opph.is_valid = true;
			opph.is_boundary = m_mesh.is_boundary(hh);
			opph.from_v = m_mesh.from_vertex_handle(hh).idx();
			opph.to_v = m_mesh.to_vertex_handle(hh).idx();
			hh = m_mesh.opposite_halfedge_handle(hh);
			opph.opph_is_boundary = m_mesh.is_boundary(hh);
			opph.opph_oppv = m_mesh.to_vertex_handle(m_mesh.next_halfedge_handle(hh)).idx();
			k++;
			if (k >= MAX_VERTEX) printf("%d error1\n", k);
		}

		c_vertex[idx].x0 = index->x0;
		c_vertex[idx].y0 = index->y0;
		c_vertex[idx].dx = index->dx;
		c_vertex[idx].dy = index->dy;
		c_vertex[idx].nx = index->nx;
		c_vertex[idx].ny = index->ny;
		c_vertex_rot[idx][0].x = index->rot[0][0]; c_vertex_rot[idx][0].y = index->rot[0][1]; c_vertex_rot[idx][0].z = index->rot[0][2];
		c_vertex_rot[idx][1].x = index->rot[1][0]; c_vertex_rot[idx][1].y = index->rot[1][1]; c_vertex_rot[idx][1].z = index->rot[1][2];
		c_vertex_rot[idx][2].x = index->rot[2][0]; c_vertex_rot[idx][2].y = index->rot[2][1]; c_vertex_rot[idx][2].z = index->rot[2][2];

		k = 0;
		for (auto it = index->vec_vh.begin (); it != index->vec_vh.end (); ++it) {
			c_vertex_nearVert[idx][k++] = it->idx();
			if (k >= MAX_NEAR_V) printf ( "%d error3\n", k );
		}

		std::set<MyMesh::FaceHandle> set_tmp;
		set_tmp.clear();
		int count = 0;
		for (int i = 0; i < index->contain.size (); ++i) {
			std::vector<OpenMesh::FaceHandle> const &contain = index->contain[i];
			//int j = 0;
			for (auto f_it = contain.begin (); f_it != contain.end (); ++f_it) {
				auto fv_it = m_mesh.cfv_begin ( *f_it );
				int3 tmp_face;
				float3 tmp_point[3];
				MyMesh::Point a, b, c, aa, bb, cc;
				
				tmp_face.x = fv_it->idx();
				aa = index->plane_map(m_mesh.point(*fv_it));
				tmp_point[0].x = aa[0]; tmp_point[0].y = aa[1]; tmp_point[0].z = aa[2];
				++fv_it;

				tmp_face.y = fv_it->idx();
				bb = index->plane_map(m_mesh.point(*fv_it));
				tmp_point[1].x = bb[0]; tmp_point[1].y = bb[1]; tmp_point[1].z = bb[2];
				++fv_it;

				tmp_face.z = fv_it->idx();
				cc = index->plane_map(m_mesh.point(*fv_it));
				tmp_point[2].x = cc[0]; tmp_point[2].y = cc[1]; tmp_point[2].z = cc[2];

				if (set_tmp.find(*f_it) == set_tmp.end())
				{
					set_tmp.insert(*f_it);
					c_vertex_nerbFace[idx][count] = tmp_face;
					c_vertex_planeMap[idx][count * 3] = tmp_point[0];
					c_vertex_planeMap[idx][count * 3 + 1] = tmp_point[1];
					c_vertex_planeMap[idx][count * 3 + 2] = tmp_point[2];
					count++;
				}

				//if (i > MAX_GRID || j >= MAX_FACE) printf ( "%d %d --> error2\n", i, j );
				if (count > MAX_FACES)printf("%d --> error2\n", count);
			}
		}
		// pull#1:增加
		c_once_have_water[idx] = false;
		Tensor22 tensor = m_mesh.property(m_tensor, *v_it);
		c_tensor[idx].x = tensor[0];
		c_tensor[idx].y = tensor[1];
		c_tensor[idx].z = tensor[2];
		c_tensor[idx].w = tensor[3];
		MyMesh::Point wind_vel = m_mesh.property(m_wind_velocity, *v_it);
		c_wind_velocity[idx].x = wind_vel[0];
		c_wind_velocity[idx].y = wind_vel[1];
		c_wind_velocity[idx].z = wind_vel[2];
	}
}

void Simulator::post_data ()
{
	int idx = 0;
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it, ++idx) {
		m_mesh.property(m_depth, *v_it) = c_depth[idx];
	}
}
}
