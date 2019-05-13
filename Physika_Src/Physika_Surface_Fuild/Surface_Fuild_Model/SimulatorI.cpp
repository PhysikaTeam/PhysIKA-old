#include "swe.cuh"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
namespace Physika{
void SimulatorI::init(int argc, char** argv) {
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
void SimulatorI::run_cuda ( int frame )
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
void SimulatorI::output_obj_cuda(int frame) {
	CoordSystem g_system = gen_coord_system_by_z_axis(m_g);
	CoordSystem screen_system = gen_coord_system_by_z_axis(MyMesh::Point(0, -1, 0));
	auto rotate_by_g = [&](MyMesh::Point p) {
			MyMesh::Point delta(p - m_rotate_center);
			MyMesh::Point coord(delta | g_system[0], delta | g_system[1], delta | g_system[2]);
			coord = IndexOnVertex::coord_conv(g_system.data(), screen_system.data(), coord);
			delta = coord[0] * screen_system[0] + coord[1] * screen_system[1] + coord[2] * screen_system[2];
			return delta + m_rotate_center;
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
void SimulatorI::run() {
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
}
