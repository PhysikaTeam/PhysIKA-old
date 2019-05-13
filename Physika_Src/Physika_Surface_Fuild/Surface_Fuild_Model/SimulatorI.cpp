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
void SimulatorI::set_initial_constants() {
	
}
void SimulatorI::generate_origin() {
	m_origin.clear();
	
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
void SimulatorI::generate_mesh(){
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
}
void SimulatorI::add_properties() {
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
void SimulatorI::calculate_tensor() {
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
void SimulatorI::set_initial_conditions() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		MyMesh::Point p(m_mesh.point(*v_it));
		m_mesh.property(m_depth, *v_it) = (p[0] < 10.0f && p[2] < 44.0f && p[2] > 20.0f) ? 1.5f : 0.0f;
		m_mesh.property(m_velocity, *v_it) = MyMesh::Point(0, 0, 0);
		if (m_mesh.is_boundary(*v_it)) {
			m_mesh.property(m_boundary, *v_it)->set_depth(BoundaryCondition::DEP_FIXED, this, *v_it);
			m_mesh.property(m_boundary, *v_it)->set_velocity(BoundaryCondition::VEL_NOACTION, this, *v_it);
		}
		}
}
void SimulatorI::output_obj() {
	output_obj_cuda(m_stepnum);
}
void SimulatorI::advect_filed_values() {
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
				if (m_mesh.is_boundary(vh)) {
					auto bound = m_mesh.property(m_boundary, vh);
					bound->apply_depth(depth_new);
					bound->apply_velocity(velocity_new);
				}
				velocity_new = IndexOnVertex::index_conv(m_mesh.data(vh).index, index, velocity_new);
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
	swap(m_depth, m_float_temp);
	swap(m_velocity, m_vector_temp);
}
void SimulatorI::update_depth() {
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
		if (d > 2.0f)
			d = 1.8f;
		m_mesh.property(m_depth, *v_it) = d;
	}
}
void SimulatorI::release_index() {
	m_mesh.release_face_normals();
	m_mesh.release_vertex_normals();
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it) {
		delete m_mesh.data(*v_it).index;
	}
}
void SimulatorI::release_properties() {
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
void SimulatorI::prepareData ()
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
void SimulatorI::post_data ()
{
	int idx = 0;
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it, ++idx) {
		m_mesh.property(m_depth, *v_it) = c_depth[idx];
	}
}
}
