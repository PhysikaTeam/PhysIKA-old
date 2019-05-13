#include "SimulatorBase.h"
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
namespace Physika{
SimulationBase::SimulatorBase(){}
SimulationBase::~SimulatorBase(){}
virtual void SimulationBase::init(int argc, char** argv){}
virtual void SimulationBase::run(){}
virtual void SimulationBase::run_cuda (int frame){}
virtual void SimulationBase::output_obj_cuda(int frame){}
virtual void SimulationBase::post_data (){}
virtual void SimulationBase::clear(){}
virtual void SimulationBase::set_initial_constants(){}
virtual void SimulationBase::generate_origin(){}
virtual void SimulationBase::generate_mesh(){}
virtual void SimulationBase::add_properties(){}
void SimulationBase::add_index_to_vertex(int ring){
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
void SimulationBase::match_bottom_height(){
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
virtual void SimulationBase::calculate_tensor(){}
virtual void SimulationBase::set_initial_conditions(){}
virtual void SimulationBase::output_obj(){}
virtual void SimulationBase::output_wind(){}
virtual void SimulationBase::edit_mesh(){}
virtual void SimulationBase::edit_mesh_update_normal(){}
virtual void SimulationBase::edit_mesh_update_index(){}
virtual void SimulationBase::edit_mesh_update_tensor(){}
virtual void SimulationBase::advect_filed_values(){}
virtual void SimulationBase::update_depth(){}
virtual void SimulationBase::release_index(){}
virtual void SimulationBase::release_properties(){}
void SimulationBase::update_midvels() {
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
void SimulationBase::extrapolate_depth() {
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
void SimulationBase::force_boundary_depth() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it)
	if (m_mesh.is_boundary(*v_it)) {
		BoundaryCondition *bc = m_mesh.property(m_boundary, *v_it);
		bc->apply_depth(m_mesh.property(m_depth, *v_it));
	}
}
void SimulationBase::calculate_pressure() {
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
void SimulationBase::update_velocity() {
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
void SimulationBase::force_boundary_velocity() {
	for (auto v_it = m_mesh.vertices_begin(); v_it != m_mesh.vertices_end(); ++v_it)
	if (m_mesh.is_boundary(*v_it))
		m_mesh.property(m_boundary, *v_it)->apply_velocity(m_mesh.property(m_velocity, *v_it));
}
void SimulationBase::velocity_fast_march() {
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

SimulationBase::BoundaryCondition::BoundaryCondition()
:dtype(DEP_NOACTION), vtype(VEL_NOACTION) { }

void SimulationBase::BoundaryCondition::set_depth(DepthType type, Simulator *sim, MyMesh::VertexHandle vh) {
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

void SimulationBase::BoundaryCondition::set_velocity(VelocityType type, Simulator *sim, MyMesh::VertexHandle vh) {
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

void SimulationBase::BoundaryCondition::apply_depth(float &depth) {
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

void Simulation::BoundaryCondition::apply_velocity(MyMesh::Point &velocity) {
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
