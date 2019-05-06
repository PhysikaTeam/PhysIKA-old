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
using namespace std;
namespace Physika{
Simulator::Simulator() { }
Simulator::~Simulator() { }
void Simulator::clear() {
	release_index();
	release_properties();
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
