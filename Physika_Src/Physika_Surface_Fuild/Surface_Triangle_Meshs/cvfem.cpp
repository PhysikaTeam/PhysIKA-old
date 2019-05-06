#include "cvfem.h"
#include "indexonvertex.h"
#include <vector>
#include <iostream>

using namespace std;
namespace Physika{
template<typename T>
static T _point_interpolation ( MyMesh const &mesh, OpenMesh::VPropHandleT<T> prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh ) {
	// P是“拍平”平面上的点，z必须为0
	IndexOnVertex *pindex = mesh.data ( vh ).index;
	auto fv_it = mesh.cfv_begin ( fh );
	MyMesh::Point a = pindex->plane_map ( mesh.point ( *fv_it ) );
	T fa = mesh.property ( prop, *fv_it );
	fa = IndexOnVertex::index_conv ( mesh.data ( *fv_it ).index, mesh.data ( vh ).index, fa );
	fv_it++;
	MyMesh::Point b = pindex->plane_map ( mesh.point ( *fv_it ) );
	T fb = mesh.property ( prop, *fv_it );
	fb = IndexOnVertex::index_conv ( mesh.data ( *fv_it ).index, mesh.data ( vh ).index, fb );
	fv_it++;
	MyMesh::Point c = pindex->plane_map ( mesh.point ( *fv_it ) );
	T fc = mesh.property ( prop, *fv_it );
	fc = IndexOnVertex::index_conv ( mesh.data ( *fv_it ).index, mesh.data ( vh ).index, fc );
	auto triangle_area = []( MyMesh::Point const &a, MyMesh::Point const &b ) {
		return a[0] * b[1] - a[1] * b[0];
	};
	MyMesh::Point pa = a - p;
	MyMesh::Point pb = b - p;
	MyMesh::Point pc = c - p;
	MyMesh::Point ab = b - a;
	MyMesh::Point ac = c - a;
	auto abc = triangle_area ( ab, ac );
	auto pab = triangle_area ( pa, pb );
	auto pbc = triangle_area ( pb, pc );
	auto pca = triangle_area ( pc, pa );
	//extern bool inter_debug;
	//if (inter_debug) {
	//	auto ret = ((fa * pbc + fb * pca + fc * pab) / abc);
	//	cout << "==inter==" << endl;
	//	cout << fa << '\t' << mesh.data(vh).index->to_nature_coord(fa) << endl;
	//	cout << fb << '\t' << mesh.data(vh).index->to_nature_coord(fb) << endl;
	//	cout << fc << '\t' << mesh.data(vh).index->to_nature_coord(fc) << endl;
	//	cout << ret << '\t' << mesh.data(vh).index->to_nature_coord(ret) << endl;
	//	cout << abc << '\t' << pab << '\t' << pbc << '\t' << pca << endl;
	//}
	return (fa * pbc + fb * pca + fc * pab) / abc;
}

template<typename T>
static MyMesh::Point _gradient ( MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c, T fa, T fb, T fc ) {
	MyMesh::Point ab = b - a;
	MyMesh::Point ac = c - a;
	T abc2 = ab[0] * ac[1] - ab[1] * ac[0];
	T dx = (b[1] - c[1]) * fa + (c[1] - a[1]) * fb + (a[1] - b[1]) * fc;
	T dy = (c[0] - b[0]) * fa + (a[0] - c[0]) * fb + (b[0] - a[0]) * fc;
	return MyMesh::Point ( dx / abc2, dy / abc2, 0 );
}

float point_interpolation ( MyMesh const &mesh, VFloatPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh ) {
	return _point_interpolation ( mesh, prop, p, vh, fh );
}

MyMesh::Point point_interpolation ( MyMesh const &mesh, VVectorPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh ) {
	return _point_interpolation ( mesh, prop, p, vh, fh );
}

MyMesh::Point point_gradient ( MyMesh const &mesh, VFloatPropHandle prop, MyMesh::Point p, MyMesh::VertexHandle vh, MyMesh::FaceHandle fh ) {
	IndexOnVertex *pindex = mesh.data ( vh ).index;
	auto fv_it = mesh.cfv_begin ( fh );
	MyMesh::Point a = pindex->plane_map ( mesh.point ( *fv_it ) );
	auto fa = mesh.property ( prop, *fv_it );
	fa = IndexOnVertex::index_conv ( mesh.data ( *fv_it ).index, mesh.data ( vh ).index, fa );
	fv_it++;
	MyMesh::Point b = pindex->plane_map ( mesh.point ( *fv_it ) );
	auto fb = mesh.property ( prop, *fv_it );
	fb = IndexOnVertex::index_conv ( mesh.data ( *fv_it ).index, mesh.data ( vh ).index, fb );
	fv_it++;
	MyMesh::Point c = pindex->plane_map ( mesh.point ( *fv_it ) );
	auto fc = mesh.property ( prop, *fv_it );
	fc = IndexOnVertex::index_conv ( mesh.data ( *fv_it ).index, mesh.data ( vh ).index, fc );
	return _gradient ( a, b, c, fa, fb, fc );
}

MyMesh::Point vertex_gradient ( MyMesh const &mesh, VFloatPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center ) {
	IndexOnVertex *pindex = mesh.data ( vh_center ).index;
	MyMesh::Point p_target = pindex->plane_map ( mesh.point ( vh_target ) );
	auto f_target = mesh.property ( prop, vh_target );
	static vector<MyMesh::Point> vec_point;
	static vector<float> vec_value;
	vec_point.clear ();
	vec_value.clear ();
	for (auto vv_it = mesh.cvv_iter ( vh_target ); vv_it.is_valid (); ++vv_it) {
		vec_point.push_back ( pindex->plane_map ( mesh.point ( *vv_it ) ) );
		vec_value.push_back ( mesh.property ( prop, *vv_it ) );
	}
	float sum_area = 0;
	MyMesh::Point sum_grad = MyMesh::Point ( 0, 0, 0 );
	for (size_t i = 0; i < vec_point.size (); i++) {
		size_t j = i + 1;
		if (j == vec_point.size ())
			j = 0;
		static auto const _gradient_by_area = []( MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c, float fa, float fb, float fc ) {
			auto dx = (b[1] - c[1]) * fa + (c[1] - a[1]) * fb + (a[1] - b[1]) * fc;
			auto dy = (c[0] - b[0]) * fa + (a[0] - c[0]) * fb + (b[0] - a[0]) * fc;
			return MyMesh::Point ( dx / 2, dy / 2, 0 );
		};
		MyMesh::Point face_grad = _gradient_by_area ( p_target, vec_point[j], vec_point[i], f_target, vec_value[j], vec_value[i] );
		auto area = ((vec_point[j] - p_target) % (vec_point[i] - p_target))[2] / 2;
		sum_area += area;
		sum_grad += face_grad;
		if (vec_point.size () == 2) break;
	}
	return sum_grad / sum_area;
}

float vertex_divergence ( MyMesh const &mesh, VVectorPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center ) {
	IndexOnVertex *pindex = mesh.data ( vh_center ).index;
	MyMesh::Point p_target = pindex->plane_map ( mesh.point ( vh_target ) );
	MyMesh::Point f_target = mesh.property ( prop, vh_target );
	static vector<MyMesh::Point> vec_point;
	static vector<MyMesh::Point> vec_value;
	vec_point.clear ();
	vec_value.clear ();
	for (auto vv_it = mesh.cvv_iter ( vh_target ); vv_it.is_valid (); ++vv_it) {
		vec_point.push_back ( pindex->plane_map ( mesh.point ( *vv_it ) ) );
		vec_value.push_back ( IndexOnVertex::index_conv ( mesh.data ( *vv_it ).index, pindex, mesh.property ( prop, *vv_it ) ) );
	}
	float sum_area = 0;
	float sum_div = 0;
	for (size_t i = 0; i < vec_point.size (); i++) {
		size_t j = i + 1;
		if (j == vec_point.size ())
			j = 0;
		static auto const _divergence_by_area = []( MyMesh::Point const &a, MyMesh::Point const &b, MyMesh::Point const &c, MyMesh::Point fa, MyMesh::Point fb, MyMesh::Point fc ) {
			auto dx = (b[1] - c[1]) * fa[0] + (c[1] - a[1]) * fb[0] + (a[1] - b[1]) * fc[0];
			auto dy = (c[0] - b[0]) * fa[1] + (a[0] - c[0]) * fb[1] + (b[0] - a[0]) * fc[1];
			return dx / 2 + dy / 2;
		};
		float face_div = _divergence_by_area ( p_target, vec_point[j], vec_point[i], f_target, vec_value[j], vec_value[i] );
		auto area = ((vec_point[j] - p_target) % (vec_point[i] - p_target))[2] / 2;
		sum_area += area;
		sum_div += face_div;
		if (vec_point.size () == 2) break;
	}
	return sum_div / sum_area;
}

float vertex_laplace ( MyMesh const &mesh, VFloatPropHandle prop, MyMesh::VertexHandle vh_target, MyMesh::VertexHandle vh_center ) {
	IndexOnVertex *pindex = mesh.data ( vh_center ).index;
	MyMesh::Point p_target = pindex->plane_map ( mesh.point ( vh_target ) );
	auto f_target = mesh.property ( prop, vh_target );
	static vector<MyMesh::Point> vec_point;
	static vector<float> vec_value;
	vec_point.clear ();
	vec_value.clear ();
	for (auto vv_it = mesh.cvv_iter ( vh_target ); vv_it.is_valid (); ++vv_it) {
		vec_point.push_back ( pindex->plane_map ( mesh.point ( *vv_it ) ) );
		vec_value.push_back ( IndexOnVertex::index_conv ( mesh.data ( *vv_it ).index, pindex, mesh.property ( prop, *vv_it ) ) );
	}
	float sum_area = 0;
	float sum_mult = 0;
	auto cot = []( MyMesh::Point const &a, MyMesh::Point const &o, MyMesh::Point const &b ) {
		MyMesh::Point oa = a - o;
		MyMesh::Point ob = b - o;
		auto s = (ob % oa)[2];
		auto c = ob | oa;
		return c / s;
	};
	for (size_t i = 0; i < vec_point.size (); i++) {
		size_t a = i + 1;
		if (a == vec_point.size ())
			a = 0;
		size_t b = (i == 0) ? vec_point.size () - 1 : i - 1;
		auto cota = cot ( p_target, vec_point[a], vec_point[i] );
		auto cotb = cot ( vec_point[i], vec_point[b], p_target );
		auto area = ((vec_point[a] - p_target) % (vec_point[i] - p_target))[2];
		if (area != 0) {
			sum_mult += (cota + cotb) * (vec_value[i] - f_target);
			sum_area += area;
		}
		if (vec_point.size () == 2) break;
	}
	return sum_mult / (sum_area / 3);
}

// Surface tension utility functions
MyMesh::Point computeNormali(std::vector<MyMesh::Point> const &points, MyMesh::Point const &ipoint, bool boundaryFlag) {
	MyMesh::Point tempresult(0, 0, 0);
	size_t current, next;
	float totalarea = 0.0f;

	for (size_t i = boundaryFlag ? 1 : 0; i < points.size(); i++) {
		current = i;
		next = (i == points.size() - 1) ? 0 : i + 1;

		MyMesh::Point b = ipoint - points[next];
		MyMesh::Point c = ipoint - points[current];

		MyMesh::Point area = c % b; // cross(c, b);
		float maga = area.norm(); // mag(area);

		float asintemp = maga / (b.norm() * c.norm());
		if (asintemp > 1.0f) asintemp = 1.0f;
		if (asintemp < -1.0f) asintemp = -1.0f;

		float angle = asin(asintemp);
		if ((b | c) < 0)
			angle = (float)M_PI - angle;

		tempresult += area * (angle / maga); /// weight by angle
		// tempresult += area / (maga * maga); /// weight by inverse area
	}

	return tempresult.normalized();
}
float areaGradP(std::vector<MyMesh::Point> const &points, MyMesh::Point const &ipoint, MyMesh::Point const &normali, bool boundaryFlag) {
	MyMesh::Point tempresult(0, 0, 0);
	size_t current, next;
	float totalarea = 0.0f;

	for (size_t i = boundaryFlag ? 1 : 0; i < points.size(); i++) {
		current = i;
		next = (i == points.size() - 1) ? 0 : i + 1;

		MyMesh::Point a = points[next] - points[current];
		MyMesh::Point b = ipoint - points[next];
		MyMesh::Point c = ipoint - points[current];

		MyMesh::Point area = c % b; // cross(c, b);
		MyMesh::Point norm = area.normalized();// normalized(area);

		float cosxy = norm[2];// dot(norm, Vec3f(0,0,1.0f));
		float cosyz = norm[0];// dot(norm, Vec3f(1.0f,0,0));
		float coszx = norm[1];// dot(norm, Vec3f(0,1.0f,0));

		float par_x = 0.5f * (-a[1]) * cosxy + 0.5f * (a[2]) * coszx;
		float par_y = 0.5f * (-a[2]) * cosyz + 0.5f * (a[0]) * cosxy;
		float par_z = 0.5f * (-a[0]) * coszx + 0.5f * (a[1]) * cosyz;

		tempresult += MyMesh::Point(par_x, par_y, par_z);
		totalarea += 0.5f * area.norm() / 3 * abs(dot(normali, norm));
	}

	if (boundaryFlag == false) {
		int sign = dot(tempresult, normali) >= 0.0f ? 1 : -1;
		return tempresult.norm() / totalarea * sign;
	}
	else { // on boundary
		return dot(tempresult, normali) / totalarea;
	}
}
}
/////////////////////////////// Surface tension utility functions
