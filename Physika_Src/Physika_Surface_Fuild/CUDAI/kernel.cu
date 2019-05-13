#include "kernel.cuh"
#include "Physika_Surface_Fuild/Surface_Utilities/cutil_math.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/indexonvertex.h"
#include "Physika_Surface_Fuild/Surface_Triangle_Meshs/cvfem.h"

#define COMPUTE_DIV_METHOD 2

__constant__ FluidParamsSWE		swSimData;

#ifdef HYBRID


//extern FluidParams		fcuda;		// CPU Fluid params
//extern FluidParams*	mcuda;		// GPU Fluid params
//
//extern bufList			fbuf;		// GPU Particle buffers
__device__	 int		swAddsize;

#endif

__device__ bool on_face_2d ( float3 p, float3 a, float3 b, float3 c )
{
	float3 pa = a - p;
	float3 pb = b - p;
	float3 pc = c - p;
	float3 ab = b - a;
	float3 ac = c - a;
	float abc = ab.x*ac.y - ab.y*ac.x;
	float pab = pa.x*pb.y - pa.y*pb.x;
	float pbc = pb.x*pc.y - pb.y*pc.x;
	float pca = pc.x*pa.y - pc.y*pa.x;
	if (abc < 0) {
		abc = -abc;
		pab = -pab;
		pbc = -pbc;
		pca = -pca;
	}
	float eps = -abc * 1e-3f;
	return (pab > eps && pbc > eps && pca > eps);
}

__device__ float3 index_conv (int index_from, int index_to, float3 vec_from, bufListSWE buf)
{
	float3 coord_from[3], coord_to[3];
	coord_from[0] = buf.m_vertex_rot[index_from][0]; coord_from[1] = buf.m_vertex_rot[index_from][1]; coord_from[2] = buf.m_vertex_rot[index_from][2];
	coord_to[0] = buf.m_vertex_rot[index_to][0]; coord_to[1] = buf.m_vertex_rot[index_to][1]; coord_to[2] = buf.m_vertex_rot[index_to][2];

	float3 axis_z = cross(coord_from[2], coord_to[2]);
	float axis_z_norm = length ( axis_z );
	if (axis_z_norm < 2e-4f) {
		float3 vec = vec_from.x*coord_from[0] + vec_from.y*coord_from[1] + vec_from.z*coord_from[2];
		return make_float3 ( dot ( vec, coord_to[0] ), dot ( vec, coord_to[1] ), dot ( vec, coord_to[2] ) );
	}
	else {
		axis_z /= axis_z_norm;
		float3 axis_from_x = cross ( coord_from[2], axis_z );
		float3 axis_to_x = cross ( coord_to[2], axis_z );
		float3 vec = vec_from.x*coord_from[0] + vec_from.y*coord_from[1] + vec_from.z*coord_from[2];
		vec = make_float3 ( dot ( vec, axis_from_x ), dot ( vec, coord_from[2] ), dot ( vec, axis_z ) );
		vec = vec.x*axis_to_x + vec.y*coord_to[2] + vec.z*axis_z;
		return make_float3 ( dot ( vec, coord_to[0] ), dot ( vec, coord_to[1] ), dot ( vec, coord_to[2] ) );
	}
}

__device__ float3 point_interpolation3 (float3 p, float3 fa, float3 fb, float3 fc, float3 a, float3 b, float3 c, bufListSWE buf)
{
	float3 pa = a - p;
	float3 pb = b - p;
	float3 pc = c - p;
	float3 ab = b - a;
	float3 ac = c - a;
	float abc = ab.x*ac.y - ab.y*ac.x;
	float pab = pa.x*pb.y - pa.y*pb.x;
	float pbc = pb.x*pc.y - pb.y*pc.x;
	float pca = pc.x*pa.y - pc.y*pa.x;
	
	return (fa * pbc + fb * pca + fc * pab) / abc;
}

__device__ float3 point_interpolation3Test(float3 &p, float3 &fa, float3 &fb, float3 &fc, float3 &a, float3 &b, float3 &c)
{
	float3 pa = a - p;
	float3 pb = b - p;
	float3 pc = c - p;
	float3 ab = b - a;
	float3 ac = c - a;
	float abc = ab.x*ac.y - ab.y*ac.x;
	float pab = pa.x*pb.y - pa.y*pb.x;
	float pbc = pb.x*pc.y - pb.y*pc.x;
	float pca = pc.x*pa.y - pc.y*pa.x;

	return (fa * pbc + fb * pca + fc * pab) / abc;
}

__device__ float point_interpolation1 ( float3 p, float fa, float fb, float fc, float3 a, float3 b, float3 c, bufListSWE buf )
{
	float3 pa = a - p;
	float3 pb = b - p;
	float3 pc = c - p;
	float3 ab = b - a;
	float3 ac = c - a;
	float abc = ab.x*ac.y - ab.y*ac.x;
	float pab = pa.x*pb.y - pa.y*pb.x;
	float pbc = pb.x*pc.y - pb.y*pc.x;
	float pca = pc.x*pa.y - pc.y*pa.x;

	return (fa * pbc + fb * pca + fc * pab) / abc;
}

__device__ int nearest_vertex(float3 p, int i, bool flag, bufListSWE buf)
{
	int vh = -1;
	bool is_first = true;
	float min_sqrnorm;
	for (int k = 0; k < MAX_NEAR_V; k++) {
		int idx = buf.m_vertex_nearVert[i][k];
		if (idx < 0) break;
		if (!flag) { // for velocity fast march
			// HACK: 多计算一圈v
			if (buf.m_depth[idx] <= swSimData.m_depth_threshold && !buf.m_on_water_boundary[idx])
				continue;
		}
		float3 p_tmp = buf.m_point[idx];
		float3 a = p_tmp - buf.m_point[i];
		float a_norm = length(a);
		a = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
		float a_new_norm = length(a);
		a = (a_new_norm == 0) ? a : (a * (a_norm / a_new_norm));
		float new_sqrnorm = dot(a - p, a - p);
		if (is_first || new_sqrnorm < min_sqrnorm) {
			vh = idx;
			min_sqrnorm = new_sqrnorm;
			is_first = false;
		}
	}
	return vh;
}

__device__ float3 vertex_gradient3 (int i, float3 p_target, float f_target, bufListSWE buf)
{
	float3 vec_point[MAX_VERTEX] = { make_float3 ( 0, 0, 0 ) };
	float vec_value[MAX_VERTEX] = { 0 };

	int size = 0;
	for (int j = 0; j < MAX_VERTEX; j++) {
		int idx = buf.m_vertex_oneRing[i][j];
		if (idx < 0) break;
		float3 p_tmp = buf.m_point[idx]; // plane mapping
		float3 a = p_tmp - buf.m_point[i];
		float3 b = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
		float b_norm = length ( b );
		float3 p = (b_norm == 0 ? b : b*(length ( a ) / b_norm));

		vec_point[j] = p;
		vec_value[j] = buf.m_tmp[i*MAX_VERTEX+j]; // scalar
		size++;
	}

	float sum_area = 0;
	float3 sum_grad = make_float3 ( 0, 0, 0 );
	for (int j = 0; j < size; j++) {
		int k = j + 1;
		if (k == size) k = 0;

		float dx = (vec_point[k].y - vec_point[j].y)*f_target + (vec_point[j].y - p_target.y)*vec_value[k] + (p_target.y - vec_point[k].y)*vec_value[j];
		float dy = (vec_point[j].x - vec_point[k].x)*f_target + (p_target.x - vec_point[j].x)*vec_value[k] + (vec_point[k].x - p_target.x)*vec_value[j];
		float3 face_grad = make_float3 ( dx / 2.0, dy / 2.0, 0 );
		float area = (cross ( vec_point[k] - p_target, vec_point[j] - p_target ).z) / 2.0;
		sum_area += area;
		sum_grad += face_grad;
		if (size == 2) break;
	}

	return sum_grad / sum_area;
}

__device__ float vertex_divergence3 ( int i, float3 p_target, float3 f_target, bufListSWE buf )
{
	float3 vec_point[MAX_VERTEX] = { make_float3 ( 0, 0, 0 ) };
	float3 vec_value[MAX_VERTEX] = { make_float3 ( 0, 0, 0 ) };

	int size = 0;
	for (int j = 0; j < MAX_VERTEX; j++) {
		int idx = buf.m_vertex_oneRing[i][j];
		if (idx < 0) break; // continue
		float3 p_tmp = buf.m_point[idx]; // plane mapping
		float3 a = p_tmp - buf.m_point[i];
		float3 b = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
		float b_norm = length ( b );
		float3 p = (b_norm == 0 ? b : b*(length ( a ) / b_norm));

		float3 tmp = index_conv ( idx, i, buf.m_velocity[idx], buf );
		vec_point[j] = p;
		vec_value[j] = tmp;
		size++;
	}

	float sum_area = 0;
	float sum_div = 0;
	for (int j = 0; j < size; j++) {
		int k = j + 1;
		if (k == size) k = 0;

		float dx = (vec_point[k].y - vec_point[j].y)*f_target.x + (vec_point[j].y - p_target.y)*vec_value[k].x + (p_target.y - vec_point[k].y)*vec_value[j].x;
		float dy = (vec_point[j].x - vec_point[k].x)*f_target.y + (p_target.x - vec_point[j].x)*vec_value[k].y + (vec_point[k].x - p_target.x)*vec_value[j].y;
		float face_div = dx / 2.0 + dy / 2.0;
		float area = (cross ( vec_point[k] - p_target, vec_point[j] - p_target ).z) / 2.0;
		sum_area += area;
		sum_div += face_div;
		if (size == 2) break;
	}

	return sum_div / sum_area;
}

__device__ float3 computeNormali(float3 points[MAX_VERTEX], int nnum, float3 &ipoint, bool boundaryFlag) {
	float3 tempresult = make_float3(0, 0, 0);
	size_t current, next;
	float totalarea = 0.0f;

	for (size_t i = boundaryFlag ? 1 : 0; i < nnum; i++) {
		current = i;
		next = (i == nnum - 1) ? 0 : i + 1;

		float3 b = ipoint - points[next];
		float3 c = ipoint - points[current];

		float3 area = cross(c, b);
		float maga = length(area);

		float asintemp = maga / (length(b) * length(c));
		if (asintemp > 1.0f) asintemp = 1.0f;
		if (asintemp < -1.0f) asintemp = -1.0f;

		float angle = asin(asintemp);
		if (dot(b, c) < 0)
			angle = (float)M_PI - angle;

		tempresult += (area / maga) * angle; ///weight by angle
		//tempresult += area / (maga * maga); ///weight by inverse area
	}
	return normalize(tempresult);
}

__device__ float areaGradP(float3 points[MAX_VERTEX], int nnum, float3 ipoint, float3 normali, bool boundaryFlag)
{
	float3 tempresult = make_float3(0, 0, 0);
	size_t current, next;
	float totalarea = 0.0f;

	for (size_t i = boundaryFlag ? 1 : 0; i < nnum; i++) {
		current = i;
		next = (i == nnum - 1) ? 0 : i + 1;

		float3 a = points[next] - points[current];
		float3 b = ipoint - points[next];
		float3 c = ipoint - points[current];

		float3 area = cross(c, b);
		float3 norm = normalize(area);

		float cosxy = norm.z;
		float cosyz = norm.x;
		float coszx = norm.y;

		float par_x = 0.5f * (-a.y) * cosxy + 0.5f * (a.z) * coszx;
		float par_y = 0.5f * (-a.z) * cosyz + 0.5f * (a.x) * cosxy;
		float par_z = 0.5f * (-a.x) * coszx + 0.5f * (a.y) * cosyz;

		tempresult += make_float3(par_x, par_y, par_z);
		totalarea += 0.5 * length(area) / 3 * abs(dot(normali, norm));
	}

	if (boundaryFlag == false) {
		int sign = dot(tempresult, normali) >= 0.0f ? 1 : -1;
		return length(tempresult) / totalarea * sign;
	} else { // on boundary
		return dot(tempresult, normali) / totalarea;
	}
}

__global__ void update_mivels_kern ( bufListSWE buf, int vnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index				
	if (i >= vnum) return;
	
	float3 vel = buf.m_velocity[i];
	float3 midvel = make_float3(0, 0, 0);
	if (length(vel) == 0) {
		buf.m_midvel[i] = make_float3(0, 0, 0);
		return;
	}
	bool flag = false;
	float3 p = vel * swSimData.dt / -2.0;
	MyVertex m_v = buf.m_vertex[i];
	if (p.x < m_v.x0 || p.x >= m_v.x0 + m_v.nx * m_v.dx || p.y < m_v.y0 || p.y >= m_v.y0 + m_v.ny * m_v.dy) {
		midvel = vel;
	} else {
		for (int j = 0; j < MAX_FACES;j++){
			int3 tmp = buf.m_vertex_nerbFace[i][j];
			if (tmp.x < 0)
				break;
			if (on_face_2d(p, buf.m_vertex_planeMap[i][j * 3], buf.m_vertex_planeMap[i][j * 3 + 1], buf.m_vertex_planeMap[i][j * 3 + 2])) {
				float3 fa = index_conv(tmp.x, i, buf.m_velocity[tmp.x], buf);
				float3 fb = index_conv(tmp.y, i, buf.m_velocity[tmp.y], buf);
				float3 fc = index_conv(tmp.z, i, buf.m_velocity[tmp.z], buf);
				midvel = point_interpolation3(p, fa, fb, fc, buf.m_vertex_planeMap[i][j * 3], buf.m_vertex_planeMap[i][j * 3 + 1], buf.m_vertex_planeMap[i][j * 3 + 2], buf);
				flag = true;
				break;
			}
		}
		if (!flag)
			midvel = vel;
	}
	buf.m_midvel[i] = midvel;
}

__global__ void advect_field_values_kern ( bufListSWE buf, int vnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	float3 p = buf.m_midvel[i] * -swSimData.dt;
	MyVertex const &m_v = buf.m_vertex[i];
	float depth_new;
	float3 velocity_new;

	int3 fh = make_int3(-1, -1, -1);
	float3 planeMap[3];
	if (p.x < m_v.x0 || p.x >= m_v.x0 + m_v.nx * m_v.dx || p.y < m_v.y0 || p.y >= m_v.y0 + m_v.ny * m_v.dy) {
		// not found: fh = make_int3(-1, -1, -1);
	} else {
		for (int j = 0; j < MAX_FACES; j++){
			int3 tmp = buf.m_vertex_nerbFace[i][j];
			if (tmp.x < 0) break;
			if (on_face_2d(p, buf.m_vertex_planeMap[i][j * 3], buf.m_vertex_planeMap[i][j * 3 + 1], buf.m_vertex_planeMap[i][j * 3 + 2])) {
				fh = tmp;
				planeMap[0] = buf.m_vertex_planeMap[i][j * 3];
				planeMap[1] = buf.m_vertex_planeMap[i][j * 3 + 1];
				planeMap[2] = buf.m_vertex_planeMap[i][j * 3 + 2];
				break;
			}
		}
	}
	if (fh.x < 0) {
		int nearv = nearest_vertex(p, i, true, buf);
		depth_new = buf.m_depth[nearv];
		velocity_new = buf.m_velocity[nearv];
		velocity_new = index_conv(nearv, i, velocity_new, buf);
		if (buf.m_depth_boundary_type[nearv]) {
			switch (buf.m_depth_boundary_type[nearv]) {
			case 2: depth_new = buf.m_depth_boundary_value[nearv]; break;
			case 1:default: break;
			}
		}
		if (buf.m_boundary[nearv]) {
			switch (buf.m_boundary[nearv]) {
			case 2: velocity_new = dot(velocity_new, buf.m_value0[nearv]) * buf.m_value0[nearv]; break;
			case 3: velocity_new = buf.m_value0[nearv]; break;
			case 1:default: break;
			}
		}
	}
	else {
		float3 fa = index_conv(fh.x, i, buf.m_velocity[fh.x], buf);
		float3 fb = index_conv(fh.y, i, buf.m_velocity[fh.y], buf);
		float3 fc = index_conv(fh.z, i, buf.m_velocity[fh.z], buf);
		depth_new = point_interpolation1(p, buf.m_depth[fh.x], buf.m_depth[fh.y], buf.m_depth[fh.z], planeMap[0], planeMap[1], planeMap[2], buf);
		float face_depth_max = fmax(buf.m_depth[fh.x], fmax(buf.m_depth[fh.y], buf.m_depth[fh.z]));
		float face_depth_min = fmin(buf.m_depth[fh.x], fmin(buf.m_depth[fh.y], buf.m_depth[fh.z]));
		depth_new = fmin(fmax(depth_new, face_depth_min), face_depth_max);
		velocity_new = point_interpolation3(p, fa, fb, fc, planeMap[0], planeMap[1], planeMap[2], buf);
	}
	buf.m_float_tmp[i] = depth_new;
	buf.m_vector_tmp[i] = velocity_new;
}

__global__ void update_depth_velocity_kern(bufListSWE buf, int vnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	buf.m_depth[i] = buf.m_float_tmp[i];
	buf.m_velocity[i] = buf.m_vector_tmp[i];
}

__global__ void extrapolate_depth_kern(bufListSWE buf, int vnum) {
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index
	if (i >= vnum) return;

	buf.m_on_water_boundary[i] = false;
	float d = buf.m_depth[i];
	buf.m_float_tmp[i] = d;
	if (d <= swSimData.m_depth_threshold) {
		bool close_to_water = false;
		for (int j = 0; j < MAX_VERTEX; j++) {
			int idx = buf.m_vertex_oneRing[i][j];
			if (idx < 0) break;
			if (buf.m_depth[idx] > swSimData.m_depth_threshold) {
				close_to_water = true;
				break;
			}
		}
		buf.m_on_water_boundary[i] = close_to_water;
		if (close_to_water) {
			int cnt = 0;
			float ex_depth = 0;
			for (int j = 0; j < MAX_VERTEX; j++) {
				VertexOppositeHalfedge const &opph(buf.m_vertex_opph[i][j]);
				if (!opph.is_valid)
					break;
				if (opph.is_boundary)
					continue; // 不存在此三角形
				if (opph.opph_is_boundary)
					continue; // 此边不存在相对的面
				if (buf.m_depth[opph.from_v] <= swSimData.m_depth_threshold ||
					buf.m_depth[opph.to_v] <= swSimData.m_depth_threshold)
					continue; // 不是有水的面
				float3 o = buf.m_point[i];
				float3 a = buf.m_point[opph.from_v] - o;
				float3 b = buf.m_point[opph.to_v] - o;
				float3 c = buf.m_point[opph.opph_oppv] - o;
				float3 pa = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
				float3 pb = make_float3(dot(b, buf.m_vertex_rot[i][0]), dot(b, buf.m_vertex_rot[i][1]), 0);
				float3 pc = make_float3(dot(c, buf.m_vertex_rot[i][0]), dot(c, buf.m_vertex_rot[i][1]), 0);
				float pa_norm = length(pa);
				float pb_norm = length(pb);
				float pc_norm = length(pc);
				pa = (pa_norm == 0) ? pa : pa * (length(a) / pa_norm);
				pb = (pb_norm == 0) ? pb : pb * (length(b) / pb_norm);
				pc = (pc_norm == 0) ? pc : pc * (length(c) / pc_norm);
				float fa = buf.m_depth[opph.from_v];
				float fb = buf.m_depth[opph.to_v];
				float fc = buf.m_depth[opph.opph_oppv];
				float this_ex_depth = point_interpolation1(make_float3(0, 0, 0), fa, fb, fc, pa, pb, pc, buf);
				cnt++;
				ex_depth += this_ex_depth;
			}
			if (cnt == 0) {
				for (int j = 0; j < MAX_VERTEX; j++) {
					int vvh = buf.m_vertex_oneRing[i][j];
					if (vvh < 0)
						break;
					if (buf.m_depth[vvh] <= swSimData.m_depth_threshold)
						continue;
					for (int k = (buf.m_boundary[vvh] ? 1 : 0); k < MAX_VERTEX; k++) {
						int vk = buf.m_vertex_oneRing[vvh][k];
						if (vk < 0)
							break;
						if (buf.m_depth[vk] <= swSimData.m_depth_threshold)
							continue;
						int nk = k + 1;
						if (buf.m_vertex_oneRing[vvh][nk] < 0)
							nk = 0;
						int vnk = buf.m_vertex_oneRing[vvh][nk];
						if (buf.m_depth[vnk] <= swSimData.m_depth_threshold)
							continue;
						float3 o = buf.m_point[i];
						float3 a = buf.m_point[vvh] - o;
						float3 b = buf.m_point[vk] - o;
						float3 c = buf.m_point[vnk] - o;
						float3 pa = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
						float3 pb = make_float3(dot(b, buf.m_vertex_rot[i][0]), dot(b, buf.m_vertex_rot[i][1]), 0);
						float3 pc = make_float3(dot(c, buf.m_vertex_rot[i][0]), dot(c, buf.m_vertex_rot[i][1]), 0);
						float pa_norm = length(pa);
						float pb_norm = length(pb);
						float pc_norm = length(pc);
						pa = (pa_norm == 0) ? pa : pa * (length(a) / pa_norm);
						pb = (pb_norm == 0) ? pb : pb * (length(b) / pb_norm);
						pc = (pc_norm == 0) ? pc : pc * (length(c) / pc_norm);
						float fa = buf.m_depth[vvh];
						float fb = buf.m_depth[vk];
						float fc = buf.m_depth[vnk];
						float this_ex_depth = point_interpolation1(make_float3(0, 0, 0), fa, fb, fc, pa, pb, pc, buf);
						cnt++;
						ex_depth += this_ex_depth;
					}
				}
			}
			if (cnt == 0) {
				buf.m_float_tmp[i] = 0;
			} else {
				float extrapolation = ex_depth / cnt;
				if (extrapolation < 0)
					buf.m_float_tmp[i] = extrapolation;
				else
					buf.m_float_tmp[i] = 0;
			}
		} else {
			buf.m_float_tmp[i] = -1e-4f;
		}
	}
}

__global__ void force_boundary_depth_kern(bufListSWE buf, int vnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	if (buf.m_depth_boundary_type[i]) {
		switch (buf.m_depth_boundary_type[i]) {
		case 2: buf.m_depth[i] = buf.m_depth_boundary_value[i]; break;
		case 1:default: break;
		}
	}
}

__global__ void compute_pressure_kern(bufListSWE buf, int vnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
	if (i >= vnum) return;

	float b = buf.m_bottom[i];
	float d = buf.m_depth[i];
	if (d <= swSimData.m_depth_threshold) {
		float3 g = swSimData.m_g;
		float3 g_ind = make_float3(dot(g, buf.m_vertex_rot[i][0]), dot(g, buf.m_vertex_rot[i][1]), dot(g, buf.m_vertex_rot[i][2]));
		float pg = -g_ind.z * b;
		buf.m_pressure_gravity[i] = pg;
		if (buf.m_boundary[i]) {
			// 模型边界且是水的边界的点，从附近水的边界但非模型边界的点外插
			// 但也可能无法外插，需要初始化，以免得出不可预料的值
			buf.m_pressure_surface[i] = 0;
			return;
		}
		bool close_to_water = buf.m_on_water_boundary[i];
		if (close_to_water) {
			float coef_LL = swSimData.m_gamma * swSimData.m_water_boundary_tension_multiplier;
			float coef_SO = coef_LL * (1 + cosf(swSimData.m_water_boundary_theta)) / 2;
			float coef_LS = coef_LL * (1 - cosf(swSimData.m_water_boundary_theta)) / 2;
			float extrapolate_depth = buf.m_depth[i];
			float3 bottom_center = buf.m_point[i] + buf.m_vertex_rot[i][2] * b;
			float3 normal_face = buf.m_vertex_rot[i][2];
			float3 ex = cross(make_float3(0, 0, 1), normal_face);
			if (length(ex) < 0.1f)
				ex = cross(make_float3(0, 1, 0), normal_face);
			ex = normalize(ex);
			float3 ey = cross(normal_face, ex);
			struct VVertex {
				bool have_water;
				float3 bottom_point;
				float3 water_point;
			} ring[MAX_VERTEX];
			size_t ring_size = 0;
			for (int j = 0; j < MAX_VERTEX; j++) {
				int idx = buf.m_vertex_oneRing[i][j];
				if (idx < 0)
					break;
				VVertex &vertex(ring[j]);
				ring_size++;
				float b = buf.m_bottom[idx];
				float d = buf.m_depth[idx];
				vertex.have_water = (d > swSimData.m_depth_threshold);
				float3 bottom_point = buf.m_point[idx] + buf.m_vertex_rot[idx][2] * b;
				float3 planed_bottom_point = bottom_point - bottom_center;
				float len = length(planed_bottom_point);
				planed_bottom_point = normalize(make_float3(dot(planed_bottom_point, ex), dot(planed_bottom_point, ey), 0)) * len;
				vertex.bottom_point = planed_bottom_point;
				vertex.water_point = make_float3(vertex.bottom_point.x, vertex.bottom_point.y, buf.m_depth[idx]);
			}
			float3 n = make_float3(0, 0, 0);
			int num_water_boundary = 0;
			for (size_t i = 0; i < ring_size; i++) {
				size_t prev = (i == 0) ? ring_size - 1 : i - 1;
				size_t succ = (i == ring_size - 1) ? 0 : i + 1;
				if (!ring[prev].have_water && !ring[i].have_water && ring[succ].have_water) {
					n += normalize(make_float3(ring[i].bottom_point.y, -ring[i].bottom_point.x, 0));
					num_water_boundary++;
				} else if (ring[prev].have_water && !ring[i].have_water && !ring[succ].have_water) {
					n += normalize(make_float3(-ring[i].bottom_point.y, ring[i].bottom_point.x, 0));
					num_water_boundary++;
				}
			}
			n = normalize(n);
			if (num_water_boundary == 2) {
				auto partial_area = [](float3 const &center, float3 const &curr, float3 const &succ) {
					float3 b = center - succ;
					float3 c = center - curr;
					float3 area = cross(c, b) / 2;
					float3 norm = normalize(area);
					float cosxy = norm.z; // dot(norm, Vec3f(0,0,1.0f));
					float cosyz = norm.x; // dot(norm, Vec3f(1.0f,0,0));
					float coszx = norm.y; // dot(norm, Vec3f(0,1.0f,0));
					float3 a = curr - succ;
					float par_x = 0.5f * (a.y) * cosxy + 0.5f * (-a.z) * coszx;
					float par_y = 0.5f * (a.z) * cosyz + 0.5f * (-a.x) * cosxy;
					float par_z = 0.5f * (a.x) * coszx + 0.5f * (-a.y) * cosyz;
					return make_float3(par_x, par_y, par_z);
				};
				float3 const o = make_float3(0, 0, 0);
				float3 F = make_float3(0, 0, 0);
				float area_from_direct_n = 0;
				for (size_t i = 0; i < ring_size; i++) {
					size_t succ = (i == ring_size - 1) ? 0 : i + 1;
					if (ring[i].have_water || ring[succ].have_water) {
						F += partial_area(make_float3(0, 0, extrapolate_depth), ring[i].water_point, ring[succ].water_point) * coef_LL; // 高度为外插的负高度（为了避免边缘面片因为采样问题而过薄）
						F += partial_area(o, ring[i].bottom_point, ring[succ].bottom_point) * coef_LS;
						area_from_direct_n += dot(cross(ring[i].water_point - make_float3(0, 0, extrapolate_depth), ring[succ].water_point - make_float3(0, 0, extrapolate_depth)), n) / 6;
					}
					else {
						F += partial_area(o, ring[i].bottom_point, ring[succ].bottom_point) * coef_SO;
					}
				}
				float ps = dot(F, n) / area_from_direct_n;
				if (ps < -swSimData.m_max_p_bs)
					ps = -swSimData.m_max_p_bs;
				else if (ps > swSimData.m_max_p_bs)
					ps = swSimData.m_max_p_bs;
				buf.m_pressure_surface[i] = ps;
				return;
			} else {
				// 周围水的情况混乱，按此点有水来计算压强ps，即不continue
			}
		} else {
			// 周围无水，不需要计算压强
			buf.m_pressure_surface[i] = 0;
			return;
		}
	}
	float h = b + d;
	float3 g = swSimData.m_g;
	float3 g_ind = make_float3(dot(g, buf.m_vertex_rot[i][0]), dot(g, buf.m_vertex_rot[i][1]), dot(g, buf.m_vertex_rot[i][2]));
	float pg = -g_ind.z * h;
	float3 real1ringP[MAX_VERTEX] = { make_float3(0, 0, 0) };
	bool boundaryFlag = (buf.m_boundary[i] != 0);
	int size = 0;
	for (int j = 0; j < MAX_VERTEX; j++) {
		int idx = buf.m_vertex_oneRing[i][j];
		if (idx < 0) break;
		float hh = buf.m_bottom[idx] + buf.m_depth[idx];
		float3 pos = buf.m_point[idx] + buf.m_vertex_rot[idx][2] * hh;
		real1ringP[j] = pos;
		size++;
	}
	float3 pointi = buf.m_point[i] + buf.m_vertex_rot[i][2] * h;
	float3 normal_i = computeNormali(real1ringP, size, pointi, boundaryFlag);
	float ps = areaGradP(real1ringP, size, pointi, normal_i, boundaryFlag) * swSimData.m_gamma; //surface tension pressure
	buf.m_pressure_gravity[i] = pg;
	buf.m_pressure_surface[i] = ps;
}

__global__ void march_water_boundary_pressure_kern(bufListSWE buf, int vnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	if (buf.m_boundary[i] != 0 && buf.m_depth[i] <= swSimData.m_depth_threshold) {
		bool is_first = true;
		float min_sqrlen;
		for (int j = 0; j < MAX_VERTEX; j++) {
			int idx = buf.m_vertex_oneRing[i][j];
			if (idx < 0)
				break;
			if (buf.m_boundary[idx] != 0 || buf.m_depth[i] > swSimData.m_depth_threshold)
				continue;
			float3 delta = buf.m_point[idx] - buf.m_point[i];
			float len2 = dot(delta, delta);
			if (is_first || len2 < min_sqrlen) {
				min_sqrlen = len2;
				buf.m_pressure_surface[i] = buf.m_pressure_surface[idx];
				is_first = false;
			}
		}
	}
}

__global__ void update_velocity_kern ( bufListSWE buf, int vnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	float b = buf.m_bottom[i];
	float d = buf.m_depth[i];
	// HACK: 多计算一圈v
	if (d <= swSimData.m_depth_threshold && !buf.m_on_water_boundary[i])
		return;
	float pg = buf.m_pressure_gravity[i];
	float ps = buf.m_pressure_gravity[i];
	float p = pg + ps;
	float3 g = swSimData.m_g;
	float3 g_ind = make_float3(dot(g, buf.m_vertex_rot[i][0]), dot(g, buf.m_vertex_rot[i][1]), dot(g, buf.m_vertex_rot[i][2]));
	for (int j = 0; j < MAX_VERTEX; j++) {
		int idx = buf.m_vertex_oneRing[i][j];
		if (idx < 0) break;
		float vb = buf.m_bottom[idx];
		float vd = buf.m_depth[idx];
		float vpg = buf.m_pressure_gravity[idx];
		float vps = buf.m_pressure_surface[idx];
		float vp = vpg + vps;
		if (vd <= swSimData.m_depth_threshold) {
#if 1
			// HACK: 强行外插
			vp = pg + vps;
#else
			if (g_ind.z <= 0.0f)
				vp = (vb > h) ? pg + vps : vpg + vps;
			else
				vp = (vb > h) ? vpg + vps : pg + vps;
#endif
		}
		buf.m_tmp[i * MAX_VERTEX + j] = vp;
	}

	float3 p_target = make_float3(0,0,0); // the same
	float f_target = p; // p
	float3 grad = vertex_gradient3(i, p_target, f_target, buf);
	float3 vel = buf.m_velocity[i];
	if (swSimData.m_have_tensor) {
		float4 tensor = buf.m_tensor[i];
		vel += swSimData.dt * -swSimData.m_fric_coef * make_float3(tensor.x * vel.x + tensor.y * vel.y, tensor.z * vel.x + tensor.w * vel.y, 0);
	}
	// HACK:
	// SPECIAL CODE FOR CASE 4
	// 增加各向同性的摩擦力
	if (swSimData.m_situation == 4)
		vel += swSimData.dt * -1.0f * vel;
	vel += swSimData.dt*(-grad + make_float3(g_ind.x, g_ind.y, 0));
	
	// 风力
	if (swSimData.m_wind_coef != 0) {
		float3 wind_vel = buf.m_wind_velocity[i];
		wind_vel.z = 0;
		vel += swSimData.dt * swSimData.m_wind_coef * wind_vel * length(wind_vel);
	}

	buf.m_velocity[i] = vel;
}

__global__ void force_boundary_condition_kern ( bufListSWE buf, int vnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	if (buf.m_boundary[i]) {
		switch (buf.m_boundary[i]) {
		case 2: buf.m_velocity[i] = dot(buf.m_velocity[i], buf.m_value0[i]) * buf.m_value0[i]; break;
			//float len = length ( buf.m_velocity[i] ); if (len > 1.0) buf.m_velocity[i] = make_float3(0,0,0); break;
		case 3: buf.m_velocity[i] = buf.m_value0[i]; break;
		case 1:default: break;
		}
	}
}

__global__ void velocity_fast_march_kern ( bufListSWE buf, int vnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	float depth = buf.m_depth[i];
	if (depth <= swSimData.m_depth_threshold) {
		float3 vel;
		bool flag = false;
		int vh = nearest_vertex(make_float3(0, 0, 0), i, flag, buf);
		if (vh < 0)
			vel = make_float3 ( 0, 0, 0 );
		else
			vel = index_conv ( vh, i, buf.m_velocity[vh], buf ); // be careful

		buf.m_vector_tmp[i] = vel;
	}

}

__global__ void update_velocity_fast_march_kern ( bufListSWE buf, int vnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	if (buf.m_depth[i] <= swSimData.m_depth_threshold)
		buf.m_velocity[i] = buf.m_vector_tmp[i];
}

__global__ void calculate_delta_depth_kern(bufListSWE buf, int vnum) {
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index
	if (i >= vnum) return;
	float b = buf.m_bottom[i];
	float d = buf.m_depth[i];

	if (d > swSimData.m_depth_threshold) {
		float3 vec_p[MAX_VERTEX] = { make_float3(0, 0, 0) };
		float vec_d[MAX_VERTEX] = { 0 };
		float3 vec_u[MAX_VERTEX] = { make_float3(0, 0, 0) };
		int size = 0;
		for (int j = 0; j < MAX_VERTEX; j++) {
			int idx = buf.m_vertex_oneRing[i][j];
			if (idx < 0) break; // continue
			float3 p_tmp = buf.m_point[idx]; // plane mapping
			float3 a = p_tmp - buf.m_point[i];
			float3 b = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
			float b_norm = length(b);
			float3 p = (b_norm == 0 ? b : b*(length(a) / b_norm));

			float3 tmp = index_conv(idx, i, buf.m_velocity[idx], buf);
			vec_p[j] = p;
			vec_d[j] = fmax(buf.m_depth[idx], 0.0f);
			vec_u[j] = tmp;
			size++;
		}
		float sum = 0;
		float sum_area = 0;
		float3 p1 = make_float3(0, 0, 0);
		float d1 = buf.m_depth[i];
		float3 u1 = buf.m_velocity[i];
		for (int j = buf.m_boundary[i] == 0 ? 0 : 1; j < size; j++) {
			int k = j + 1;
			if (k == size) k = 0;
			float3 p2 = vec_p[j];
			float d2 = vec_d[j];
			float3 u2 = vec_u[j];
			float3 p3 = vec_p[k];
			float d3 = vec_d[k];
			float3 u3 = vec_u[k];
			float area = cross(p2 - p1, p3 - p1).z / 2;
			float dx = (p2.y - p3.y) * u1.x + (p3.y - p1.y) * u2.x + (p1.y - p2.y) * u3.x;
			float dy = (p3.x - p2.x) * u1.y + (p1.x - p3.x) * u2.y + (p2.x - p1.x) * u3.y;
			float divv = (dx / 2 + dy / 2) / area;
			sum_area += area;
			sum += area * (d1 + d2 + d3) / 3 * divv;
		}
		float delta_d = -swSimData.dt * sum / sum_area;
		//tempsaving used for SPH vel
		buf.m_float_tmp[i] = delta_d / swSimData.dt; ///this should be |v| along normal direction
	}
}
__global__ void update_depth_kern ( bufListSWE buf, int vnum )
{
	uint i = __mul24 ( blockIdx.x, blockDim.x ) + threadIdx.x;	// particle index
	if (i >= vnum) return;

	float b = buf.m_bottom[i];
	float d = buf.m_depth[i];

	if (d <= swSimData.m_depth_threshold && buf.sourceTerm[i] == 0.0f) return;

	if (d > swSimData.m_depth_threshold) {
		d += swSimData.dt * buf.m_float_tmp[i];
		// HACK: 强制限制水面高度
		if (swSimData.m_situation == 9 && d > 3.0f)
			d = 2.8f;
	}
	//from SPH
	d += buf.sourceTerm[i];
	
	//SPECIAL CODE FOR CASE 4
	//if(d>0.8f)
	//{
	//	d=0.7f;
	//	h=b+d;
	//}
	//SPECIAL CODE FOR CASE 4 END
	buf.m_depth[i] = d;
	//if (i==100) printf ( "%d = %f %f %f\n", i, h, length(f_target), div );
}

void updateSimParams ( FluidParamsSWE* cpufp )
{
	cudaError_t status;
#ifdef CUDA_42
	// Only for CUDA 4.x or earlier. Depricated in CUDA 5.0+
	// Original worked even if symbol was declared __device__
	status = cudaMemcpyToSymbol ( "swSimData", cpufp, sizeof ( FluidParamsSWE ) );
#else
	// CUDA 5.x+. Only works if symbol is declared __constant__
	status = cudaMemcpyToSymbol ( swSimData, cpufp, sizeof ( FluidParamsSWE ) );
#endif
}


#ifdef HYBRID
__global__ void sweInitialAdd()
{
	swAddsize = 0;
}

__global__ void addLabelSWE(bufListSWE buf, int vnum)
{

	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	float b = buf.m_bottom[i];
	float d = buf.m_depth[i];

	if (d <= swSimData.m_depth_threshold) return;

	buf.m_addId[i] = -1;
	buf.m_addLabel[i] = -1;
	if (d>MAINCUTHEIGHT)
	{
		buf.m_addId[i] = atomicAdd(&swAddsize, 1);
		buf.m_addLabel[i] = 1;

	}
	else if (d > SECCUTHEIGHT)  //1.1 = (1.5-0.7)/2
	{
		for (int j = 0; j < MAX_VERTEX; j++) {
			int idx = buf.m_vertex_oneRing[i][j];
			if (idx < 0) break;
			float vb = buf.m_bottom[idx];
			float vd = buf.m_depth[idx];

			if (vd>MAINCUTHEIGHT)
			{
				buf.m_addId[i] = atomicAdd(&swAddsize, 1);
				buf.m_addLabel[i] = 2;
				break;
			}
		}
	}

}

__global__ void addParticlesFromSWE(bufListSWE buf, bufList bufSPH, int vnum, int cpnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	float b = buf.m_bottom[i];
	float d = buf.m_depth[i];

	if (d <= swSimData.m_depth_threshold) return;

	int label = buf.m_addLabel[i];
	if (label < 0) return;


	/////THIS SUMAREA SHOULD BE CACHED
	float3 vec_point[MAX_VERTEX] = { make_float3(0, 0, 0) };
	float3 p_target = make_float3(0, 0, 0);

	int size = 0;
	for (int j = 0; j < MAX_VERTEX; j++) {
		int idx = buf.m_vertex_oneRing[i][j];
		if (idx < 0) break;
		float3 p_tmp = buf.m_point[idx]; // plane mapping
		float3 a = p_tmp - buf.m_point[i];
		float3 b = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
		float b_norm = length(b);
		float3 p = (b_norm == 0 ? b : b*(length(a) / b_norm));

		vec_point[j] = p;
		size++;
	}

	float sum_area = 0;
	for (int j = 0; j < size; j++) {
		int k = j + 1;
		if (k == size) k = 0;

		float area = (cross(vec_point[k] - p_target, vec_point[j] - p_target).z) / 2.0;
		sum_area += abs(area);
		if (size == 2) break;
	}
	/////SUMAREA

	//float radi = sqrt(sum_area);


	float deltaD = 0.0f;
	float deltaVol = 0.0f;
	float3 addCenter;

	if (label == 1)
	{
		deltaD = buf.m_depth[i] - MAINLEFT;
		deltaVol = deltaD * sum_area;
		addCenter = buf.m_point[i] + buf.m_vertex_rot[i][2] * (MAINLEFT+deltaD * 0.5);   //here, is rot[2] really outer normal? this will really work if rot[2] is outer norm.

		buf.m_depth[i] = MAINLEFT;
	}
	if (label == 2)
	{
		deltaD = buf.m_depth[i] - SECLEFT;
		deltaVol = deltaD * sum_area;
		addCenter = buf.m_point[i] + buf.m_vertex_rot[i][2] * (SECLEFT+deltaD * 0.5);

		buf.m_depth[i] = SECLEFT;
	}

	//add SPH particle
	int sphNewIndex = cpnum + buf.m_addId[i];

	bufSPH.mpos[sphNewIndex] = addCenter;
	bufSPH.mvel[sphNewIndex] = buf.m_float_tmp[i] * buf.m_vertex_rot[i][2] * 0.005f;
	bufSPH.mveleval[sphNewIndex] = bufSPH.mvel[sphNewIndex];
	bufSPH.mforce[sphNewIndex] = make_float3(0, 0, 0);
	bufSPH.mpress[sphNewIndex] = 0;
	bufSPH.mtype[sphNewIndex] = 0; //fluid
	bufSPH.mrestdens[sphNewIndex] = 600;
	bufSPH.mmass[sphNewIndex] = 600 * deltaVol * pow(0.005,3.0); ///0.005:simscale, 600:restdens
	bufSPH.malpha[sphNewIndex].tid[0] = 1.0f; bufSPH.malpha[sphNewIndex].tid[1] = 0.0f; //TYPE_NUM == 2;
	bufSPH.mclr[sphNewIndex] = COLORA(1, 0, 0, 1);

	bufSPH.sweVindex[sphNewIndex] = -1;
	bufSPH.mgcell[sphNewIndex] = GRID_UNDEF;

	//printf("OPos: (%f,%f,%f)\nPos: (%f,%f,%f)\nVel: (%f,%f,%f)\nMass: %f\n", buf.m_point[i].x, buf.m_point[i].y, buf.m_point[i].z, addCenter.x, addCenter.y, addCenter.z, bufSPH.mvel[sphNewIndex].x, bufSPH.mvel[sphNewIndex].y, bufSPH.mvel[sphNewIndex].z, 0.35 * deltaVol);

}

__global__ void sweAfterAdd(bufListSWE buf)
{
	buf.singleValueBuf[0] = swAddsize;

}

__device__ float3 contributeSPHInfluence(int i, float3 pos, int cell, bufListSWE buf, bufList bufSPH, float3 norm)
{

	float3 dist;
	float dsq, c, sum;
	float massj;

	float searchR2 = LABEL_DIS * LABEL_DIS;
	//float inmax = tempmax;
	//int j, isboundj, isboundi;

	//register float cmterm;
	////register float3 alphagrad[MAX_FLUIDNUM];

	//sum = 0.0;
	//float tempkern = -10.0;
	float3 tempmomentum = make_float3(0, 0, 0);

	if (bufSPH.mgridcnt[cell] == 0) return tempmomentum;

	int cfirst = bufSPH.mgridoff[cell];
	int clast = cfirst + bufSPH.mgridcnt[cell];
	for (int cndx = cfirst; cndx < clast; cndx++) {
		uint j = bufSPH.mgrid[cndx];
		if (bufSPH.mtype[j] == 0 && bufSPH.sweVindex[j] == i)// && dot(bufSPH.mvel[j], norm) >= 0.0f)  //mveleval?
			//here the dot() part is to say outer norm dir and vel dir be opposite, take care
		{
			float volume = bufSPH.mmass[j] / bufSPH.mrestdens[j];
			buf.sourceTerm[i] += volume;

			tempmomentum += bufSPH.mmass[j] * bufSPH.mvel[j];//mveleval?

			//delete particle
			bufSPH.mpos[j] = make_float3(-1000.0f, -1000.0f, -1000.0f);
			bufSPH.mgcell[j] = GRID_UNDEF;
			//printf("collected\n");
		}

	}

	return tempmomentum;
}



__global__ void collectLabelParticles(bufListSWE buf, bufList bufSPH, int vnum, GridInfo gi)
{

	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	float b = buf.m_bottom[i];
	float d = buf.m_depth[i];

	/////THIS SUMAREA SHOULD BE CACHED
	float3 vec_point[MAX_VERTEX] = { make_float3(0, 0, 0) };
	float3 p_target = make_float3(0, 0, 0);

	int size = 0;
	for (int j = 0; j < MAX_VERTEX; j++) {
		int idx = buf.m_vertex_oneRing[i][j];
		if (idx < 0) break;
		float3 p_tmp = buf.m_point[idx]; // plane mapping
		float3 a = p_tmp - buf.m_point[i];
		float3 b = make_float3(dot(a, buf.m_vertex_rot[i][0]), dot(a, buf.m_vertex_rot[i][1]), 0);
		float b_norm = length(b);
		float3 p = (b_norm == 0 ? b : b*(length(a) / b_norm));

		vec_point[j] = p;
		size++;
	}

	float sum_area = 0;
	for (int j = 0; j < size; j++) {
		int k = j + 1;
		if (k == size) k = 0;

		float area = (cross(vec_point[k] - p_target, vec_point[j] - p_target).z) / 2.0;
		sum_area += abs(area);
		if (size == 2) break;
	}
	/////SUMAREA

	float3 pos = buf.m_point[i];
	float3 norm = buf.m_vertex_rot[i][2];
	/////LOCATE VERTEX POSITION IN GRID
	register float3 gridMin = gi.gridMin;
	register float3 gridDelta = gi.gridDelta;
	register int3 gridRes = gi.gridRes;
	register int3 gridScan = gi.gridScanMax;

	register int		gs;
	register float3		gcf;
	register int3		gc;

	gcf = (pos - gridMin) * gridDelta;
	gc = make_int3(int(gcf.x), int(gcf.y), int(gcf.z));
	gs = (gc.y * gridRes.z + gc.z)*gridRes.x + gc.x;

	uint cell;
	if (gc.x >= 1 && gc.x <= gridScan.x && gc.y >= 1 && gc.y <= gridScan.y && gc.z >= 1 && gc.z <= gridScan.z) {
		cell = gs;
	}
	else {
		cell = GRID_UNDEF;
	}
	////LOCATE



	buf.sourceTerm[i] = 0.0f;



	if (cell == GRID_UNDEF){
		return;
	}
	// Get search cell
	int nadj = (1 * gi.gridRes.z + 1)*gi.gridRes.x + 1;
	cell -= nadj;


	float3 momentum = make_float3(0, 0, 0);//possibly a momentum source term to be added later

	for (int c = 0; c<gi.gridAdjCnt; c++)
	{
		momentum += contributeSPHInfluence(i, pos, cell + gi.gridAdj[c], buf, bufSPH, norm);
	}
	buf.sourceTerm[i] *= 200*200*200; //0.005 simscale

	buf.sourceTerm[i] /= sum_area;

}

__global__ void showBug(bufListSWE buf, int vnum)
{
	uint i = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;	// particle index				
	if (i >= vnum) return;

	if (i == 5060)
		printf("%d, %f\n", i, buf.sourceTerm[i]);
}
#endif
