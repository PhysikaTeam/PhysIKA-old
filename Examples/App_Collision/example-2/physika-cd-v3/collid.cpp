#include "cmesh.h"
#include <set>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "tmbvh.hpp"


#include "collid.h"



//extern mesh *cloths[16];
//extern mesh *lions[16];

//extern vector<int> vtx_set;
//extern set<int> cloth_set;
//extern set<int> lion_set;
static bvh *bvhCloth = NULL;

bool findd;

#include <omp.h>

# define	TIMING_BEGIN \
	{double tmp_timing_start = omp_get_wtime();

# define	TIMING_END(message) \
	{double tmp_timing_finish = omp_get_wtime();\
	double  tmp_timing_duration = tmp_timing_finish - tmp_timing_start;\
	printf("%s: %2.5f seconds\n", (message), tmp_timing_duration);}}


//#define POVRAY_EXPORT
#define OBJ_DIR "e:\\temp\\output-objs\\"

//#define VEC_CLOTH

#pragma warning(disable: 4996)

inline double fmax(double a, double b, double c)
{
  double t = a;
  if (b > t) t = b;
  if (c > t) t = c;
  return t;
}

inline double fmin(double a, double b, double c)
{
  double t = a;
  if (b < t) t = b;
  if (c < t) t = c;
  return t;
}

inline int project3(const vec3f &ax, 
	const vec3f &p1, const vec3f &p2, const vec3f &p3)
{
  double P1 = ax.dot(p1);
  double P2 = ax.dot(p2);
  double P3 = ax.dot(p3);
  
  double mx1 = fmax(P1, P2, P3);
  double mn1 = fmin(P1, P2, P3);

  if (mn1 > 0) return 0;
  if (0 > mx1) return 0;
  return 1;
}

inline int project6(vec3f &ax, 
	 vec3f &p1, vec3f &p2, vec3f &p3, 
	 vec3f &q1, vec3f &q2, vec3f &q3)
{
  double P1 = ax.dot(p1);
  double P2 = ax.dot(p2);
  double P3 = ax.dot(p3);
  double Q1 = ax.dot(q1);
  double Q2 = ax.dot(q2);
  double Q3 = ax.dot(q3);
  
  double mx1 = fmax(P1, P2, P3);
  double mn1 = fmin(P1, P2, P3);
  double mx2 = fmax(Q1, Q2, Q3);
  double mn2 = fmin(Q1, Q2, Q3);

  if (mn1 > mx2) return 0;
  if (mn2 > mx1) return 0;
  return 1;
}

#include "ccd.h"

bool
vf_test (
	vec3f &p0, vec3f &p00, vec3f &q0, vec3f &q00, vec3f &q1, vec3f &q10, vec3f &q2, vec3f &q20)
{
	vec3f qi, baryc;
	double ret = 1;
		//Intersect_VF(q00, q10, q20, q0, q1, q1, p00, p0, qi, baryc);

	return ret > -0.5;
}

bool 
ccd_contact (
			 vec3f &p0, vec3f &p00, vec3f &p1, vec3f &p10, vec3f &p2, vec3f &p20,
			vec3f &q0, vec3f &q00, vec3f &q1, vec3f &q10, vec3f &q2, vec3f &q20)
{
	if (vf_test(p0, p00, q0, q00, q1, q10, q2, q20))
		return true;
	if (vf_test(p1, p10, q0, q00, q1, q10, q2, q20))
		return true;
	if (vf_test(p2, p20, q0, q00, q1, q10, q2, q20))
		return true;
	if (vf_test(q0, q00, p0, p00, p1, p10, p2, p20))
		return true;
	if (vf_test(q1, q10, p0, p00, p1, p10, p2, p20))
		return true;
	if (vf_test(q2, q20, p0, p00, p1, p10, p2, p20))
		return true;

	return false;
}

// very robust triangle intersection test
// uses no divisions
// works on coplanar triangles

bool 
tri_contact (vec3f &P1, vec3f &P2, vec3f &P3, vec3f &Q1, vec3f &Q2, vec3f &Q3) 
{
  vec3f p1;
  vec3f p2 = P2-P1;
  vec3f p3 = P3-P1;
  vec3f q1 = Q1-P1;
  vec3f q2 = Q2-P1;
  vec3f q3 = Q3-P1;
  
  vec3f e1 = p2-p1;
  vec3f e2 = p3-p2;
  vec3f e3 = p1-p3;

  vec3f f1 = q2-q1;
  vec3f f2 = q3-q2;
  vec3f f3 = q1-q3;

  vec3f n1 = e1.cross(e2);
  vec3f m1 = f1.cross(f2);

  vec3f g1 = e1.cross(n1);
  vec3f g2 = e2.cross(n1);
  vec3f g3 = e3.cross(n1);

  vec3f  h1 = f1.cross(m1);
  vec3f h2 = f2.cross(m1);
  vec3f h3 = f3.cross(m1);

  vec3f ef11 = e1.cross(f1);
  vec3f ef12 = e1.cross(f2);
  vec3f ef13 = e1.cross(f3);
  vec3f ef21 = e2.cross(f1);
  vec3f ef22 = e2.cross(f2);
  vec3f ef23 = e2.cross(f3);
  vec3f ef31 = e3.cross(f1);
  vec3f ef32 = e3.cross(f2);
  vec3f ef33 = e3.cross(f3);

  // now begin the series of tests
  if (!project3(n1, q1, q2, q3)) return false;
  if (!project3(m1, -q1, p2-q1, p3-q1)) return false;

  if (!project6(ef11, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef12, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef13, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef21, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef22, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef23, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef31, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef32, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(ef33, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(g1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(g2, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(g3, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h1, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h2, p1, p2, p3, q1, q2, q3)) return false;
  if (!project6(h3, p1, p2, p3, q1, q2, q3)) return false;

  return true;
}




bool checkSelfIJ(int i, int j, mesh *cloth)
{
	tri3f &a = cloth->_tris[i];
	tri3f &b = cloth->_tris[j];

	for (int k=0; k<3; k++)
		for (int l=0; l<3; l++)
			if (a.id(k) == b.id(l))
				return false;

	vec3f p0 = cloth->_vtxs[a.id0()];
	vec3f p1 = cloth->_vtxs[a.id1()];
	vec3f p2 = cloth->_vtxs[a.id2()];
	vec3f q0 = cloth->_vtxs[b.id0()];
	vec3f q1 = cloth->_vtxs[b.id1()];
	vec3f q2 = cloth->_vtxs[b.id2()];

	if (tri_contact(p0, p1, p2, q0, q1, q2)) {
		if (i < j)
			printf("self contact found at (%d, %d)\n", i, j);
		else
			printf("self contact found at (%d, %d)\n", j, i);
		
		return true;
	} else
		return false;
}

bool checkSelfIJ(int ma, int fa, int mb, int fb, vector<mesh *>cloths)
{
	tri3f &a = cloths[ma]->_tris[fa];
	tri3f &b = cloths[mb]->_tris[fb];

	if (ma == mb)
	for (int k = 0; k<3; k++)
		for (int l = 0; l<3; l++)
			if (a.id(k) == b.id(l)) {
				//printf("covertex triangle found!\n");
				return false;
			}

	vec3f p0 = cloths[ma]->_vtxs[a.id0()];
	vec3f p1 = cloths[ma]->_vtxs[a.id1()];
	vec3f p2 = cloths[ma]->_vtxs[a.id2()];
	vec3f q0 = cloths[mb]->_vtxs[b.id0()];
	vec3f q1 = cloths[mb]->_vtxs[b.id1()];
	vec3f q2 = cloths[mb]->_vtxs[b.id2()];

#ifdef FOR_DEBUG
	if (findd) {
		std::cout << p0;
		std::cout << p1;
		std::cout << p2;
		std::cout << q0;
		std::cout << q1;
		std::cout << q2;
	}
#endif

	if (tri_contact(p0, p1, p2, q0, q1, q2)) {
		return true;
	}
	else
		return false;
}




extern void mesh_id(int id, vector<mesh *> &m, int &mid, int &fid);

void colliding_pairs(std::vector<mesh *> &ms, vector<tri_pair> &input, vector<tri_pair> &ret)
{
	printf("potential set %d\n", input.size());

	for (int i = 0; i < input.size(); i++) {
		unsigned int a, b;
		input[i].get(a, b);

#ifdef FOR_DEBUG
		findd = false;
		if (a == 369 && b == 3564) {
			findd = true;
		}

		if (b == 369 && a == 3564) {
			findd = true;
		}
#endif

		int ma, mb, fa, fb;
		mesh_id(a, ms, ma, fa);
		mesh_id(b, ms, mb, fb);

		if (checkSelfIJ(ma, fa, mb, fb, ms))
			ret.push_back(tri_pair(a, b));
	}
}

// CPU with BVH
// reconstruct BVH and front ...

// CPU with BVH
// refit BVH and reuse front ...

extern int getCollisionsGPU(int *, int *, int *, double *, int *, int *,double*);
extern int getSelfCollisionsSH(int *);
extern void pushMesh2GPU(int  numFace, int numVert, void *faces, void *nodes);
extern void updateMesh2GPU(void *nodes,void *prenodes,REAL thickness);

static tri3f *s_faces;
static vec3f *s_nodes;
static int s_numFace = 0, s_numVert = 0;

void updateMesh2GPU(vector <mesh *> &ms,REAL thickness)
{
	vec3f *curVert = s_nodes;

	//rky
	vec3f *preVert = new vec3f[s_numVert];
	vector<vec3f> tem;
	vec3f *oldcurVert = preVert;
	for (int i = 0; i < ms.size(); i++) {
		mesh *m = ms[i];
		memcpy(oldcurVert, m->_ovtxs, sizeof(vec3f)*m->_num_vtx);
		oldcurVert += m->_num_vtx;
	}

	for (int i = 0; i < ms.size(); i++) {
		mesh *m = ms[i];
		memcpy(curVert, m->_vtxs, sizeof(vec3f)*m->_num_vtx);
		curVert += m->_num_vtx;
	}

	for (int i = 0; i < ms.size(); i++)
	{
		for (int j = 0; j < ms[i]->_num_vtx; j++)
		{
			tem.push_back(ms[i]->_vtxs[j]);
			tem.push_back(ms[i]->_ovtxs[j]);
		}
	}

	updateMesh2GPU(s_nodes, preVert,thickness);
}

void pushMesh2GPU(vector<mesh *> &ms)
{
	for (int i = 0; i < ms.size(); i++) {
		s_numFace += ms[i]->_num_tri;
		s_numVert += ms[i]->_num_vtx;
	}

	s_faces = new tri3f[s_numFace];
	s_nodes = new vec3f[s_numVert];

	int curFace = 0;
	int vertCount = 0;
	vec3f *curVert = s_nodes;
	for (int i = 0; i < ms.size(); i++) {
		mesh *m = ms[i];
		for (int j = 0; j < m->_num_tri; j++) {
			tri3f &t = m->_tris[j];
			s_faces[curFace++] = tri3f(t.id0() + vertCount, t.id1() + vertCount, t.id2() + vertCount);
		}
		vertCount += m->_num_vtx;

		memcpy(curVert, m->_vtxs, sizeof(vec3f)*m->_num_vtx);
		curVert += m->_num_vtx;
	}

	pushMesh2GPU(s_numFace, s_numVert, s_faces, s_nodes);
}



extern void initGPU();

// GPU with BVH
// refit BVH and reuse front ...


// GPU with Spatial Hashing
// rebuild SH, and check again ...


void drawBVH(int level)
{
	if (bvhCloth == NULL) return;
	//bvhCloth->visualize(level);
}


bool cmp(vector<tri_pair> a, vector<tri_pair> b) {
	unsigned int ta[4], tb[4];
	for (int i = 0; i < 2; i++) {
		a[i].get(ta[i * 2], ta[i * 2 + 1]);
		b[i].get(tb[i * 2], tb[i * 2 + 1]);
	}
	if (ta[0] != tb[0])
		return ta[0] < tb[0];
	else if (ta[2] != tb[2])
		return ta[2] < tb[2];
	else if (ta[1] != tb[1])
		return ta[1] < tb[1];
	else
		return ta[3] < tb[3];
}

void body_collide_gpu(vector<mesh_pair> mpair, vector<CollisionDate> bodys, vector<vector<tri_pair> > &contacts,int &CCDtime, vector<ImpactInfo> &contact_info, double thickness) {
	static bvh *bvhC = NULL;
	static front_list fIntra;
	static std::vector<mesh *> meshes;

	static vector<int> _tri_offset;

	//printf("1\n");
#define MAX_CD_PAIRS 14096

	int *buffer = new int[MAX_CD_PAIRS * 2];
	int *time_buffer = new int[1];

	int *buffer_vf_ee = new int[MAX_CD_PAIRS];
	int *buffer_vertex_id = new int[MAX_CD_PAIRS * 4];
	double *buffer_dist = new double[MAX_CD_PAIRS];

	int *buffer_CCD = new int[MAX_CD_PAIRS];

	int count = 0;

	//TIMING_BEGIN

		if (bvhC == NULL)
		{
			for (int i = 0; i < bodys.size(); i++)
			{
				meshes.push_back(bodys[i].ms);
				_tri_offset.push_back(i == 0 ? bodys[0].ms->_num_tri : (_tri_offset[i - 1] + bodys[i].ms->_num_tri));
			}
			bvhC = new bvh(meshes);

			bvhC->self_collide(fIntra, meshes);
			
			initGPU();
			pushMesh2GPU(meshes);
			bvhC->push2GPU(true);							

			fIntra.push2GPU(bvhC->root());
		}

	updateMesh2GPU(meshes, thickness);

#ifdef FOR_DEBUG
	vec3f *pts = meshes[0]->getVtxs() + 3126;
	printf("XXXXXXXXXXXX3126: %lf, %lf, %lf\n", pts->x, pts->y, pts->z);
#endif

	count = getCollisionsGPU(buffer, buffer_vf_ee, buffer_vertex_id, buffer_dist, time_buffer, buffer_CCD, &thickness);
	//TIMING_END("end checking")

	tri_pair *pairs = (tri_pair *)buffer;
	vector<tri_pair> ret(pairs, pairs + count);
	//std::sort(ret.begin(), ret.end());

	for (int i = 0; i < count; i++)
	{
		ImpactInfo tem = ImpactInfo(buffer[i * 2], buffer[i * 2 + 1], buffer_vf_ee[i],
			buffer_vertex_id[i * 4], buffer_vertex_id[i * 4 + 1], buffer_vertex_id[i * 4 + 2], buffer_vertex_id[i * 4 + 3],
			buffer_dist[i], time_buffer[0],buffer_CCD[i]);

		contact_info.push_back(tem);
	}

	CCDtime = time_buffer[0];

	//Find mesh id and face id
	for (int i = 0; i < count; i++) {
		vector<tri_pair> tem;
		int mid1, mid2;
		unsigned int fid1, fid2;
		ret[i].get(fid1, fid2);
			
		for (int j = 0; j < _tri_offset.size(); j++) {
			if (fid1 <= _tri_offset[j]) {
				mid1 = j == 0 ? 0 : j;
				break;
			}
		}

		tem.push_back(tri_pair(mid1, fid1==0?0:fid1 -(mid1 == 0 ? 0 : _tri_offset[mid1 - 1])));

		int temtt = fid1 - 1 - (mid1 == 0 ? 0 : _tri_offset[mid1 - 1]);
			
		for (int j = 0; j < _tri_offset.size(); j++) {
			if (fid2 <= _tri_offset[j]) {
				mid2 = j == 0 ? 0 : j;
				break;
			}
		}

		tem.push_back(tri_pair(mid2, fid2 == 0 ? 0 : fid2 - (mid2 == 0 ? 0 : _tri_offset[mid2 - 1])));

		contacts.push_back(tem);
		//contact_time.push_back(time_buffer[i]);
	}
	//std::sort(contacts.begin(), contacts.end(), cmp);
	delete[] buffer;
	delete[] time_buffer;
	delete[] buffer_vf_ee;
	delete[] buffer_vertex_id;
	delete[] buffer_dist;
	delete[] buffer_CCD;
}

