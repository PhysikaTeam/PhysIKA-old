#include "cmesh.h"
#include <set>
#include <iostream>
#include <stdio.h>

using namespace std;
#include "mat3f.h"
#include "box.h"
#include "tmbvh.hpp"

#define MAX_CD_PAIRS 4096*1024

extern mesh *cloths[16];
extern mesh *lions[16];

extern vector<int> vtx_set;
extern set<int> cloth_set;
extern set<int> lion_set;
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

void checkModel()
{
	cloth_set.clear();
	lion_set.clear();

	TIMING_BEGIN
	printf("start checking ...\n");


	for (int idx=0; idx<16; idx++) {
		mesh *lion = lions[idx];
		if (lion == NULL) continue;

		for (int k=0; k<16; k++) {
			mesh *cloth = cloths[k];
			if (cloth == NULL) continue;

#pragma omp parallel for
			for (int i = 0; i<lion->getNbFaces(); i++) {
				if (lion->_fflags && lion->_fflags[i] == 1)
					continue;

				for (int j = 0; j < cloth->getNbFaces(); j++) {
					BOX &ab = lion->_bxs[i];
					BOX &bb = cloth->_bxs[j];
					if (!ab.overlaps(bb))
						continue;

					tri3f &a = lion->_tris[i];
					tri3f &b = cloth->_tris[j];

					vec3f p0 = lion->_vtxs[a.id0()];
					vec3f p1 = lion->_vtxs[a.id1()];
					vec3f p2 = lion->_vtxs[a.id2()];
					vec3f q0 = cloth->_vtxs[b.id0()];
					vec3f q1 = cloth->_vtxs[b.id1()];
					vec3f q2 = cloth->_vtxs[b.id2()];

					if (tri_contact(p0, p1, p2, q0, q1, q2)) {
						printf("#triangle contact found at (%d, %d) = (%d, %d, %d) (%d, %d, %d)\n",
							i, j,
							a.id0(), a.id1(), a.id2(),
							b.id0(), b.id1(), b.id2());
#if 0
						if (b.id0() == 6975 && b.id1() == 6974 && b.id2() == 6987) {
							vec3f t1 = lerp(q0, q1, 0.5);
							vec3f t2 = lerp(q0, q2, 0.5);
							printf("XXXZX\n");
						}
#endif
						lion_set.insert(i);
						cloth_set.insert(j);
					}

				}
			}
		}
	}
	TIMING_END("end checking")
}

bool checkSelfIJ(mesh *ma, int fa, mesh *mb, int fb)
{
	tri3f &a = ma->_tris[fa];
	tri3f &b = mb->_tris[fb];

	if (ma == mb)
		for (int k = 0; k<3; k++)
			for (int l = 0; l<3; l++)
				if (a.id(k) == b.id(l)) {
					//printf("covertex triangle found!\n");
					return false;
				}

	vec3f p0 = ma->_vtxs[a.id0()];
	vec3f p1 = ma->_vtxs[a.id1()];
	vec3f p2 = ma->_vtxs[a.id2()];
	vec3f q0 = mb->_vtxs[b.id0()];
	vec3f q1 = mb->_vtxs[b.id1()];
	vec3f q2 = mb->_vtxs[b.id2()];

	return tri_contact(p0, p1, p2, q0, q1, q2);
}

bool checkSelfIJ(int i, int j, mesh *cloth)
{
	tri3f &a = cloth->_tris[i];
	tri3f &b = cloth->_tris[j];

	for (int k = 0; k<3; k++)
		for (int l = 0; l<3; l++)
			if (a.id(k) == b.id(l))
				return false;

	vec3f p0 = cloth->_vtxs[a.id0()];
	vec3f p1 = cloth->_vtxs[a.id1()];
	vec3f p2 = cloth->_vtxs[a.id2()];
	vec3f q0 = cloth->_vtxs[b.id0()];
	vec3f q1 = cloth->_vtxs[b.id1()];
	vec3f q2 = cloth->_vtxs[b.id2()];

	if (tri_contact(p0, p1, p2, q0, q1, q2)) {
		/*		if (i < j)
		printf("self contact found at (%d, %d)\n", i, j);
		else
		printf("self contact found at (%d, %d)\n", j, i);
		*/
		return true;
	}
	else
		return false;
}


bool checkSelfIJ(int ma, int fa, int mb, int fb, vector<mesh *>cloths, bool output=0)
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

#if 0
	for (int i = 0; i < cloths[ma]->_num_tri; i++) {
		tri3f &a = cloths[ma]->_tris[i];
		if (a.id0() == 4592 && a.id1() == 4644 && a.id2() == 4885)
			printf("find %d\n", i);

		if (a.id0() == 8075 && a.id1() == 9500 && a.id2() == 8362)
			printf("find %d\n", i);
	}

	if (a.id0() == 4592 && a.id1() == 4644 && a.id2() == 4885 &&
		b.id0() == 8075 && b.id1() == 9500 && b.id2() == 8362) {
		printf("fffffind %d, %d\n", fa, fb);
		output = true;
	}
	if (b.id0() == 4592 && b.id1() == 4644 && b.id2() == 4885 &&
		a.id0() == 8075 && a.id1() == 9500 && a.id2() == 8362) {
		printf("fffffind %d, %d\n", fa, fb);
		output = true;
	}

//#ifdef FOR_DEBUG
	if (output) {
/*
		std::cout << p0;
		std::cout << p1;
		std::cout << p2;
		std::cout << q0;
		std::cout << q1;
		std::cout << q2;
*/
		printf("%d: %lf, %lf, %lf\n", a.id0(), p0.x, p0.y, p0.z);
		printf("%d: %lf, %lf, %lf\n", a.id1(), p1.x, p1.y, p1.z);
		printf("%d: %lf, %lf, %lf\n", a.id2(), p2.x, p2.y, p2.z);
		printf("%d: %lf, %lf, %lf\n", b.id0(), q0.x, q0.y, q0.z);
		printf("%d: %lf, %lf, %lf\n", b.id1(), q1.x, q1.y, q1.z);
		printf("%d: %lf, %lf, %lf\n", b.id2(), q2.x, q2.y, q2.z);
	}
//#endif
#endif
	if (tri_contact(p0, p1, p2, q0, q1, q2)) {
		return true;
	}
	else
		return false;
}

void findMatchNodes(const char *ifile, const char *ofile)
{
	mesh *cloth = cloths[0];
	vec3f *cvtxs = cloth->getVtxs();

	mesh *obstacle = lions[0];
	vec3f *ovtxs = obstacle->getVtxs();

	FILE *fpIn = fopen(ifile, "rt");
	if (!fpIn) return;

	std::vector<int> inodes;

	char buffer[1024];
	while (fgets(buffer, 1024, fpIn)) {
		int i;
		if (!sscanf(buffer, "%d", &i))
			break;
		inodes.push_back(i);
	}
	fclose(fpIn);


	//remove duplicated...
	{
		int olen = inodes.size();
		::sort(inodes.begin(), inodes.end());
		inodes.erase(::unique(inodes.begin(), inodes.end()), inodes.end());
		int nlen = inodes.size();

		if (olen > nlen) {
			printf("%d duplicated nodes have been removed!!!\n", olen - nlen);
		}
	}

	FILE *fp = fopen(ofile, "wt");
	if (!fp) return;

	for (int t=0; t<inodes.size(); t++) {
		int i = inodes[t];

		double minDist = 10000;
		int minIdx = -1;

		for (int j = 0; j < obstacle->getNbVertices(); j++) {
			double dist = (cvtxs[i] - ovtxs[j]).length();
			if (dist < minDist) {
				minDist = dist;
				minIdx = j;
			}
		}

		if (minIdx == -1) {
			printf("Impossible!\n");
			exit(0);
		}
		//fprintf(fp, "%d %d\n", i, minIdx);
		fprintf(fp, "Attach 0 %d 0 %d 0 4000\n", i, minIdx);

	}

	fclose(fpIn);
	fclose(fp);
}

void findMatchNodes2()
{
	mesh *cloth = cloths[0];
	vec3f *cvtxs = cloth->getVtxs();

	mesh *obstacle = lions[0];
	vec3f *ovtxs = obstacle->getVtxs();

	FILE *fpIn = fopen("input-nodes.txt", "rt");
	FILE *fp = fopen("nodes.txt", "wt");
	
	char buffer[1024];
	while (fgets(buffer, 1024, fpIn)) {
		int i;
		if (!sscanf(buffer, "%d", &i))
			break;

		double minDist = 10000;
		int minIdx = -1;

		for (int j = 0; j < obstacle->getNbVertices(); j++) {
			double dist = (cvtxs[i] - ovtxs[j]).length();
			if (dist < minDist) {
				minDist = dist;
				minIdx = j;
			}
		}

		if (minIdx == -1) {
			printf("Impossible!\n");
			exit(0);
		}
		fprintf(fp, "Attach 0 %d 0 %d 0 4000\n", i, minIdx);
	}

	fclose(fpIn);
	fclose(fp);
}

void projectOutside(vec3f *vtx, int idx, REAL off)
{
	mesh *obstacle = lions[0];
	vec3f *ovtxs = obstacle->getVtxs();

	tri3f &t = obstacle->_tris[idx];
	const vec3f &x0 = ovtxs[t.id0()];
	const vec3f &x1 = ovtxs[t.id1()];
	const vec3f &x2 = ovtxs[t.id2()];

	vec3f n = (x2 - x0).cross(x1 - x0);
	n.normalize();

	REAL d = (*vtx-x0).dot(n);

	if (d<0)
		return;
	else
		//*vtx = vec3f(0, 0, 0);
		*vtx -= n*(d + off);

#if 0
#if 1
	vec3f n = (x2 - x0).cross(x1 - x0);
	n.normalize();

	if (n.z > 0)
		printf("here!\n");

	*vtx += n*(depth-0.1);
#else
	vec3f pt = (x0 + x1 + x2)*0.333333;
	*vtx = pt;
#endif
#endif
}

inline REAL stp (const vec3f &u, const vec3f &v, const vec3f &w)
{
	return u.dot(v.cross(w));
}

inline REAL signed_vf_distance(const vec3f &x,
	const vec3f &y0, const vec3f &y1, const vec3f &y2,
	vec3f *n, REAL *w)
{
	vec3f _n; if (!n) n = &_n;
	REAL _w[4]; if (!w) w = _w;

	vec3f y10 = y1 - y0;
	y10.normalize();
	vec3f y20 = y2 - y0;
	y20.normalize();

	*n = y10.cross(y20);
	if ((*n).dot(*n) < 1e-6)
		return 9999999;

	(*n).normalize();
	REAL h = (x - y0).dot(*n);

	vec3f xx = x - (*n)*h;

	REAL b0 = stp(y1 - xx, y2 - xx, *n),
		b1 = stp(y2 - xx, y0 - xx, *n),
		b2 = stp(y0 - xx, y1 - xx, *n);

	w[0] = 1;
	w[1] = -b0 / (b0 + b1 + b2);
	w[2] = -b1 / (b0 + b1 + b2);
	w[3] = -b2 / (b0 + b1 + b2);

	return h;
}

REAL distance(vec3f *vtx, int idx) 
{
	mesh *obstacle = lions[0];
	vec3f *ovtxs = obstacle->getVtxs();

	tri3f &t = obstacle->_tris[idx];
	const vec3f &x0 = ovtxs[t.id0()];
	const vec3f &x1 = ovtxs[t.id1()];
	const vec3f &x2 = ovtxs[t.id2()];

	return ((x0 + x1 + x2)*.333333 - *vtx).length();

#if 0
	vec3f n;
	REAL w[4];
	REAL d = signed_vf_distance(*vtx, x0, x1, x2, &n, w);

	bool inside = (min(-w[1], min(-w[2], -w[3])) >= -1e-6);
//	if (!inside)
//		return 999999;

//	if (n.z > 0)
//		printf("here");

	return d;
#endif
}

bool projectOutside(vec3f *vtx, REAL off)
{
	mesh *obstacle = lions[0];
	vec3f *ovtxs = obstacle->getVtxs();
	int num = obstacle->getNbFaces();

	REAL minDist = 999999;
	REAL sDist = 0;
	int keepIdx = -1;
	for (int i = 0; i < num; i++) {
//		if (i == 103)
//			printf("ere\n"); 

		double dist = distance(vtx, i);
		double fdist = fabs(dist);
		if (fdist < minDist) {
			minDist = fdist;
			sDist = dist;
			keepIdx = i;
		}
	}

	if (keepIdx == -1) return false;
	//if (sDist > 0) return false;

	projectOutside(vtx, keepIdx, off);
	return true;
}

void projectOutside(REAL off)
{
	mesh *cloth = cloths[0];
	vec3f *cvtxs = cloth->getVtxs();
	int count = 0;

	for (int i = 0; i < cloth->getNbVertices(); i++) {
		//if (i != 11)
		//	continue;

		//printf("%d of %d ....\n", i, cloth->getNbVertices());
		if (projectOutside(cvtxs + i, off)) {
			count++;
			//printf("######%d\n", i);
		}
	}

	printf("%d vtx project outside...\n", count);
}

//Deforms `mesh` to remove its interpenetration from `base`.
//This is posed as least square optimization problem which can be solved
//faster with sparse solver.
//
void remove_interpenetration_fast()
{

}


void findMatchByInputID()
{
	mesh *cloth = cloths[0];
	vec3f *cvtxs = cloth->getVtxs();

	mesh *obstacle = lions[0];
	vec3f *ovtxs = obstacle->getVtxs();
	int num = obstacle->getNbVertices();

	do {
		int vid;
		scanf("%d", &vid);
		if (vid < 0)
			return;

		vec3f pt = cvtxs[vid];
		double minDist = 10000;
		int minIdx = -1;

		for (int j=0; j<num; j++) {
			double dist = (pt - ovtxs[j]).length();
			if (dist < minDist) {
				minDist = dist;
				minIdx = j;
			}
		}

		if (minIdx == -1) {
			printf("Impossible!\n");
		}
		else
			printf("Find vid=%d\n", minIdx);

	} while (true);
}

void findMatchNodes()
{
	mesh *cloth = cloths[0];
	vec3f *cvtxs = cloth->getVtxs();

	mesh *obstacle = lions[0];
	vec3f *ovtxs = obstacle->getVtxs();

	FILE *fp = fopen("nodes.txt", "wt");
	for (int i = 0; i < cloth->getNbVertices(); i++) {
		if (cvtxs[i].z < 1.62)
			continue;

		double minDist = 10000;
		int minIdx = -1;

		for (int j = 0; j < obstacle->getNbVertices(); j++) {
			double dist = (cvtxs[i] - ovtxs[j]).length();
			if (dist < minDist) {
				minDist = dist;
				minIdx = j;
			}
		}

		if (minIdx == -1) {
			printf("Impossible!\n");
			exit(0);
		}
		fprintf(fp, "Attach 0 %d 0 %d 0 4000\n", i, minIdx);

	}
	fclose(fp);
}

void checkOverlapNodes()
{
#if 1
	vtx_set.clear();

	FILE *fp = fopen("vpairs.txt", "rt");
	assert(fp != NULL);

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		int id1, id2;
		if (sscanf(buf, "%d,%d", &id1, &id2) == 2) {
			assert(id1 != id2);

			vtx_set.push_back(id1);
			vtx_set.push_back(id2);
		}
	}

	fclose(fp);
#else
	mesh *cloth = cloths[0];
	vec3f *vtxs = cloth->getVtxs();
	double eps = 0.001;

	vtx_set.clear();
	FILE *fp = fopen("vpairs.txt", "wt");
	assert(fp != NULL);

	int count = 0;
	TIMING_BEGIN

		for (int i = 0; i < cloth->getNbVertices(); i++) {
			for (int j = 0; j < cloth->getNbVertices(); j++) {
				if (i >= j) continue;

				if ((vtxs[i] - vtxs[j]).length() < eps) {
					vtx_set.push_back(i);
					vtx_set.push_back(j);
					fprintf(fp, "%d, %d\n", i, j);
					count++;
				}

			}
		}

	TIMING_END("end checking")
	printf("totally %d overlapping vtx pairs ...\n", count);

	fclose(fp);
#endif
}

void checkSelfCPU_Naive()
{
	cloth_set.clear();
	lion_set.clear();

	int count = 0;
	TIMING_BEGIN
	printf("start checking self...\n");
	for (int k=0; k<16; k++) {
		mesh *cloth = cloths[k];
		if (cloth == NULL) continue;
	
		for (int i=0; i<cloth->getNbFaces(); i++)
		for (int j=0; j<cloth->getNbFaces(); j++) {
			if (i >= j) continue;

			if (checkSelfIJ(i, j, cloth)) {
				cloth_set.insert(i);
				cloth_set.insert(j);
				count++;
			}
		}
	}
	TIMING_END("end checking")
	printf("totally %d colliding pairs ...\n", count);
}

extern void mesh_id(int id, vector<mesh *> &m, int &mid, int &fid);

void colliding_pairs(std::vector<mesh *> &ms, vector<tri_pair> &input, vector<tri_pair> &ret)
{
	/*
	if (checkSelfIJ(0, 9314, 0, 17550, ms, true)) {
		printf("Intersection!!!\n");
	}
	else
		printf("No!!!\n");
	return;
	*/

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
void checkSelfCPU_Rebuild()
{
	bvh *bvhC = NULL;
	front_list fIntra;
	std::vector<mesh *> meshes;

	cloth_set.clear();
	lion_set.clear();
	vector<tri_pair> ret;

	TIMING_BEGIN

	//if (bvhC == NULL)
	{
		for (int i = 0; i < 16; i++)
			if (cloths[i] != NULL)
				meshes.push_back(cloths[i]);

		bvhC = new bvh(meshes);
		bvhC->self_collide(fIntra, meshes);
	}

	bvhC->refit(meshes);

	vector<tri_pair> fret;
	fIntra.propogate(meshes, fret);

	colliding_pairs(meshes, fret, ret);
	std::sort(ret.begin(), ret.end());

	for (int i = 0; i < ret.size(); i++) {
		unsigned int a, b;
		ret[i].get(a, b);
		//printf("self contact found at (%d, %d)\n", a, b);
		cloth_set.insert(a);
		cloth_set.insert(b);
	}

	delete bvhC;

	TIMING_END("end checking")
	printf("totally %d colliding pairs ...\n", ret.size());
}

// CPU with BVH
// refit BVH and reuse front ...
void checkSelfCPU_Refit(const char *ofile)
{
	static front_list fIntra;
	static std::vector<mesh *> meshes;

	cloth_set.clear();
	lion_set.clear();
	vector<tri_pair> ret;

	TIMING_BEGIN

	if (bvhCloth == NULL)
	{
		for (int i = 0; i < 16; i++)
		if (cloths[i] != NULL)
			meshes.push_back(cloths[i]);

		bvhCloth = new bvh(meshes);
		bvhCloth->self_collide(fIntra, meshes);
	}

	bvhCloth->refit(meshes);
#ifdef FOR_DEBUG
	vec3f *pts = meshes[0]->getVtxs() + 3126;
	printf("XXXXXXXXXXXX3126: %lf, %lf, %lf\n", pts->x, pts->y, pts->z);
#endif

	vector<tri_pair> fret;
	fIntra.propogate(meshes, fret);

	colliding_pairs(meshes, fret, ret);
	TIMING_END("end checking")

	std::sort(ret.begin(), ret.end());

	FILE *fp = NULL;
	if (ofile)
		fp = fopen(ofile, "wt");

	for (int i = 0; i < ret.size(); i++) {
		unsigned int a, b;
		ret[i].get(a, b);
		printf("#self contact found at (%d, %d)\n", a, b);
		if (fp)
			fprintf(fp, "%d, %d\n", a, b);

		cloth_set.insert(a);
		cloth_set.insert(b);
	}

	if (fp) fclose(fp);
	printf("totally %d colliding pairs ...\n", ret.size());

	for (set<int>::iterator it = cloth_set.begin(); it != cloth_set.end(); it++)
		printf("%d\n", *it);

	printf("totally %d colliding triangles ...\n", cloth_set.size());

}

extern void pushMesh2GPU(int  numFace, int numVert, void *faces, void *nodes);
extern void updateMesh2GPU(void *nodes);
static tri3f *s_faces;
static vec3f *s_nodes;
static int s_numFace = 0, s_numVert = 0;

void updateMesh2GPU(vector <mesh *> &ms)
{
#ifdef USE_GPU
	vec3f *curVert = s_nodes;
	for (int i = 0; i < ms.size(); i++) {
		mesh *m = ms[i];
		memcpy(curVert, m->_vtxs, sizeof(vec3f)*m->_num_vtx);
		curVert += m->_num_vtx;
	}

	updateMesh2GPU(s_nodes);
#endif
}

void pushMesh2GPU(vector<mesh *> &ms)
{
#ifdef USE_GPU
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
#endif
}

extern int getSelfCollisionsGPU(int *);
extern int getSelfCollisionsSH(int *);

extern void initGPU();

// GPU with BVH
// refit BVH and reuse front ...
void checkSelfGPU()
{
#ifdef USE_GPU
	static bvh *bvhC = NULL;
	static front_list fIntra;
	static std::vector<mesh *> meshes;

	cloth_set.clear();
	lion_set.clear();

	int *buffer = new int[MAX_CD_PAIRS * 2];
	int count = 0;

	TIMING_BEGIN

	if (bvhC == NULL)
	{

		for (int i = 0; i < 16; i++)
			if (cloths[i] != NULL)
				meshes.push_back(cloths[i]);

		bvhC = new bvh(meshes);
		bvhC->self_collide(fIntra, meshes);

		initGPU();
		pushMesh2GPU(meshes);
		bvhC->push2GPU(true);
		fIntra.push2GPU(bvhC->root());
	}

	updateMesh2GPU(meshes);

#ifdef FOR_DEBUG
	vec3f *pts = meshes[0]->getVtxs()+3126;
	printf("XXXXXXXXXXXX3126: %lf, %lf, %lf\n", pts->x, pts->y, pts->z);
#endif

	count = getSelfCollisionsGPU(buffer);
	TIMING_END("end checking")

	if (count > MAX_CD_PAIRS)
		printf("Too many contacts ...\n");

	tri_pair *pairs = (tri_pair *)buffer;
	vector<tri_pair> ret(pairs, pairs+count);
	std::sort(ret.begin(), ret.end());

	for (int i = 0; i < ret.size(); i++) {
		unsigned int a, b;
		ret[i].get(a, b);
		//printf("self contact found at (%d, %d)\n", a, b);
		cloth_set.insert(a);
		cloth_set.insert(b);
	}

	delete[] buffer;
	printf("totally %d colliding pairs ...\n", ret.size());
#endif
}


// GPU with Spatial Hashing
// rebuild SH, and check again ...
void checkSelfGPU_SH()
{
#ifdef USE_GPU
	static bool init = false;
	static front_list fIntra;
	static std::vector<mesh *> meshes;

	cloth_set.clear();
	lion_set.clear();

	int *buffer = new int[MAX_CD_PAIRS * 2];
	int count = 0;

	TIMING_BEGIN

		if (!init)
		{
			init = true;

			for (int i = 0; i < 16; i++)
				if (cloths[i] != NULL)
					meshes.push_back(cloths[i]);

			initGPU();
			pushMesh2GPU(meshes);
		}

	updateMesh2GPU(meshes);

	count = getSelfCollisionsSH(buffer);
	TIMING_END("end checking")

	if (count > MAX_CD_PAIRS)
		printf("Too many contacts ...\n");

	tri_pair *pairs = (tri_pair *)buffer;
	vector<tri_pair> ret(pairs, pairs + count);
	std::sort(ret.begin(), ret.end());

	for (int i = 0; i < ret.size(); i++) {
		unsigned int a, b;
		ret[i].get(a, b);
		//printf("self contact found at (%d, %d)\n", a, b);
		cloth_set.insert(a);
		cloth_set.insert(b);
	}

	delete[] buffer;
	printf("totally %d colliding pairs ...\n", ret.size());
#endif
}


void drawBVH(int level)
{
	if (bvhCloth == NULL) return;
	bvhCloth->visualize(level);
}

bvh *bvh1=NULL, *bvh2=NULL;

bool checkCollision(mesh *b1, mesh *b2)
{
	if (bvh1 == NULL) {
		std::vector<mesh *> meshes;
		meshes.push_back(b1);

		bvh1 = new bvh(meshes);
		bvh2 = new bvh(meshes);
	}

	std::vector<mesh *> meshes1;
	meshes1.push_back(b1);
	std::vector<mesh *> meshes2;
	meshes2.push_back(b2);

	bvh1->refit(meshes1);
	bvh2->refit(meshes2);

	std::vector<tri_pair> ret;
	bvh1->collide(bvh2, ret);
	
	for (int i = 0; i < ret.size(); i++) {
		tri_pair &t = ret[i];
		unsigned int id0, id1;
		t.get(id0, id1);
		if (checkSelfIJ(b1, id0, b2, id1))
			return true;
	}

	//return (ret.size() > 0);
	return false;
}
