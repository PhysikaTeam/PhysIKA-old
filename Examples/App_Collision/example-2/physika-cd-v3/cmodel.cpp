#include "cmesh.h"
#include <set>
#include <iostream>

using namespace std;
#include "mat3f.h"
#include "box.h"

#include <stdio.h>

#include "Collision.h"
Collision cdMgr;

// initModel
#if 0
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <helper_functions.h>
#endif

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

mesh *cloths[16];
mesh *lions[16];

set<int> cloth_set;
set<int> lion_set;
set<int> dummy_set;

BOX g_box;
static int sidx=0;

extern void clearFronts();
double ww, hh, dd;

typedef struct {
	int e1, e2;
} my_pair;

int ee_num=0;
my_pair ees[1024];

void initEEs()
{
	FILE *fp = fopen("c:\\temp\\ee.txt", "rt");
	char buffer[512];

	while (fgets(buffer, 512, fp)) {
		sscanf(buffer, "%d, %d", &ees[ee_num].e1, &ees[ee_num].e2);
		ee_num++;
	}

	fclose(fp);
}

#ifdef HI_RES
int Nx = 501;
int Nz = 501;
double xmin = 0.f, xmax = 500.f;
double zmin = 0.f, zmax = 500.f;
#else
int Nx = 101;
int Nz = 101;
double xmin = 0.f, xmax = 200.f;
double zmin = 0.f, zmax = 200.f;
#endif

#include "cmesh.h"
//#include "mesh_defs.h"
#include <vector>
using namespace std;

// for fopen
#pragma warning(disable: 4996)

bool readtrfile(const char *path, vec3f &shift)
{
	FILE *fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (strstr(buf, "<translate")) { //translate
			char *idx = strstr(buf, "x=\"");
			if (idx) {
				sscanf(idx+strlen("x=\""), "%lf", &shift.x);
			}

			idx = strstr(buf, "y=\"");
			if (idx) {
				sscanf(idx+strlen("y=\""), "%lf", &shift.y);
			}

			idx = strstr(buf, "z=\"");
			if (idx) {
				sscanf(idx+strlen("z=\""), "%lf", &shift.z);
			}
		}
	}

	fclose(fp);
	return true;
}

bool readobjfile_Vtx(const char *path, unsigned int numVtx, vec3f *vtxs, double scale, vec3f shift, bool swap_xyz)
{
	FILE *fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	int idx = 0;
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
				double x, y, z;
				sscanf(buf+2, "%lf%lf%lf", &x, &y, &z);

				vec3f p;
				if (swap_xyz)
					p = vec3f(z, x, y)*scale+shift;
				else
					p = vec3f(x, y, z)*scale+shift;

				vtxs[idx++] = p;
		}
	}

	if (idx != numVtx)
		printf("vtx num do not match!\n");

	fclose(fp);
	return true;
}

bool readobjfile(const char *path, 
				 unsigned int &numVtx, unsigned int &numTri, 
				 tri3f *&tris, vec3f *&vtxs, double scale, vec3f shift, bool swap_xyz)
{
	vector<tri3f> triset;
	vector<vec3f> vtxset;

	FILE *fp = fopen(path, "rt");
	if (fp == NULL) return false;

	char buf[1024];
	while (fgets(buf, 1024, fp)) {
		if (buf[0] == 'v' && buf[1] == ' ') {
				double x, y, z;
				sscanf(buf+2, "%lf%lf%lf", &x, &y, &z);

				if (swap_xyz)
					vtxset.push_back(vec3f(z, x, y)*scale+shift);
				else
					vtxset.push_back(vec3f(x, y, z)*scale+shift);
		} else
			if (buf[0] == 'f' && buf[1] == ' ') {
				int id0, id1, id2, id3;
				bool quad = false;

				sscanf(buf+2, "%d", &id0);
				char *nxt = strchr(buf+2, ' ');
				sscanf(nxt+1, "%d", &id1);
				nxt = strchr(nxt+1, ' ');
				sscanf(nxt+1, "%d", &id2);

				nxt = strchr(nxt+1, ' ');
				if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
					if (sscanf(nxt+1, "%d", &id3))
						quad = true;
				}

				id0--, id1--, id2--, id3--;
				triset.push_back(tri3f(id0, id1, id2));

				if (quad)
					triset.push_back(tri3f(id0, id2, id3));
			}
	}
	fclose(fp);

	if (triset.size() == 0 || vtxset.size() == 0)
		return false;

	numVtx = vtxset.size();
	vtxs = new vec3f[numVtx];
	for (unsigned int i=0; i<numVtx; i++)
		vtxs[i] = vtxset[i];

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i=0; i<numTri; i++)
		tris[i] = triset[i];

	return true;
}


bool read_transf(const char *path, matrix3f &trf, vec3f &shifted, bool swap_xyz)
{
// reading this typical input ...
//<rotate angle="109.093832" x="-0.115552" y="0.038672" z="-0.992548"/>
//<scale value="1.000000"/>
//<translate x="-0.181155" y="0.333876" z="0.569190"/>

	FILE *fp = fopen(path, "rt");
	if (fp == NULL)
		return false;

	char buffer[1024];
	float angle, x, y, z;
	trf = matrix3f::identity();

	while (fgets(buffer, 1024, fp)) {
		char *ptr = NULL;
		if (ptr = strstr(buffer, "rotate")) {
			ptr += strlen("rotate")+1;

			sscanf(strstr(ptr, "angle=")+strlen("angle=")+1, "%g", &angle);

			if (swap_xyz) {
				sscanf(strstr(ptr, "x=")+strlen("x=")+1, "%g", &z);
				sscanf(strstr(ptr, "y=")+strlen("y=")+1, "%g", &x);
				sscanf(strstr(ptr, "z=")+strlen("z=")+1, "%g", &y);
			} else {
				sscanf(strstr(ptr, "x=")+strlen("x=")+1, "%g", &x);
				sscanf(strstr(ptr, "y=")+strlen("y=")+1, "%g", &y);
				sscanf(strstr(ptr, "z=")+strlen("z=")+1, "%g", &z);
			}

			trf *= matrix3f::rotation(-vec3f(x, y, z), (angle/180.f)*M_PI);
		} else if (ptr = strstr(buffer, "scale")) {
			ptr += strlen("scale")+1;
			sscanf(strstr(ptr, "value=")+strlen("value=")+1, "%g", &angle);

			trf *= matrix3f::scaling(angle, angle, angle);
		} else if (ptr = strstr(buffer, "translate")) {
			ptr += strlen("translate")+1;

			if (swap_xyz) {
				sscanf(strstr(ptr, "x=")+strlen("x=")+1, "%g", &z);
				sscanf(strstr(ptr, "y=")+strlen("y=")+1, "%g", &x);
				sscanf(strstr(ptr, "z=")+strlen("z=")+1, "%g", &y);
			} else {
				sscanf(strstr(ptr, "x=")+strlen("x=")+1, "%g", &x);
				sscanf(strstr(ptr, "y=")+strlen("y=")+1, "%g", &y);
				sscanf(strstr(ptr, "z=")+strlen("z=")+1, "%g", &z);
			}

			shifted = vec3f(x, y, z);
		}
	}

	fclose(fp);
	return true;
}


bool readobjdir(const char *path, 
				 unsigned int &numVtx, unsigned int &numTri, 
				 tri3f *&tris, vec3f *&vtxs, double scale2, vec3f shift2, bool swap_xyz)
{
	char objfile[1024];
	char transfile[1024];

	vector<tri3f> triset;
	vector<vec3f> vtxset;
	int idxoffset = 0;

	for (int i=0; i<16; i++) {
		sprintf(objfile, "%sobs_%02d.obj", path, i);
		sprintf(transfile, "%s0000obs%02d.txt", path, i);

		matrix3f trf;
		vec3f shifted;

		if (false == read_transf(transfile, trf, shifted, swap_xyz))
		{
			printf("trans file %s read failed...\n", transfile);
			return false;
		}
		
		FILE *fp = fopen(objfile, "rt");
		if (fp == NULL) return false;

		char buf[1024];
		while (fgets(buf, 1024, fp)) {
			if (buf[0] == 'v' && buf[1] == ' ') {
					float x, y, z;
					if (swap_xyz)
						sscanf(buf+2, "%g%g%g", &y, &z, &x);
					else
						sscanf(buf+2, "%g%g%g", &x, &y, &z);

					vec3f pt = vec3f(x, y, z)*trf+shifted;

					vtxset.push_back(pt*scale2+shift2);
			} else
				if (buf[0] == 'f' && buf[1] == ' ') {
					int id0, id1, id2, id3;
					bool quad = false;

					sscanf(buf+2, "%d", &id0);
					char *nxt = strchr(buf+2, ' ');
					sscanf(nxt+1, "%d", &id1);
					nxt = strchr(nxt+1, ' ');
					sscanf(nxt+1, "%d", &id2);

					nxt = strchr(nxt+1, ' ');
					if (nxt != NULL && nxt[1] >= '0' && nxt[1] <= '9') {// quad
						if (sscanf(nxt+1, "%d", &id3))
							quad = true;
					}

					id0--, id1--, id2--, id3--;
					triset.push_back(tri3f(id0+idxoffset, id1+idxoffset, id2+idxoffset));

					if (quad)
						triset.push_back(tri3f(id0+idxoffset, id2+idxoffset, id3+idxoffset));
				}
		}
		fclose(fp);

		if (triset.size() == 0 || vtxset.size() == 0)
			return false;

		idxoffset = vtxset.size();
	}

	numVtx = vtxset.size();
	vtxs = new vec3f[numVtx];
	for (unsigned int i=0; i<numVtx; i++)
		vtxs[i] = vtxset[i];

	numTri = triset.size();
	tris = new tri3f[numTri];
	for (unsigned int i=0; i<numTri; i++)
		tris[i] = triset[i];

	return true;
}

void initObjs(char *path, int stFrame)
{
	unsigned int numVtx=0, numTri=0;
	vec3f *vtxs = NULL;
	tri3f *tris = NULL;

	double scale = 1.;
	vec3f shift(0, 0, 0);

	char buff[512];

#ifdef VEC_CLOTH
	//sprintf(buff, "E:\\work\\cudaCloth-5.5\\meshes\\qwoman2\\body000.obj", sidx);
	//sprintf(buff, "E:\\temp4\\kkkk-cpu\\0000_00zzz.obj");
	//sprintf(buff, "E:\\data\\man_kneeling_sequence\\man_kneels00.obj");
	sprintf(buff, "E:\\data\\man_kneeling_5split\\body000.obj");

#else
#ifdef WIN32
	sprintf(buff, "%s\\%04d_ob.obj", path, stFrame);
#else
	sprintf(buff, "%s/%04d_ob.obj", path, stFrame);
#endif
#endif

	if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false)) {
		printf("loading %s ...\n", buff);

		lions[0] = new mesh(numVtx, numTri, tris, vtxs);
		lions[0]->updateNrms();
		printf("Read obj file don (#tri=%d, #vtx=%d)\n", numTri, numVtx);
	}
	else
	for (int idx=0; idx<16; idx++) {
#ifdef WIN32
		sprintf(buff, "%s\\obs_%02d.obj", path, idx);
#else
		sprintf(buff, "%s/obs_%02d.obj", path, idx);
#endif
		printf("loading %s ...\n", buff);

		if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false)) {
			lions[idx] = new mesh(numVtx, numTri, tris, vtxs);
			lions[idx]->updateNrms();
			printf("Read obj file don (#tri=%d, #vtx=%d)\n", numTri, numVtx);
		}
	}
}


void initObj(const char *ofile)
{
	unsigned int numVtx = 0, numTri = 0;
	vec3f *vtxs = NULL;
	tri3f *tris = NULL;
	vec2f *texs = NULL;
	tri3f *ttris = NULL;

	double scale = 1.;
	vec3f shift(0, 0, 0);

	if (readobjfile(ofile, numVtx, numTri, tris, vtxs, scale, shift, false)) {
		//printf("loading %s ...\n", buff);

		lions[0] = new mesh(numVtx, numTri, tris, vtxs);
		lions[0]->updateNrms();
		printf("Read obj file don (#tri=%d, #vtx=%d)\n", numTri, numVtx);

		vector<unsigned int> ttris;
		vector<vec3f> tvtxs;

		for (int i = 0; i < numTri; i++) {
			ttris.push_back(tris[i].id0());
			ttris.push_back(tris[i].id1());
			ttris.push_back(tris[i].id2());
		}
		for (int i = 0; i < numVtx; i++)
			tvtxs.push_back(vtxs[i]);

		cdMgr.Transform_Mesh(numVtx, numTri, ttris, tvtxs, tvtxs, 1);
	}
}

extern void beginDraw(BOX &);
extern void endDraw();

void drawModel(bool tri, bool pnt, bool edge, bool re, int level)
{
	if (!g_box.empty())
		beginDraw(g_box);

	if (edge)
		for (int i = 0; i<16; i++)
			//for (int i=0; i<1; i++)
			if (cloths[i])
				cloths[i]->display(tri, false, re, level, false, i, cloth_set);

#ifndef FIXED
	for (int i = 0; i<16; i++)
		if (pnt && lions[i])
			lions[i]->display(tri, false, false, level, true, i, lion_set);
#endif

	if (!g_box.empty())
		endDraw();
}

void initAnimation()
{
}

inline unsigned int IDX(int i, int j, int Ni, int Nj)
{
	if (i < 0 || i >= Ni) return -1;
	if (j<0 || j >= Nj) return -1;
	return i*Nj+j;
}

void initCloths(char *path, int stFrame)
{
	unsigned int numVtx=0, numTri=0;
	vec3f *vtxs = NULL;
	tri3f *tris = NULL;

	double scale = 1.f;
	vec3f shift;

	sidx = stFrame;

	for (int idx=0; idx<16; idx++) {
		char buff[512];
#ifdef WIN32
		sprintf(buff, "%s\\%04d_%02d.obj", path, sidx, idx);
#else
		sprintf(buff, "%s/%04d_%02d.obj", path, sidx, idx);
#endif
		printf("loading %s ...\n", buff);

		if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false)) {
			cloths[idx] = new mesh(numVtx, numTri, tris, vtxs);
			g_box += cloths[idx]->bound();
			cloths[idx]->updateNrms();
			printf("Read cloth from obj file (#tri=%d, #vtx=%d)\n", numTri, numVtx);
			cout << g_box.getMin() << endl;
			cout << g_box.getMax() << endl;
			ww = g_box.width();
			hh = g_box.height();
			dd = g_box.depth();
			cout << "w =" << g_box.width() << ", h =" << g_box.height() << ", d =" << g_box.depth() << endl;
		}
	}

	sidx++;
}


void initCloth(const char *cfile)
{
	unsigned int numVtx = 0, numTri = 0;
	vec3f *vtxs = NULL;
	tri3f *tris = NULL;
	vec2f *texs = NULL;
	tri3f *ttris = NULL;

	double scale = 1.f;
	vec3f shift;
	int idx = 0;

	if (readobjfile(cfile, numVtx, numTri, tris, vtxs, scale, shift, false)) {
		cloths[idx] = new mesh(numVtx, numTri, tris, vtxs);
		g_box += cloths[idx]->bound();
		cloths[idx]->updateNrms();
		printf("Read cloth from obj file (#tri=%d, #vtx=%d)\n", numTri, numVtx);
		cout << g_box.getMin() << endl;
		cout << g_box.getMax() << endl;
		ww = g_box.width();
		hh = g_box.height();
		dd = g_box.depth();
		cout << "w =" << g_box.width() << ", h =" << g_box.height() << ", d =" << g_box.depth() << endl;

		vector<unsigned int> ttris;
		vector<vec3f> tvtxs;

		for (int i = 0; i < numTri; i++) {
			ttris.push_back(tris[i].id0());
			ttris.push_back(tris[i].id1());
			ttris.push_back(tris[i].id2());
		}
		for (int i = 0; i < numVtx; i++)
			tvtxs.push_back(vtxs[i]);

		cdMgr.Transform_Mesh(numVtx, numTri, ttris, tvtxs, tvtxs, 0);
	}
}

void initModel(int argc, char **argv, char *path, int st)
{
//	int devID = findCudaDevice(argc, (const char **)argv);

#ifndef VEC_CLOTH
	initCloths(path, st);
#endif

	initObjs(path, st);

//	initEEs();

#ifdef POVRAY_EXPORT
	// output POVRAY file
	{
	char pvfile[512];
	sprintf(pvfile, "c:\\temp\\pov\\cloth0000.pov");
	cloth->updateNrms(); // we need normals
	cloth->povray(pvfile, true);
	}
#endif
}

inline void addTri(unsigned int mid, unsigned int fid)
{
	if (mid == 0)
		cloth_set.insert(fid);
	else if (mid == 1)
		lion_set.insert(fid);
	else
		assert(0);
}

void initModel(const char *cfile, const char *ofile)
{
	initCloth(cfile);
	initObj(ofile);
}

void checkModel()
{
	cdMgr.Collid();

	int ret = cdMgr.getNumContacts();
	//printf("Found %d contacts...\n");

	vector<vector<tri_pair> > pairs = cdMgr.getContactPairs();

	int count = 0;
	for (int i = 0; i < pairs.size(); i++) {
		tri_pair &t1 = pairs[i][0];
		tri_pair &t2 = pairs[i][1];

		if (t1.id0() == t2.id0()) //self cd
			continue;

		printf("%d: (%d, %d) - (%d, %d)\n", count + 1, t1.id0(), t1.id1(), t2.id0(), t2.id1());
		addTri(t1.id0(), t1.id1());
		addTri(t2.id0(), t2.id1());
		count++;
	}

	printf("Found %d inter-object contacts...\n", count);
}

void quitModel()
{
	for (int i=0; i<16; i++)
		if (cloths[i])
			delete cloths[i];
	for (int i=0; i<16; i++)
		if (lions[i])
			delete lions[i];
}

extern void beginDraw(BOX &);
extern void endDraw();
/*
int fixed_nodes[] = {52, 145, 215, 162, 214, 47, 48, 38, 221, 190, 1, 42,  25, 34, 65, 63};
int fixed_num = 0;
*/
//int fixed_num = 0;
/*
int fixed_nodes[] = {
0, 23638, 23639, 23640, 23641, 23642, 23643, 23644,
23645, 23646, 23647, 23648, 23649, 23650, 23651, 
23652, 23653, 23654, 23655, 23656, 23657, 23658,
23659, 23660, 23661, 23662, 23663, 23664, 23665, 
23666, 23667, 23668, 
68, 69, 70, 71, 72, 73, 74, 75,
76, 77, 78, 79, 80, 81, 82, 83, 
84, 85, 86, 87, 88, 89, 90, 91, 
92, 93, 94, 95, 96, 97, 98, 99};
int fixed_num = sizeof(fixed_nodes)/sizeof(int);
*/
/*
int fixed_nodes[] = {
11252, 11253, 11254, 11255, 11256, 11257, 11258, 11259, 11260,
11261, 11262, 11263, 11264, 11265, 11539, 11327,
4552, 4553, 4554, 4555, 4556, 4557, 4558, 4559, 4560,
4561, 4562, 4563, 4564, 4654};
*/
/*
5388, 5390, 34, 34, 11910, 11910, 11911, 3955, 16165, 16164, 16139, 
16140, 16140, 3945, 16144, 16142, 5146, 5388, 5388, 5387, 509, 509, 5386, 5525,
9818, 1881, 9791, 9788, 1870, 9789,
*/

//bishop
int fixed_nodes[] = {
	9681, 14539, 14538, 14537, 14536, 14535, 14534, 14533,
	14532, 14531, 14530, 14529, 14528, 14527, 14526, 14525,
	29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,
	0, 4897, 4896, 4895, 4894, 4893, 4892, 4891, 4890, 4889,
	4888, 4887, 4886, 4885, 4884, 4883, 4882,
	9710, 9711, 9712, 9713, 9714, 9715, 9716,
	9717, 9718, 9719, 9720, 9721, 9722, 9723, 9724
};
//int fixed_num = sizeof(fixed_nodes)/sizeof(int);
int fixed_num = 0;

int find_nodes[200];
int find_num=0;
//int find_nodes[] = {2509, 11702, 11688, 5289, 5303, 434, 5128, 463, 5458, 5455, 11843, 11845, 11802, 11672, 11541, 2509};
//int find_num = fixed_num;

/*
int find_nodes[] = {11537, 11537, 2564, 11538, 11538, 11538, 2563, 2563, 2563, 2563, 11685,
11685, 11685, 2562, 11850, 11850, 5464, 5462, 5462, 487, 5280, 5280, 5280, 488,
488, 488, 5122, 5122, 5122, 5122, 5123, 5123, 5123, 5307, 5307, 5310, 5310, 5413,
5411, 491, 491, 5401, 5401, 492, 492, 5315, 5315, 29, 29, 11712, 11712, 2567,
2567, 2567, 11791, 11792, 11792, 11803, 11803, 11708, 11708, 11707, 11707, 11707};
int find_num = sizeof(find_nodes)/sizeof(int);
*/
void drawOther();

void dumpModel()
{
}

void loadModel()
{
}

void FindMatch(mesh *cloth, mesh *obj)
{
	find_num = fixed_num;
	printf("Match list:");
	for (int i=0; i<fixed_num; i++) {
		vec3f pt = cloth->_vtxs[fixed_nodes[i]];

		double mDist = 1000;
		int keep = -1;
		for (int j=0; j<obj->_num_vtx; j++) {
			double dist = (pt-obj->_vtxs[j]).squareLength();

			if (dist < mDist) {
				mDist = dist;
				keep = j;
			}
		}
		find_nodes[i] = keep;
		printf("%d, ", keep);
	}
	printf("\n");

	printf("\t\"handles\": [\n");
	for (int i=0; i<fixed_num; i++) {
		if (i != fixed_num-1)
			printf("\t\t{\"cloth\": 0, \"type\": \"attach\", \"nodes\": [%d, %d]},\n", fixed_nodes[i], find_nodes[i]);
		else
			printf("\t\t{\"cloth\": 0, \"type\": \"attach\", \"nodes\": [%d, %d]}\n", fixed_nodes[i], find_nodes[i]);
	}
	printf("\t],\n");
}


void findMatch()
{
	FindMatch(cloths[0], lions[0]);
}

static int objIdx =0;
static bool first = true;

bool loadVtx(char *cfile, char *ofile, bool orig)
{
	/*FILE *fp = fopen(cfile, "rb");
	if (fp) {
		fread(orig ? cloth->getOVtxs() : cloth->getVtxs(), sizeof(double), cloth->getNbVertices()*3, fp);
		fclose(fp);
	}

	fp = fopen(ofile, "rb");
	if (fp) {
		fread(orig ? lions[0]->getOVtxs() : lions[0]->getVtxs(), sizeof(double), lions[0]->getNbVertices()*3, fp);
		fclose(fp);
	}
	*/
	return true;
}

bool dynamicModel(char *path, bool output, bool rev)
{
	char buff[512];
	BOX gbox;

	if (rev)
		sidx -= 2;

	for (int k = 0; k<16; k++) {
#ifdef WIN32
		sprintf(buff, "%s\\%04d_%02d.obj", path, sidx, k);
#else
		sprintf(buff, "%s/%04d_%02d.obj", path, sidx, k);
#endif


		printf("loading %s ...\n", buff);

		FILE  *fp = fopen(buff, "rb");
		if (fp == NULL || cloths[k] == NULL)
			continue;
		fclose(fp);

		double scale = 1.f;
		vec3f shift;
		matrix3f trf;

#define FIXED_RES
#ifdef FIXED_RES
		memcpy(cloths[k]->getOVtxs(), cloths[k]->getVtxs(), sizeof(vec3f)*cloths[k]->getNbVertices());
		readobjfile_Vtx(buff, cloths[k]->getNbVertices(), cloths[k]->getVtxs(), scale, shift, false);
#else
		delete cloths[k];

		unsigned int numVtx = 0, numTri = 0;
		vec3f *vtxs = NULL;
		tri3f *tris = NULL;

		if (readobjfile(buff, numVtx, numTri, tris, vtxs, scale, shift, false)) {
			unsigned int numEdge = 0;
			edge4f *edges = NULL;
			buildEdges(numTri, tris, numEdge, edges);

			cloths[k] = new mesh(numVtx, numTri, numEdge, tris, edges, vtxs, NULL);
			printf("Read obj file don (#tri=%d, #vtx=%d, #edge=%d)\n", numTri, numVtx, numEdge);
		}
#endif
		cloths[k]->updateNrms();
		/*
		gbox += cloths[k]->bound();
		cout << gbox.getMin() << endl;
		cout << gbox.getMax() << endl;
		cout << "w =" << gbox.width() << ", h =" << gbox.height() << ", d =" << gbox.depth() << endl;
		cout << "w =" << gbox.width()/ww << ", h =" << gbox.height()/hh << ", d =" << gbox.depth()/dd << endl;
		*/
	}

	/*		if (output) {
	sprintf(buff, "%s%04d-cloth.obj", OBJ_DIR, objIdx++);
	cloth->exportObj(buff, true, 0);
	}
	*/
#ifdef VEC_CLOTH
	//sprintf(buff, "E:\\work\\cudaCloth-5.5\\meshes\\qwoman2\\body%03d.obj", sidx == 0 ? 0 : sidx-1);
	//sprintf(buff, "E:\\work\\cudaCloth-5.5\\meshes\\qwoman2\\body%03d.obj", sidx == 0 ? 0 : sidx-1);
	//sprintf(buff, "E:\\data\\man_kneeling_sequence\\man_kneels%02d.obj", sidx == 0 ? 0 : sidx-1);
	sprintf(buff, "E:\\data\\man_kneeling_5split\\body%03d.obj", sidx == 0 ? 0 : sidx - 1);
#else
#ifdef WIN32
	sprintf(buff, "%s\\%04d_ob.obj", path, sidx);
#else
	sprintf(buff, "%s/%04d_ob.obj", path, sidx);
#endif
#endif

	if (lions[0])
		memcpy(lions[0]->getOVtxs(), lions[0]->getVtxs(), sizeof(vec3f)*lions[0]->getNbVertices());

	if (lions[0] && readobjfile_Vtx(buff, lions[0]->getNbVertices(), lions[0]->getVtxs(), 1., vec3f(), false))
	{
		printf("loading %s ...\n", buff);
		lions[0]->updateNrms();
	}
	else
	{
		vec3f shift;
		matrix3f trf;

		for (int idx = 0; idx<16; idx++) {
			if (lions[idx] == NULL) continue;

#ifdef WIN32
			sprintf(buff, "%s\\%04dobs%02d.txt", path, sidx, idx);
#else
			sprintf(buff, "%s/%04dobs%02d.txt", path, sidx, idx);
#endif
			printf("loading %s ...\n", buff);
			//readtrfile(buff, shift);
			if (read_transf(buff, trf, shift, false))
				lions[idx]->update(trf, shift);

			if (output) {
#ifdef WIN32
				sprintf(buff, "%s\\%04d-obj-%02d.obj", path, sidx, idx);
#else
				sprintf(buff, "%s\\%04d-obj-%02d.obj", path, sidx, idx);
#endif
				lions[idx]->exportObj(buff, false, idx);
			}
		}
	}

	sidx += 1;
	return true;
}
