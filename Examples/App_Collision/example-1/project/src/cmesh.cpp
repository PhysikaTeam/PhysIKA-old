#if defined(WIN32)
#define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#endif
#include <GL/gl.h>

#include <stdio.h>
#include <string.h>

//#include "mesh_defs.h"
#include "cmesh.h"
#include "box.h"

#include <set>
using namespace std;

// for fopen
#pragma warning(disable: 4996)

inline vec3f update(vec3f &v1, vec3f &v2, vec3f &v3)
{
	vec3f s = (v2-v1);
	return s.cross(v3-v1);
}

inline vec3f
update(tri3f &tri, vec3f *vtxs)
{
	vec3f &v1 = vtxs[tri.id0()];
	vec3f &v2 = vtxs[tri.id1()];
	vec3f &v3 = vtxs[tri.id2()];

	return update(v1, v2, v3);
}

void mesh::update(matrix3f &trf, vec3f &shift)
{
	if (_first) {
		_first = false;
		memcpy(_ivtxs, _vtxs, sizeof(vec3f)*_num_vtx);
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_vtxs[i] = _ivtxs[i]*trf + shift;

}

void mesh::updateVtxs(vec3f &off) {
	for (unsigned int i = 0; i<_num_vtx; i++)
		_vtxs[i] += off;

	_off += off;
	updateBxs();
}

void mesh::updateNrms()
{
	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i] = vec3f::zero();

	for (unsigned int i=0; i<_num_tri; i++) {
		vec3f n = ::update(_tris[i], _vtxs);
		n.normalize();

		_nrms[_tris[i].id0()] += n;
		_nrms[_tris[i].id1()] += n;
		_nrms[_tris[i].id2()] += n;
	}

	for (unsigned int i=0; i<_num_vtx; i++)
		_nrms[i].normalize();
}

bool mesh::load(FILE *fp)
{
	fread(_vtxs, sizeof(vec3f), _num_vtx,  fp);
	
	// skip the _vels ...
	fread(_nrms, sizeof(vec3f), _num_vtx, fp);

	return true;
}

void initRedMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {1.0, 0.1, 0.2, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 600.0);
}

void initBlueMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {0.0, 1.0, 1.0, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 60.0);
}

void initYellowMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {1.0, 1.0, 0.0, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 60.0);
}

void initGrayMat(int side)
{
	GLfloat matAmb[4] =    {1.0, 1.0, 1.0, 1.0};
	GLfloat matDiff[4] =   {0.5, 0.5, 0.5, 1.0};
	GLfloat matSpec[4] =   {1.0, 1.0, 1.0, 1.0};
	glMaterialfv(side, GL_AMBIENT, matAmb);
	glMaterialfv(side, GL_DIFFUSE, matDiff);
	glMaterialfv(side, GL_SPECULAR, matSpec);
	glMaterialf(side, GL_SHININESS, 60.0);
}

vec3f areaColor(vec3f &x0, vec3f &x1, vec3f &x2, double areaIn)
{
	if (areaIn < 0)
		return vec3f(0.5, 0.5, 0.4);

	vec3f tmp = (x1 - x0).cross(x2 - x0);
	double area = sqrt(tmp.dot(tmp))*0.5;

	double cr = area / areaIn;
	cr = clamp(cr, 1., 1.2) - 1.0;

	if (cr < 0.1)
		return vec3f(lerp(1., 0., (0.1 - cr) * 10), 1, 0);
	else
		return vec3f(1, lerp(0., 1., (0.2 - cr) * 10), 0);
}

void setMat(vec3f &cr)
{
	GLfloat front[4] = { cr[0], cr[1], cr[2], 1 };
	GLfloat back[4] = { cr[0], cr[1], cr[2], 1 };
	glMaterialfv(GL_FRONT, GL_AMBIENT_AND_DIFFUSE, front);
	glMaterialfv(GL_BACK, GL_AMBIENT_AND_DIFFUSE, back);
}

typedef struct {
	int e1, e2;
} my_pair;

extern int ee_num;
extern my_pair ees[];
extern bool b[];
extern std::vector<int> bunnySet;

void setMat(int i)
{
	if (i == 75)
		initBlueMat(GL_FRONT);
	else {
		bool find = false;
		for (int j = 0; j < bunnySet.size(); j++)
			if (i == bunnySet[j])
				find = true;
		if (find)
			initRedMat(GL_FRONT);
		else
			initGrayMat(GL_FRONT);
	}
}

void mesh::display(bool t, bool p, bool e, int level, bool rigid,
	set<int> &ids, vector<int> &vids, int id)
{
	glPointSize(10.f);

	if (rigid) {
	//glEnable(GL_POLYGON_OFFSET_FILL);
	glPolygonOffset(5.f, 5.f);
	}

#if 0
	if (rigid) {
		glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
		initGrayMat(GL_FRONT);
	} else {
		initRedMat(GL_FRONT);

		if (id == 0) {
			initRedMat(GL_FRONT);
			initRedMat(GL_BACK);
		} else
		if (id == 1) {
			initYellowMat(GL_FRONT);
			initYellowMat(GL_BACK);
		} else {
			initBlueMat(GL_FRONT);
			initBlueMat(GL_BACK);
		}
	}
#endif

	glShadeModel(GL_SMOOTH);
	glEnableClientState( GL_VERTEX_ARRAY );
	glEnableClientState( GL_NORMAL_ARRAY );

	glVertexPointer(3, GL_DOUBLE, sizeof(double)*3, _vtxs);
	glNormalPointer(GL_DOUBLE, sizeof(double)*3, _nrms);

	if (t) {
		glEnable(GL_LIGHTING);

		glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);
	}
	else {
		if (ids.size()) {
			glEnable(GL_LIGHTING);
			tri3f *buffer = new tri3f[ids.size()];

			int i=0;
			for (set<int>::iterator it=ids.begin(); it!=ids.end(); it++, i++)
				buffer[i] = _tris[*it];

			glDrawElements( GL_TRIANGLES, ids.size()*3, GL_UNSIGNED_INT, buffer);

			delete buffer;
		}
	}

	if (!rigid && false) {
		glDisable(GL_LIGHTING);
		glBegin(GL_POINTS);
		for (int i = 0; i < _num_vtx; i++) {
			//if (_vtxs[i].z < 0.12)
			//	continue;

			if (_vtxs[i].x < 0.35 && _vtxs[i].x > -0.35)
				continue;

			printf("%d\n", i);
			glColor3f(1, 1, 0);
			glVertex3dv(_vtxs[i].v);
		}
		glEnd();
		exit(0);
	}

	if (vids.size()) {
		glDisable(GL_LIGHTING);
		glColor3f(0.0f, 0.f, 1.f);

		glBegin(GL_LINES);
		for (vector<int>::iterator it = vids.begin(); it != vids.end(); it++)
			glVertex3dv(_vtxs[*it].v);
		glEnd();
	}

/*	if (!rigid) {
		glDisable(GL_LIGHTING);
		glBegin(GL_LINES);
			glColor3f(1.0f, 0.f, 0.f);

			glVertex3dv(_vtxs[101088].v);
			glVertex3dv(_vtxs[304368].v);
		glEnd();
	}*/


	if (!rigid) {
		extern int fixed_nodes[];
		extern int fixed_num;
		glDisable(GL_LIGHTING);
//		glDrawElements(GL_POINTS, fixed_num, GL_INT, fixed_nodes);
		glBegin(GL_POINTS);
		glColor3f(1.0, 1.0, 1.0);
		for (int i=0; i<fixed_num; i++)
			glVertex3dv(_vtxs[fixed_nodes[i]].v);
		glEnd();
	} else {
		extern int find_nodes[];
		extern int find_num;
		glDisable(GL_LIGHTING);
		//		glDrawElements(GL_POINTS, fixed_num, GL_INT, fixed_nodes);
		glBegin(GL_POINTS);
		glColor3f(0.0, 1.0, 0.0);
		for (int i=0; i<find_num; i++)
			glVertex3dv(_vtxs[find_nodes[i]].v);
		glEnd();

	}

	if (!rigid && false) 
	{
	glDisable(GL_LIGHTING);
	glBegin(GL_POINTS);
	glColor3f(1.0, 0, 0);
	glVertex3dv(_vtxs[7276].v);
	glVertex3dv(_vtxs[19470].v);

	glColor3f(1.0, 1.0, 1.0);
	glVertex3dv(_vtxs[1048].v);
	glVertex3dv(_vtxs[16350].v);
	glEnd();

	glBegin(GL_LINES);
	glColor3f(1.0, 0, 0);
	glVertex3dv(_vtxs[7276].v);
	glVertex3dv(_vtxs[19470].v);

	glColor3f(1.0, 1.0, 1.0);
	glVertex3dv(_vtxs[1048].v);
	glVertex3dv(_vtxs[16350].v);
	
//	glVertex3dv(_vtxs[FIXED_V1].v);
//	glVertex3dv(_vtxs[FIXED_V2].v);

//	glVertex3dv(_vtxs[0].v);
//	glVertex3dv(_vtxs[33].v);
	glEnd();
	}

	glDisableClientState( GL_VERTEX_ARRAY );
	glDisableClientState( GL_NORMAL_ARRAY );

	if (rigid)
	glDisable(GL_POLYGON_OFFSET_FILL);

#ifdef XXXXXX
	if (e && ee_num) {
		glDisable(GL_LIGHTING);
		glBegin(GL_LINES);
		for (unsigned int i=0; i<ee_num; i++) {
			edge4f &e=_edges[rigid ? ees[i].e2 : ees[i].e1];
			vec3f deltaP = _vtxs[e.vid0()]-_vtxs[e.vid1()];

			glColor3f(1.f, 1.f, 1.f);

			glVertex3dv(_vtxs[e.vid0()].v);
			glVertex3dv(_vtxs[e.vid1()].v);
		}
		glEnd();
		glEnable(GL_LIGHTING);
	}
#endif
}

void mesh::display(bool dl)
{
	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);

	glShadeModel(GL_SMOOTH);
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_NORMAL_ARRAY);

	if (dl)
		glVertexPointer(3, GL_DOUBLE, sizeof(double) * 3, _ivtxs);
	else
		glVertexPointer(3, GL_DOUBLE, sizeof(double) * 3, _vtxs);

	glNormalPointer(GL_DOUBLE, sizeof(double) * 3, _nrms);

	glEnable(GL_LIGHTING);

	glDrawElements(GL_TRIANGLES, _num_tri * 3, GL_UNSIGNED_INT, _tris);


	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
}

#pragma warning(disable: 4996)

static char s_path[512];
static bool s_first;

static void pr_head(FILE *fp)
{
	fprintf(fp, "#include \"colors.inc\"\n");
	fprintf(fp, "#include \"textures.inc\"\n");
	fprintf(fp, "#include \"setting.inc\"\n");
}

static void pr_tail(FILE *fp)
{
#ifdef FOR_SIG06
	fprintf(fp, "#include \"tail.inc\"\n");
#endif
}

static void mesh_head(FILE *fp)
{
	fprintf(fp, "mesh2 {\n");
}

static void mesh_tail(FILE *fp, bool color)
{
	if (!color) {
		fprintf(fp, "#include \"material.inc\"\n");
	} else
		fprintf(fp, "\tpigment {rgb 1}\n");

	fprintf(fp, "}\n");
}

static void vertex_part(unsigned int num, vec3f *vtxs, FILE *fp)
{
	fprintf(fp, "\tvertex_vectors {\n");
	fprintf(fp, "\t\t%d,\n", num);
	for(unsigned int i=0; i<num; i++) {
		fprintf(fp, "\t\t<%lf, %lf, %lf>,\n", vtxs[i].x, vtxs[i].y, vtxs[i].z);
	}
	fprintf(fp, "\t}\n");

}

static void normal_part(unsigned int num, vec3f *nrms, FILE *fp)
{
	fprintf(fp, "\tnormal_vectors {\n");
	fprintf(fp, "\t\t%d,\n", num);
	for(unsigned int i=0; i<num; i++) {
		fprintf(fp, "\t\t<%lf, %lf, %lf>,\n", nrms[i].x, nrms[i].y, nrms[i].z);
	}
	fprintf(fp, "\t}\n");

}

static void face_part(unsigned int num, tri3f *tris, FILE *fp)
{
	if (s_first) {
		char fname[512];
		strcpy(fname, s_path);
		strcat(fname, "\\face_index.inc");

		FILE *fp = fopen(fname, "wt");
		fprintf(fp, "\tface_indices {\n");
		fprintf(fp, "\t\t%d,\n", num);
		for(unsigned int i=0; i<num; i++) {
			fprintf(fp, "\t\t<%d, %d, %d>,\n", tris[i].id0(), tris[i].id1(), tris[i].id2());
		}
		fprintf(fp, "\t}\n");
		fprintf(fp, "\trotate y*90\n");
		fprintf(fp, "\trotate z*90\n");
		fclose(fp);
	}

	fprintf(fp, "#include \"face_index.inc\"\n");
}

/*
static void face_part(unsigned int num, tri3f *tris, unsigned int *parts, FILE *fp)
{
	if (s_first) {
		char fname[512];
		strcpy(fname, s_path);
		strcat(fname, "\\face_index.inc");

		FILE *fp = fopen(fname, "wt");
		fprintf(fp, "\tface_indices {\n");

		int count = 0;
		for (unsigned int i=0; i<num; i++)
			if (parts[i] == 3)
				count++;

		fprintf(fp, "\t\t%d,\n", num-count);
		for(unsigned int i=0; i<num; i++) {
			if (parts[i] == 3) continue; // skip the bottom plane
			fprintf(fp, "\t\t<%d, %d, %d>, %d,\n", tris[i].id0(), tris[i].id1(), tris[i].id2(), parts[i]);
		}
		fprintf(fp, "\t}\n");
		fclose(fp);
	}

	fprintf(fp, "#include \"texture_list.inc\"\n");
	fprintf(fp, "#include \"face_index.inc\"\n");
}
*/

/*
static void texture_part(unsigned int part_num, color3 *part_colors, FILE *fp)
{
	if (s_first) {
		char fname[512];
		strcpy(fname, s_path);
		strcat(fname, "\\texture_list.inc");

		FILE *fp = fopen(fname, "wt");
		fprintf(fp, "\ttexture_list {\n");
		fprintf(fp, "\t\t%d,\n", part_num);
		for(unsigned int i=0; i<part_num; i++) {
			fprintf(fp, "\t\ttexture{pigment{rgb <%lf, %lf, %lf>} finish{ Metal }}\n", part_colors[i]._rgbs[0]/255.f, part_colors[i]._rgbs[1]/255.f, part_colors[i]._rgbs[2]/255.f);
		}
		fprintf(fp, "\t}\n");
		fclose(fp);
	}

	fprintf(fp, "#include \"texture_list.inc\"\n");
}
*/

void
mesh::povray(char *fname, bool first) //output to povray
{
	FILE *fp = fopen(fname, "wt");
	if (fp == NULL) {
		if (first) {
			printf("Cannot write pov files (at c:\\temp\\pov)\n");
		}
		return;
	}
	bool color = true;

	s_first = first;
	strcpy(s_path, fname);
	char *idx = strrchr(s_path, '\\');
	*idx = 0;
	pr_head(fp);

	mesh_head(fp);
	vertex_part(_num_vtx, _vtxs, fp);

/*	if (color) {
		normal_part(_num_vtx, _nrms, fp);
		//texture_part(_num_parts, _part_colors, fp);
		face_part(_num_tri, _tris, _parts, fp);
	} else
	{
		face_part(_num_tri, _tris, fp);
	}*/
		normal_part(_num_vtx, _nrms, fp);
		face_part(_num_tri, _tris, fp);

	mesh_tail(fp, color);

	pr_tail(fp);

	fclose(fp);
}

static const float scale = 1.f;
static const float kScale = 1.f;

void mesh::exportObj(const char *fname, bool cloth, int id)
{
	FILE *fp = fopen(fname, "wt");

	fprintf(fp, "# %d vertices\n", _num_vtx);
	fprintf(fp, "# %d triangles\n", _num_tri);

	if (cloth)
		fprintf(fp, "g Cloth\n");
	else
		fprintf(fp, "g Rigid%02d\n", id);

	for (unsigned int i=0; i<_num_vtx; i++) {
			vec3f pt = _vtxs[i];

/*			if (!cloth)
				pt -= _nrms[i]*scale;
*/

			pt *= kScale;
			fprintf(fp, "v %f %f %f\n", pt.x, pt.y, pt.z);
	}

	for (unsigned int i=0; i<_num_vtx; i++) {
		fprintf(fp, "n %f %f %f\n", _nrms[i].x, _nrms[i].y, _nrms[i].z);
	}

	for (unsigned int i=0; i<_num_tri; i++) {
		fprintf(fp, "f %d %d %d\n", _tris[i].id0()+1, _tris[i].id1()+1, _tris[i].id2()+1);
	}
	fclose(fp);
}

void generateTex(unsigned int numTri, tri3f *tris, unsigned int numVtx, vec3f *vtxs, vec2f *texs)
{
	for (int i=0; i<numVtx; i++)
		texs[i] = vec2f(1.0, 0);
}

void mesh::setFFlags(int *ptr)
{
	_fflags = ptr;
}

mesh::mesh(unsigned int numVtx, unsigned int numTri, tri3f *tris, vec3f *vtxs, vec2f *texs, tri3f *ttris, vec3f offset, vec3f axis, REAL theta)
{
	_first = true;

	_num_vtx = numVtx;
	_num_tri = numTri;

	_tris = tris;


#if 0
	glPushMatrix();
	glLoadIdentity();
	//glTranslated(offset.x, offset.y, offset.z);
	glRotated(theta, axis.x, axis.y, axis.z);
	GLdouble matrixd[16];
	glGetDoublev(GL_MODELVIEW_MATRIX, matrixd);

	glPopMatrix();
#endif

	matrix3f trf = matrix3f::rotation(axis, theta /180.0* M_PI);

	//_vtxs = vtxs;
	_vtxs = new vec3f[numVtx];
	_ivtxs = new vec3f[numVtx];

	for (int i = 0; i < numVtx; i++) {
		_ivtxs[i] = vtxs[i];
		_vtxs[i] = vtxs[i]*trf + offset;
	}

	_off = offset;

	_axis = axis;
	_theta = theta;

	_ovtxs = new vec3f[numVtx];
	_fflags = NULL;

	_nrms = new vec3f[numVtx];
	_bxs = new BOX[numTri];
	_areas = new double[numTri];

	calcAreas(texs, ttris);
	updateBxs();
}

mesh::~mesh()
{
	//_tris is shared by bunnys...
	//delete [] _tris;
	delete [] _vtxs;
	delete [] _ivtxs;
	delete [] _ovtxs;
	delete [] _nrms;
	delete[] _bxs;
	if (_fflags) delete[] _fflags;
}

#include "vec3f.h"

//#define FLT_EPSILON     1.192092896e-07F        /* smallest such that 1.0+FLT_EPSILON != 1.0 */
#define CLAMP(a, b, c)		if((a)<(b)) (a)=(b); else if((a)>(c)) (a)=(c)

inline float projectPointOntoLine(const vec3f &p, const vec3f &a, const vec3f &b)
{
	vec3f ba = b-a;
	vec3f pa = p-a;
	return pa.dot(ba)/ba.dot(ba);
}

inline void calculateEENormal(const vec3f &np1, const vec3f &np2, const vec3f &np3, const vec3f &np4, vec3f &out_normal) 
{
	vec3f line1 = np2-np1;
	vec3f line2 = np3-np1;

	// printf("l1: %f, l1: %f, l2: %f, l2: %f\n", line1[0], line1[1], line2[0], line2[1]);

	out_normal = line1.cross(line2);
	float	length = out_normal.length();
	if (length <= FLT_EPSILON)
	{ // lines are collinear
		out_normal = line1;
	}

	out_normal.normalize();
}

inline void findClosestPointsEE(const vec3f &x1, const vec3f &x2, const vec3f &x3, const vec3f &x4, float &w1, float &w2)
{
	vec3f x21 = x2- x1;
	double a = x21.dot(x21);

	vec3f x43 = x4-x3;
	double b = -x21.dot(x43);
	double c = x43.dot(x43);

	vec3f x31 =  x3-x1;
	double e = x21.dot(x31);
	double f = -x43.dot(x31);

	w1 = float((e * c - b * f) / (a * c - b * b));
	w2 = float((f - b * w1) / c);
}

// calculates the distance of 2 edges
inline float edgeEdgeDistance(const vec3f &np11, const vec3f &np12, const vec3f &np21, const vec3f &np22, float &out_a1, float &out_a2, vec3f &out_normal)
{
	vec3f temp, temp2;
	
	vec3f line1 = np12-np11;
	vec3f line2 = np22-np21;

	out_normal = line1.cross(line2);
	float length = out_normal.dot(out_normal);

	if (length < FLT_EPSILON) 
	{
		out_a2 = projectPointOntoLine(np11, np21, np22);
		if (out_a2 >= -FLT_EPSILON && out_a2 <= 1.0 + FLT_EPSILON) 
		{
			out_a1 = 0;
			calculateEENormal(np11, np12, np21, np22, out_normal);
			temp = np22 - np21;
			temp *= out_a2;
			temp2 = temp + np21;
			temp2 += np11;
			return temp2.dot(temp2);
		}

		CLAMP(out_a2, 0.0, 1.0);
		if (out_a2 > .5) 
		{ // == 1.0
			out_a1 = projectPointOntoLine(np22, np11, np12);
			if (out_a1 >= -FLT_EPSILON && out_a1 <= 1.0 + FLT_EPSILON) 
			{
				calculateEENormal(np11, np12, np21, np22, out_normal);

				// return (np22 - (np11 + (np12 - np11) * out_a1)).lengthSquared();
				return (np22-(np11+(np12-np11)*out_a1)).squareLength();
			}
		} 
		else 
		{ // == 0.0
			out_a1 = projectPointOntoLine(np21, np11, np12);
			if (out_a1 >= -FLT_EPSILON && out_a1 <= 1.0 + FLT_EPSILON) 
			{
				calculateEENormal(np11, np11, np21, np22, out_normal);

				// return (np21 - (np11 + (np12 - np11) * out_a1)).lengthSquared();
				return (np21-(np11+(np12-np11)*out_a1)).squareLength();
			}
		}

		CLAMP(out_a1, 0.0, 1.0);
		calculateEENormal(np11, np12, np21, np22, out_normal);
		if(out_a1 > .5)
		{
			if(out_a2 > .5)
				temp = np12-np22;
			else
				temp = np12-np21;
		}
		else
		{
			if(out_a2 > .5)
				temp = np11-np22;
			else
				temp = np11-np21;
		}

		return temp.squareLength();
	}
	else
	{
		out_normal.normalize();

		// If the lines aren't parallel (but coplanar) they have to intersect
		findClosestPointsEE(np11, np12, np21, np22, out_a1, out_a2);

		// If both points are on the finite edges, we're done.
		if (out_a1 >= 0.0 && out_a1 <= 1.0 && out_a2 >= 0.0 && out_a2 <= 1.0) 
		{
			// p1= np11 + (np12 - np11) * out_a1;
			vec3f p1 = (np12-np11)*out_a1+np11;
			// p2 = np21 + (np22 - np21) * out_a2;
			vec3f p2 = (np22-np21)*out_a2+np21;

			return (p1-p2).squareLength();
		}

		
		/*
		* Clamp both points to the finite edges.
		* The one that moves most during clamping is one part of the solution.
		*/
		float dist_a1 = out_a1;
		CLAMP(dist_a1, 0.0, 1.0);
		float dist_a2 = out_a2;
		CLAMP(dist_a2, 0.0, 1.0);

		// Now project the "most clamped" point on the other line.
		if (fabs(dist_a1-out_a1) > fabs(dist_a2-out_a2)) 
		{ 
			// p1 = np11 + (np12 - np11) * out_a1;
			out_a1 = dist_a1;
			vec3f p1=np11+(np12-np11)*out_a1;

			out_a2 = projectPointOntoLine(p1, np21, np22);
			CLAMP(out_a2, 0.0, 1.0);
			vec3f p2 = np21+(np22-np21)*out_a2;

			// return (p1 - (np21 + (np22 - np21) * out_a2)).lengthSquared();
			return (p1-p2).squareLength();
		} 
		else 
		{	
			// p2 = np21 + (np22 - np21) * out_a2;
			out_a2 = dist_a2;
			vec3f p2=np21+(np22-np21)*out_a2;

			out_a1 = projectPointOntoLine(p2, np11, np12);
			CLAMP(out_a1, 0.0, 1.0);
			vec3f p1 = np11+(np12-np11)*out_a1;

			// return ((np11 + (np12 - np11) * out_a1) - p2).lengthSquared();
			return (p1-p2).squareLength();
		}
	}
	
	printf("Error in edgedge_distance: end of function\n");
	return 0;
}

#ifdef DRAW_VF
#define DRAW_TRI(x1, x2, x3) 	{\
	glBegin(GL_TRIANGLES);\
	glVertex3dv(x1.v);\
	glVertex3dv(x2.v);\
	glVertex3dv(x3.v);\
	glEnd();\
}

void drawDebugVF(int idx)
{
	vec3f ox1(236.023006, 197.221259, -45.964355);
	vec3f ox2(240.705779, 194.002214, -48.944394);
	vec3f ox3(239.714874, 197.265805, -45.084964);

	vec3f v(264.144314, 307.701194, -52.967145);
	vec3f ov(264.135921, 307.703250, -52.967145);

	glPushMatrix();
	glScalef(0.5f, 0.5f, 0.5f);
	glTranslated(-v[0], -v[1], -v[2]);

	glDisable(GL_LIGHTING);
	glColor3f(1, 1, 1);
	glBegin(GL_LINES);
	glVertex3dv(ov.v);
	glVertex3dv(v.v);
	glEnd();


	if (idx != 0) {
	glColor3f(1.0, 0, 0);
	DRAW_TRI(ox1, ox2, ox3);
	}


	glColor3f(0.0, 1.0, 0);
	switch (idx) {
	case 0:
		break;

	case 1:
		{
			vec3f x1(236.022700, 197.221347, -45.964356);
			vec3f x2(240.705072, 194.002354, -48.944400);
			vec3f x3(239.714376, 197.265892, -45.084953);
			DRAW_TRI(x1, x2, x3);
			break;
		} 

	case 2:
		{
			vec3f x1(236.022652, 197.221354, -45.964355);
			vec3f x2(240.704900, 194.002379, -48.944399);
			vec3f x3(239.714316, 197.265901, -45.084953);
			DRAW_TRI(x1, x2, x3);
			break;
		}

	case 3:
		{
			vec3f x1(236.022634, 197.221357, -45.964355);
			vec3f x2(240.704835, 194.002389, -48.944398);
			vec3f x3(239.714294, 197.265904, -45.084952);
			DRAW_TRI(x1, x2, x3);
			break;
		}

	case 4:
		{
			vec3f x1(236.022406, 197.221407, -45.964357);
			vec3f x2(240.704088, 194.002537, -48.944405);
			vec3f x3(239.713782, 197.265995, -45.084942);
			DRAW_TRI(x1, x2, x3);
			break;
		}
	}

	glPopMatrix();

}

void drawVFs(int idx)
{
/*
	vec3f ox1(125.243568, 104.935150, -24.540745);
	vec3f ox2(126.417450, 102.753525, -25.883230);
	vec3f ox3(126.281105, 104.695908, -26.261549);

	vec3f v(125.788391, 104.621109, -25.384811);
	vec3f ov(125.787048, 104.628609, -25.384811);

	glPushMatrix();
	glScalef(0.1f, 0.1f, 0.1f);
	glTranslatef(-v[0], -v[1], -v[2]);

	glDisable(GL_LIGHTING);

	glColor3f(1, 1, 1);
	glBegin(GL_LINES);
	glVertex3fv(ov.v);
	glVertex3fv(v.v);
	glEnd();

	glColor3f(1.0, 0, 0);
	glBegin(GL_TRIANGLES);
	glVertex3fv(ox1.v);
	glVertex3fv(ox2.v);
	glVertex3fv(ox3.v);
	glEnd();

	glColor3f(0.0, 1.0, 0);
	if (idx == 0) {
	vec3f x1(125.243523, 104.935059, -24.540785);
	vec3f x2(126.417458, 102.753525, -25.883228);
	vec3f x3(126.281128, 104.695908, -26.261536);

	static bool first = true;

	if (first) {
		vec3f n0 = (ox3-ox1).cross(ox2-ox1);
		n0.normalize();
		printf("n0 = (%lf, %lf, %lf)\n", n0.x, n0.y, n0.z);

		vec3f n = (x3-x1).cross(x2-x1);
		n.normalize();
		printf("n1 = (%lf, %lf, %lf)\n", n.x, n.y, n.z);

		vec3f vv = v-ov;
		vv.normalize();
		printf("vv = (%lf, %lf, %lf)\n", vv.x, vv.y, vv.z);

		float d = n.dot(vv);
		printf("dot = %lf\n", d);

		float d1 = (ov-ox1).dot(n0);
		float d2 = (v-x1).dot(n);
		printf("d1 = %g, d2 = %lf\n", d1, d2);

		exit(0);
		first = false;
	}

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}
	else if (idx == 1) {
	vec3f x1(125.243523, 104.935020, -24.540787);
	vec3f x2(126.417412, 102.753525, -25.883251);
	vec3f x3(126.280899, 104.695824, -26.261671);

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}
	else if (idx == 2) {
	vec3f x1(125.243530, 104.935020, -24.540783);
	vec3f x2(126.417412, 102.753525, -25.883249);
	vec3f x3(126.280899, 104.695824, -26.261667);

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}
	else if (idx == 3) {
	vec3f x1(125.243530, 104.935020, -24.540781);
	vec3f x2(126.417412, 102.753525, -25.883249);
	vec3f x3(126.280907, 104.695824, -26.261665);

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}
	else if (idx == 4) {
	vec3f x1(125.243530, 104.935020, -24.540779);
	vec3f x2(126.417412, 102.753525, -25.883249);
	vec3f x3(126.280907, 104.695824, -26.261663);

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}
	else if (idx == 5) {
	vec3f x1(125.243538, 104.935020, -24.540777);
	vec3f x2(126.417412, 102.753525, -25.883249);
	vec3f x3(126.280907, 104.695824, -26.261663);

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}
	else if (idx == 6) {
	vec3f x1(125.243538, 104.935020, -24.540777);
	vec3f x2(126.417412, 102.753525, -25.883249);
	vec3f x3(126.280907, 104.695824, -26.261663);

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}
	else if (idx == 7) {
	vec3f x1(125.243538, 104.935020, -24.540777);
	vec3f x2(126.417412, 102.753525, -25.883249);
	vec3f x3(126.280907, 104.695831, -26.261663);

	glBegin(GL_TRIANGLES);
	glVertex3fv(x1.v);
	glVertex3fv(x2.v);
	glVertex3fv(x3.v);
	glEnd();
	}

	glPopMatrix();
*/
}

#endif

#ifdef DRAW_EDGE

vec3f x1(-23.532579, 117.236809, 114.979080);
vec3f x2(-22.045922, 119.225469, 114.844862);
vec3f ox1(-23.533084, 117.235768, 114.979674);
vec3f ox2(-22.044988, 119.226620, 114.854074);

vec3f x3(-15.849611, 118.894978, 113.963510);
vec3f x4(-29.887381, 118.949342, 115.935694);
vec3f ox3(-15.847601, 118.900320, 113.965097);
vec3f ox4(-29.885372, 118.954743, 115.937272);

void drawEdges(bool a, bool b)
{
	static bool first = true;

	if (first) {
	float a=0, b=0;
	vec3f n;
	
	float dist = edgeEdgeDistance(x1, x2, x3, x4, a, b, n);
	printf("a = %f, b = %f, dist = %f, nlen = %f\n", a, b, dist, n.squareLength());
	dist = edgeEdgeDistance(ox1, ox2, ox3, ox4, a, b, n);
	printf("a = %f, b = %f, dist = %f, nlen = %f\n", a, b, dist, n.squareLength());
	first = false;

/*	vec3f t = x3-x4;
	t.normalize();
	x3 = x4+t*1;
	t=x5-x4;
	t.normalize();
	x5 = x4+t*1;

	t=ox3-ox4;
	t.normalize();
	ox3 = ox4+t*1;
	t=ox5-ox4;
	t.normalize();
	ox5=ox4+t*1;*/
	}

	glPushMatrix();
	glTranslatef(-x4[0], -x4[1], -x4[2]);

	glDisable(GL_LIGHTING);

//	if (b) 
	{
	glColor3f(0, 1, 1);
	glBegin(GL_LINE_STRIP);
	glVertex3dv(ox3.v);
	glVertex3dv(ox4.v);
//	glVertex3dv(ox5.v);
	glEnd();
	}
//	else
	{
	glColor3f(1, 1, 1);
	glBegin(GL_LINE_STRIP);
	glVertex3dv(x3.v);
	glVertex3dv(x4.v);
//	glVertex3dv(x5.v);
	glEnd();
	}

//	if (b)
	{
	glColor3f(0, 1, 0);
	glBegin(GL_LINES);
	glVertex3dv(ox1.v);
	glVertex3dv(ox2.v);
	glEnd();
	}
//	else
	{
	glColor3f(1.0, 0, 0);
	glBegin(GL_LINES);
	glVertex3dv(x1.v);
	glVertex3dv(x2.v);
	glEnd();
	}

	glPopMatrix();
}
#endif


void beginDraw(BOX &bx)
{
	glMatrixMode( GL_MODELVIEW );
	glPushMatrix();

	vec3f pt=bx.center();
	double len = bx.height()+bx.depth()+bx.width();
	double sc = 6.0/len;

	//glRotatef(-90, 0, 0, 1);
	glScalef(sc, sc, sc);
	glTranslatef(-pt.x, -pt.y, -pt.z);
}

void endDraw()
{
	glPopMatrix();
}


#include "GL/glut.h"

void
aabb::visualize()
{
#if 0
	glColor3f(0, 1, 0);
	glLineWidth(3.0);
#else
	glColor3f(1.0f, 1.0f, 1.0f);
#endif

	glPushMatrix();
	::vec3f org = center();
	glTranslatef(org[0], org[1], org[2]);

	float w = width();
	float h = height();
	float d = depth();

	glScalef(w, h, d);
	glutWireCube(1.f);
	glPopMatrix();
}

#include "tmbvh.hpp"

void
bvh::visualize(int level)
{
	glDisable(GL_LIGHTING);
	if (_nodes)
		_nodes[0].visualize(level);
	glEnable(GL_LIGHTING);
}