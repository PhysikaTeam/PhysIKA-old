#pragma once

#include <stdio.h>
#include "vec3f.h"
#include "mat3f.h"

#include "tri.h"
#include "edge.h"
#include "box.h"

#include <set>
#include <vector>
using namespace std;

class mesh {
public:
	unsigned int _num_vtx;
	unsigned int _num_tri;
	
	tri3f *_tris;
	BOX *_bxs;
	double *_areas;

	// used by time integration
	vec3f *_vtxs;
	vec3f *_ivtxs; // initial positions
	vec3f *_ovtxs; // previous positions
	vec3f *_nrms;
	int *_fflags;

	vec3f _off;
	vec3f _axis;
	REAL _theta;
	matrix3f _trf;

	bool _first;
	aabb _bx;

	mesh(unsigned int numVtx, unsigned int numTri, tri3f *tris, vec3f *vtxs, vec2f *texs, tri3f *ttris,
		vec3f offset=vec3f(), vec3f axis=vec3f(), REAL theta=0);
	~mesh();

	unsigned int getNbVertices() const { return _num_vtx; }
	unsigned int getNbFaces() const { return _num_tri; }
	vec3f *getVtxs() const { return _vtxs; }
	vec3f *getOVtxs() const { return _ovtxs;}
	vec3f *getIVtxs() const {return _ivtxs;}

	void update(matrix3f &, vec3f &);

	void setFFlags(int *);

	// calc norms, and prepare for display ...
	void updateNrms();

	void updateVtxs(vec3f &);

	// really displaying ...
	void display(bool tri, bool pnt, bool edge, int level, bool rigid, set<int>&, vector<int> &, int);
	void display(bool dl);

	// povray file output
	void povray(char *fname, bool first);

	// obj file putput
	void exportObj(const char *fname, bool cloth, int id);

	// load vtxs
	bool load(FILE *fp);

	void calcAreas(vec2f *texs, tri3f *ttris) {
		for (int i = 0; i < _num_tri; i++) {
			if (texs == NULL) {
				_areas[i] = -1;
				continue;
			}

			tri3f &a = ttris[i];
			vec2f &u0 = texs[a.id0()];
			vec2f &u1 = texs[a.id1()];
			vec2f &u2 = texs[a.id2()];

			double tmp = (u1-u0).cross(u2-u0);
			_areas[i] = fabs(tmp)*0.5;
		}
	}

	void updateBxs() {
		_bx.init();

		for (int i = 0; i < _num_tri; i++) {
			tri3f &a = _tris[i];
			vec3f p0 = _vtxs[a.id0()];
			vec3f p1 = _vtxs[a.id1()];
			vec3f p2 = _vtxs[a.id2()];

			BOX bx(p0, p1);
			bx += p2;
			_bxs[i] = bx;

			_bx += bx;
		}
	}

	BOX bound() {
		return _bx;
	}
};
