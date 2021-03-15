#pragma once

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

	// used by time integration
	vec3f *_vtxs;
	vec3f *_ivtxs; // initial positions
	vec3f *_ovtxs; // previous positions
	vec3f *_nrms;
	bool _first;

	mesh(unsigned int numVtx, unsigned int numTri, tri3f *tris, vec3f *vtxs);
	~mesh();

	unsigned int getNbVertices() const { return _num_vtx; }
	unsigned int getNbFaces() const { return _num_tri; }
	vec3f *getVtxs() const { return _vtxs; }
	vec3f *getOVtxs() const { return _ovtxs;}
	vec3f *getIVtxs() const {return _ivtxs;}

	void update(matrix3f &, vec3f &);

	// calc norms, and prepare for display ...
	void updateNrms();

	// really displaying ...
	void display(bool tri, bool pnt, bool edge, int level, bool rigid, int, set<int> &ids);

	// povray file output
	void povray(char *fname, bool first);

	// obj file putput
	void exportObj(char *fname, bool cloth, int id);

	// load vtxs
	bool load(FILE *fp);

	BOX bound() {
		BOX a;

		for (int i=0; i<_num_vtx; i++)
			a += _vtxs[i];

		return a;
	}
};
