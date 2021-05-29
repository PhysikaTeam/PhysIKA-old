#pragma once

#include "CollisionVec3.h"

namespace PhysIKA {
	class CollisionMesh {
	public:
		struct tri3f {
			unsigned int ids[3];
			
			tri3f() {
				ids[0] = ids[1] = ids[2] = -1;
			}

			tri3f(unsigned int id0, unsigned int id1, unsigned int id2){
				set(id0, id1, id2);
			}

			void set(unsigned int id0, unsigned int id1, unsigned int id2) {
				ids[0] = id0;
				ids[1] = id1;
				ids[2] = id2;
			}

			unsigned int id(int i) const { return ids[i]; }
			unsigned int id0() const { return ids[0]; }
			unsigned int id1() const { return ids[1]; }
			unsigned int id2() const { return ids[2]; }
		};

		tri3f* _tris = nullptr;

		// used by time integration
		vec3f* _vtxs = nullptr;
		//vec3f *_ivtxs = nullptr; // initial positions
		vec3f* _ovtxs = nullptr; // previous positions

		unsigned int _num_vtx;
		unsigned int _num_tri;

		CollisionMesh(unsigned int numVtx, unsigned int numTri, tri3f* tris, vec3f* vtxs) {
			_num_vtx = numVtx;
			_num_tri = numTri;

			_tris = tris;
			_vtxs = vtxs;
			//_ivtxs = new vec3f[numVtx];
			_ovtxs = new vec3f[numVtx];
		}

		~CollisionMesh() {
			delete[] _tris;
			delete[] _vtxs;
			//delete [] _ivtxs;
			delete[] _ovtxs;
		}

		unsigned int getNbVertices() const { return _num_vtx; }
		unsigned int getNbFaces() const { return _num_tri; }
		vec3f* getVtxs() const { return _vtxs; }
		vec3f* getOVtxs() const { return _ovtxs; }
	};
}