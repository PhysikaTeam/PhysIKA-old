#pragma once
#include "forceline.h"
#include "algorithm"
using namespace std;

class edge4f {
	unsigned int _fids[2];
	unsigned int _vids[2];

	unsigned int _vtxs[4];

public:
	FORCEINLINE void clear() {
		_fids[0] = _fids[1] = _vids[0] = _vids[1] = -1;
		_vtxs[0] = _vtxs[1] = _vtxs[2] = _vtxs[3] = -1;
	}

	FORCEINLINE edge4f() {
		clear();
	}

	FORCEINLINE edge4f(unsigned int vid0, unsigned int vid1, unsigned int fid, unsigned int vid) {
		clear();
		set(vid0, vid1, fid, vid);
	}

	FORCEINLINE void set(unsigned int vid0, unsigned int vid1, unsigned int fid, unsigned int vid) {
		_vids[0] = vid0;
		_vids[1] = vid1;
		_fids[0] = fid;

		_vtxs[1] = vid0;
		_vtxs[2] = vid1;
		_vtxs[0] = vid;
	}

	bool sort() {
		if (fid1() == -1) return false;
		if (_vids[0] < _vids[1]) return false;

		std::swap(_vids[0], _vids[1]);
		std::swap(_fids[0], _fids[1]);
		std::swap(_vtxs[0], _vtxs[3]);
		std::swap(_vtxs[1], _vtxs[2]);
		return true;
	}

	bool operator < (const edge4f &s) const
	{
		unsigned v0 = _vids[0], v1 = _vids[1];
		unsigned v3 = s._vids[0], v4 = s._vids[1];

		if (v0 > v1) std::swap(v0, v1);
		if (v3 > v4) std::swap(v3, v4);

		if (v0 == v3)
			return v1 < v4;
		else
			return v0 < v3;
	}


	bool operator == (const edge4f &s) const {
		return  (_vids[0] == s._vids[0] && _vids[1] == s._vids[1]) ||
					(_vids[1] == s._vids[0] && _vids[0] == s._vids[1]);
	}

	FORCEINLINE unsigned int fid(int i) const { return _fids[i]; }
	FORCEINLINE unsigned int vid(int i) const { return _vids[i]; }
	FORCEINLINE unsigned int fid0() const {return _fids[0];}
	FORCEINLINE unsigned int fid1() const {return _fids[1];}
	FORCEINLINE unsigned int vid0() const {return _vids[0];}
	FORCEINLINE unsigned int vid1() const {return _vids[1];}
	FORCEINLINE unsigned int *vtxs() { return _vtxs; }
	FORCEINLINE unsigned int vtxs(int i) { return _vtxs[i]; }

	FORCEINLINE void set_fid(int i, unsigned int idx) {
		_fids[i] = idx;
	}

	FORCEINLINE void set_vtxs(int i, unsigned int idx) {
		_vtxs[i] = idx;
	}
};

/*
inline bool operator<(const edge4f& a, const edge4f& b)
{
		if (a.vid0() == b.vid0())
			return a.vid1() < b.vid1();
		else
			return a.vid0() < b.vid0();
}*/

