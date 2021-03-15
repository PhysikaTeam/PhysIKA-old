#pragma once
#include "forceline.h"

class tri3f {
public:
	unsigned int _ids[3];

	FORCEINLINE tri3f() {
		_ids[0] = _ids[1] = _ids[2] = -1;
	}

	FORCEINLINE tri3f(unsigned int id0, unsigned int id1, unsigned int id2) {
		set(id0, id1, id2);
	}

	FORCEINLINE void set(unsigned int id0, unsigned int id1, unsigned int id2) {
		_ids[0] = id0;
		_ids[1] = id1;
		_ids[2] = id2;
	}

	FORCEINLINE unsigned int id(int i) { return _ids[i]; }
	FORCEINLINE unsigned int id0() {return _ids[0];}
	FORCEINLINE unsigned int id1() {return _ids[1];}
	FORCEINLINE unsigned int id2() {return _ids[2];}
};


