#ifndef SELFCOLLISIONDATE
#define SELFCOLLISIONDATE
#include "cmesh.h"

struct CollisionDate{
	CollisionDate(mesh* m, bool flag) {
		ms = m;
		enable_selfcollision = flag;
	}
	mesh* ms;
	bool enable_selfcollision;
};


struct ImpactInfo
{
	ImpactInfo(int fid1, int fid2, int vf_ee, int v, int v2, int v3, int v4, double d, double t,int CCD)
	{
		f_id[0] = fid1;
		f_id[1] = fid2;

		IsVF_OR_EE = vf_ee;

		vertex_id[0] = v;
		vertex_id[1] = v2;
		vertex_id[2] = v3;
		vertex_id[3] = v4;

		dist = d;
		time = t;

		CCDres = CCD;
	}

	int f_id[2]; //√Ê∆¨id

	//0:vf  1:ee
	int IsVF_OR_EE;

	int vertex_id[4];

	double dist;
	double time;

	int CCDres;
};

#endif