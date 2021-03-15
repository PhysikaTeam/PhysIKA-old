#ifndef COLLID
#define COLLID

#include "CollisionDate.h"
#include"Collision.h"

//void body_collide(vector<mesh_pair> mpair, vector<CollisionDate> bodys, vector<vector<tri_pair>> &contacts);

void body_collide_gpu(vector<mesh_pair> mpair, vector<CollisionDate> bodys, vector<vector<tri_pair> > &contacts, int &CCDtime, vector<ImpactInfo> &contact_info, double thickness);

#endif
