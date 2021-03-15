#include "Collision.h"
#include "collid.h"

Collision* Collision::instance = NULL;

void Collision::Transform_Pair(unsigned int a,unsigned int b) {
	if(_is_first ==true)
		mesh_pairs.push_back(mesh_pair(a, b));
}

void Collision::Transform_Mesh(unsigned int numVtx, unsigned int numTri, vector<unsigned int> tris, vector<double> vtxs,
	vector<double> p_vtxs,
	int m_id,
	bool able_selfcollision) {

	tri3f* _tris=new tri3f[numTri];
	vec3f* _vtxs = new vec3f[numVtx];
	vec3f* pre_vtxs = new vec3f[numVtx];

	for (int i = 0; i < numVtx; i++) {
		_vtxs[i] = vec3f(vtxs[i * 3], vtxs[i * 3 + 1], vtxs[i * 3 + 2]);
		pre_vtxs[i] = vec3f(p_vtxs[i * 3], p_vtxs[i * 3 + 1], p_vtxs[i * 3 + 2]);
	}

	for (int i = 0; i < numTri; i++) {
		_tris[i] = tri3f(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
	}
	
	mesh* m =new mesh(numVtx, numTri, _tris, _vtxs);

	for (int i = 0; i < numVtx; i++)
	{
		m->_ovtxs[i] = pre_vtxs[i];
	}

	dl_mesh.push_back(m);
	if (_is_first == true)
		bodys.push_back(CollisionDate(m, able_selfcollision));
	else
	{
		memcpy(bodys[m_id].ms->_ovtxs, pre_vtxs, numVtx * sizeof(vec3f));
		memcpy(bodys[m_id].ms->_vtxs, m->_vtxs, numVtx * sizeof(vec3f));
	}

	delete[] pre_vtxs;
}

void Collision::Transform_Mesh(unsigned int numVtx, unsigned int numTri, vector<unsigned int> tris,
	vector<vec3f> vtxs,
	vector<vec3f> p_vtxs,
	int m_id, bool able_selfcollision
) {
	tri3f* _tris = new tri3f[numTri];
	vec3f* _vtxs = new vec3f[numVtx];
	vec3f* pre_vtxs = new vec3f[numVtx];

	for (int i = 0; i < numVtx; i++) {
		_vtxs[i] = vtxs[i];
		pre_vtxs[i] = p_vtxs[i];
	}

	for (int i = 0; i < numTri; i++) {
		_tris[i] = tri3f(tris[i * 3], tris[i * 3 + 1], tris[i * 3 + 2]);
	}

	mesh* m = new mesh(numVtx, numTri, _tris, _vtxs);

	for (int i = 0; i < numVtx; i++)
	{
		m->_ovtxs[i] = pre_vtxs[i];
	}

	dl_mesh.push_back(m);
	if (_is_first == true)
		bodys.push_back(CollisionDate(m, able_selfcollision));
	else
	{
		memcpy(bodys[m_id].ms->_ovtxs, pre_vtxs, numVtx * sizeof(vec3f));
		memcpy(bodys[m_id].ms->_vtxs, m->_vtxs, numVtx * sizeof(vec3f));
	}

	delete[] pre_vtxs;
}

 void Collision::Collid(){
	contact_pairs.clear();
	//contact_time.clear();
	contact_info.clear();
	CCDtime = 0;

	body_collide_gpu(mesh_pairs, bodys, contact_pairs, CCDtime, contact_info, thickness);

	_is_first = false;
}