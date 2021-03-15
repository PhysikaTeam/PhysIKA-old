#ifndef SELFCOLLISION
#define SELFCOLLISION

#include "cmesh.h"
#include "tmbvh.hpp"
#include "CollisionDate.h"

#include "iostream"
#include<vector>

using namespace std;

typedef pair<int, int> mesh_pair;

class Collision{
public:

	~Collision(){
		for (int i = 0; i < dl_mesh.size(); i++) {
				delete dl_mesh[i];
			}
	}

	//调用函数接口，调用内部碰撞检测算法
	void Collid();

	//碰撞对输入接口
	void Transform_Pair(unsigned int a,unsigned int b);
	
	//模型网格输入接口，输入模型网格的点集和面集
	void Transform_Mesh(unsigned int numVtx, unsigned int numTri, vector<unsigned int> tris, 
		vector<double> vtxs, 
		vector<double> pre_vtxs,
		int m_id,bool able_selfcollision=false
		);
	void Transform_Mesh(unsigned int numVtx, unsigned int numTri, vector<unsigned int> tris,
		vector<vec3f> vtxs,
		vector<vec3f> pre_vtxs,
		int m_id, bool able_selfcollision = false
	);

	//输出接口，返回发生碰撞的模型网格和三角形面片的集合
	vector<vector<tri_pair> > getContactPairs(){ return contact_pairs; }

	//输出接口，返回发生碰撞的碰撞对数量
	int getNumContacts(){ return contact_pairs.size(); }

	//输出接口，返回碰撞对发生碰撞的时间
	//vector<double> getContactTimes() { return contact_time; }

	//返回CCD结果，1：有穿透  0：无穿透
	int getCCD_res() { return CCDtime; }

	//设置厚度
	void setThickness(double tt) { thickness = tt; }

	//返回碰撞信息
	vector<ImpactInfo> getImpactInfo() { return contact_info; }

	static Collision* getInstance()			
	{
		if (instance == NULL) {
			instance = new Collision();
			return instance;
		}
		else
			return instance;
	}


	static Collision* instance;

	Collision():_is_first(true){}
private:
	vector<CollisionDate> bodys;     
	vector<mesh_pair> mesh_pairs;
	vector<vector<tri_pair> > contact_pairs;
	vector<mesh*> dl_mesh;//delete mesh points
	int CCDtime;

	vector<ImpactInfo> contact_info;

	double thickness;


	bool _is_first;//是否第一次传入网格

};


#endif
