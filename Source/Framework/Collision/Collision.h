#pragma once

#include "Dynamics/RigidBody/TriangleMesh.h"
#include "CollidableTriangle.h"
#include "CollisionMesh.h"
#include "CollisionBVH.h"
#include "CollisionDate.h"

#include <iostream>
#include <vector>


namespace PhysIKA {
	class Collision {
	public:
		using MeshPair = std::pair<int, int>;


		~Collision() {
			for (int i = 0; i < dl_mesh.size(); i++) {
				delete dl_mesh[i];
			}
		}

		//调用函数接口，调用内部碰撞检测算法
		void collid();

		//碰撞对输入接口
		void transformPair(unsigned int a, unsigned int b);

		//模型网格输入接口，输入模型网格的点集和面集
		void transformMesh(unsigned int numVtx, unsigned int numTri, 
			std::vector<unsigned int> tris,
			std::vector<float> vtxs,
			std::vector<float> pre_vtxs,
			int m_id, bool able_selfcollision = false
		);
		void transformMesh(unsigned int numVtx, unsigned int numTri, 
			std::vector<unsigned int> tris,
			std::vector<vec3f> vtxs,
			std::vector<vec3f> pre_vtxs,
			int m_id, bool able_selfcollision = false
		);
		void transformMesh(TriangleMesh<DataType3f> mesh,
			int m_id, bool able_selfcollision = false
		);

		//输出接口，返回发生碰撞的模型网格和三角形面片的集合
		std::vector<std::vector<TrianglePair> > getContactPairs() { return contact_pairs; }

		//输出接口，返回发生碰撞的碰撞对数量
		int getNumContacts() { return contact_pairs.size(); }

		//输出接口，返回碰撞对发生碰撞的时间
		//vector<float> getContactTimes() { return contact_time; }

		//返回CCD结果，1：有穿透  0：无穿透
		int getCCD_res() { return CCDtime; }

		//设置厚度
		void setThickness(float tt) { thickness = tt; }

		//返回碰撞信息
		std::vector<ImpactInfo> getImpactInfo() { return contact_info; }

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

	private:
		Collision() = default;

	private:
		std::vector<CollisionDate> bodys;
		std::vector<MeshPair> mesh_pairs;
		std::vector<std::vector<TrianglePair>> contact_pairs;
		std::vector<CollisionMesh*> dl_mesh;//delete mesh points
		std::vector<ImpactInfo> contact_info;
		int CCDtime = 0;
		float thickness = 0.0f;
	};
}