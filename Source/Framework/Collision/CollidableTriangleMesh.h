#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/CollidableObject.h"
#include "Framework/Collision/CollisionBVH.h"
#include <vector>
#include <memory>
#include "Core/DataTypes.h"
#include "Dynamics/RigidBody/TriangleMesh.h"
#include<time.h>

namespace PhysIKA
{
	class TrianglePair;

	class bvh;
	template<typename TDataType>
	class CollidatableTriangleMesh :public CollidableObject {
		DECLARE_CLASS_1(CollidatableTriangleMesh, TDataType)
	public:
		typedef typename TDataType::Real Real;
		typedef typename TDataType::Coord Coord;

		CollidatableTriangleMesh();
		virtual ~CollidatableTriangleMesh();
		//static bvh *bvh1;
		static std::shared_ptr<bvh> bvh1;
		//static bvh *bvh2;
		static std::shared_ptr<bvh> bvh2;
		static bool checkCollision(std::shared_ptr<TriangleMesh<TDataType>> b1, std::shared_ptr<TriangleMesh<TDataType>> b2) {
			
			if (bvh1.get()== nullptr) {
				std::vector<std::shared_ptr<TriangleMesh<TDataType>>> meshes;
				meshes.push_back(b1);

				bvh1 = std::make_shared<bvh>(meshes);
				bvh2 = std::make_shared<bvh>(meshes);
			}

			std::vector<std::shared_ptr<TriangleMesh<TDataType>>> meshes1;
			meshes1.push_back(b1);
			std::vector<std::shared_ptr<TriangleMesh<TDataType>>> meshes2;
			meshes2.push_back(b2);
			clock_t start, end;
			start = clock();	
			//rebuild bvh
			bvh1->refit(meshes1);
			bvh2->refit(meshes2);
			end = clock();
			std::vector<TrianglePair> ret;

			bvh1.get()->collide(bvh2.get(), ret);
			if (ret.size()) {
				printf("check collision elapsed=%f ms.\n", (float)(end - start) * 1000 / CLOCKS_PER_SEC);
				printf("to checked ret size%d\n", ret.size());
			}
			for (size_t i = 0; i < ret.size(); i++) {
				TrianglePair &t = ret[i];
				unsigned int id0, id1;
				t.get(id0, id1);
				if (CollidableTriangle<DataType3f>::checkSelfIJ(b1.get(), id0, b2.get(), id1))
					return true;
			}
			
			//return (ret.size() > 0);
			return false;
		}

		bool initializeImpl() override;
		void updateCollidableObject() override;
		void updateMechanicalState() override;
	private:

	};
	class CollisionManager{
	public:
		CollisionManager() {};
		static std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> Meshes;

		static void mesh_id(int id, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &m, int &mid, int &fid) {
			fid = id;
			for (mid = 0; mid < m.size(); mid++)
				if (fid < m[mid]->_num_tri) {
					return;
				}
				else {
					fid -= m[mid]->_num_tri;
				}

			assert(false);
			fid = -1;
			mid = -1;
			printf("mesh_id error!!!!\n");
			abort();
		}
		static bool covertex(int id1, int id2) {
			if (Meshes.empty())
				return false;

			int mid1, fid1, mid2, fid2;

			mesh_id(id1, Meshes, mid1, fid1);
			mesh_id(id2, Meshes, mid2, fid2);

			if (mid1 != mid2)
				return false;

			TopologyModule::Triangle &f1 = Meshes[mid1]->triangleSet->getHTriangles()[fid1];
			TopologyModule::Triangle &f2 = Meshes[mid2]->triangleSet->getHTriangles()[fid2];

			for (int i = 0; i < 3; i++)
				for (int j = 0; j < 3; j++)
					if (f1[i] == f2[2])
						return true;

			return false;
		}
		static void self_mesh(std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &meshes)
		{
			Meshes = meshes;
		}
		
	};

}
