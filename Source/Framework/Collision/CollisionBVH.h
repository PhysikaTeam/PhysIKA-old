#pragma once

#include <vector>
#include <memory>
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Framework/Collision/CollidableTriangle.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include "Framework/Framework/ModuleTopology.h"
#include "CollisionMesh.h"

namespace PhysIKA {
	class front_node;
	class bvh_node;
	
	static vec3f *s_fcenters;
	class front_list : public std::vector<front_node> {
	public:
		void push2GPU(bvh_node *r1, bvh_node *r2 = NULL);
	};

	class bvh_node {
		TAlignedBox3D<float> _box;
		static bvh_node* s_current; // hyx
		int _child; // >=0 leaf with tri_id, <0 left & right
		int _parent;

		void setParent(int p);

	public:
		bvh_node();

		~bvh_node();

		void collide(bvh_node *other, std::vector<TrianglePair> &ret); // hyx

		void collide(bvh_node *other, front_list &f, int level, int ptr);

		void self_collide(front_list &lst, bvh_node *r);
		
		void construct(unsigned int id, TAlignedBox3D<float>* s_fboxes);
		
		void construct(
			unsigned int *lst, 
			unsigned int num, vec3f *s_fcenters, 
			TAlignedBox3D<float>* s_fboxes,
			bvh_node*& s_current);

		void refit(TAlignedBox3D<float> *s_fboxes);

		void resetParents(bvh_node *root);

		FORCEINLINE TAlignedBox3D<float> &box() { return _box; }
		FORCEINLINE bvh_node *left() { return this - _child; }
		FORCEINLINE bvh_node *right() { return this - _child + 1; }
		FORCEINLINE int triID() { return _child; }
		FORCEINLINE int isLeaf() { return _child >= 0; }
		FORCEINLINE int parentID() { return _parent; }
		FORCEINLINE void getLevel(int current, int &max_level);
		FORCEINLINE void getLevelIdx(int current, unsigned int *idx);

		void sprouting(bvh_node* other, front_list& append, std::vector<TrianglePair>& ret);
		void sprouting2(bvh_node* other, front_list& append, std::vector<TrianglePair>& ret); // hyx

		friend class bvh;
	};

	class front_node {
	public:
		bvh_node *_left, *_right;
		unsigned int _flag; // vailid or not
		unsigned int _ptr; // self-coliding parent;

		front_node(bvh_node *l, bvh_node *r, unsigned int ptr);
	};
	
	class aap {
	public:
		int _xyz;
		float _p;

		FORCEINLINE aap(const TAlignedBox3D<float> &total) {
			vec3f center =vec3f( total.center().getDataPtr());
			int xyz = 2;

			if (total.width() >= total.height() && total.width() >= total.depth()) {
				xyz = 0;
			}
			else
				if (total.height() >= total.width() && total.height() >= total.depth()) {
					xyz = 1;
				}

			_xyz = xyz;
			_p = center[xyz];
		}

		inline bool inside(const vec3f &mid) const {
			return mid[_xyz] > _p;
		}
	};
	
	class bvh {
		int _num = 0; // all face num
		bvh_node *_nodes = nullptr;
		TAlignedBox3D<float>* s_fboxes = nullptr;

		unsigned int *s_idx_buffer = nullptr;

		//static vec3f* s_fcenters;

	public:
		bvh() {};
		
		template<typename T>
		bvh(std::vector<std::shared_ptr<TriangleMesh<T>>> &ms) {
			_num = 0;
			_nodes = NULL;

			construct<T>(ms);
			reorder();
			resetParents(); //update the parents after reorder ...
		} // hxl
		
		bvh(const std::vector<CollisionMesh*>& ms);
		
		void refit(TAlignedBox3D<float>*s_fboxes);

		template<typename T>
		void construct(std::vector<std::shared_ptr<TriangleMesh<T>>> &ms) {
			TAlignedBox3D<float> total;

			for (int i = 0; i < ms.size(); i++)
				for (int j = 0; j < ms[i]->_num_vtx; j++) {
					total += ms[i]->triangleSet->gethPoints()[j];
				}

			_num = 0;
			for (int i = 0; i < ms.size(); i++)
				_num += ms[i]->_num_tri;

			s_fcenters = new vec3f[_num];
			s_fboxes = new TAlignedBox3D<float>[_num];

			int tri_idx = 0;
			int vtx_offset = 0;

			for (int i = 0; i < ms.size(); i++) {
				for (int j = 0; j < ms[i]->_num_tri; j++) {
					TopologyModule::Triangle &f = ms[i]->triangleSet->getHTriangles()[j];
					Vector3f &p1 = ms[i]->triangleSet->gethPoints()[f[0]];
					Vector3f &p2 = ms[i]->triangleSet->gethPoints()[f[1]];
					Vector3f &p3 = ms[i]->triangleSet->gethPoints()[f[2]];

					s_fboxes[tri_idx] += p1;
					s_fboxes[tri_idx] += p2;
					s_fboxes[tri_idx] += p3;

					auto _s = p1 + p2 + p3;
					auto sum = _s.getDataPtr();
					s_fcenters[tri_idx] = vec3f(sum);
					s_fcenters[tri_idx] /= 3.0;
					//s_fcenters[tri_idx] = (p1 + p2 + p3) / double(3.0);
					tri_idx++;
				}
				vtx_offset += ms[i]->_num_vtx;
			}

			aap pln(total);
			s_idx_buffer = new unsigned int[_num];
			unsigned int left_idx = 0, right_idx = _num;

			tri_idx = 0;
			for (int i = 0; i < ms.size(); i++)
				for (int j = 0; j < ms[i]->_num_tri; j++) {
					if (pln.inside(s_fcenters[tri_idx]))
						s_idx_buffer[left_idx++] = tri_idx;
					else
						s_idx_buffer[--right_idx] = tri_idx;

					tri_idx++;
				}

			_nodes = new bvh_node[_num * 2 - 1];
			
			_nodes[0]._box = total;
			bvh_node *s_current = _nodes + 3;
			if (_num == 1)
				_nodes[0]._child = 0;
			else {
				_nodes[0]._child = -1;

				if (left_idx == 0 || left_idx == _num)
					left_idx = _num / 2;
				_nodes[0].left()->construct(s_idx_buffer, left_idx, s_fcenters, s_fboxes, s_current);
				_nodes[0].right()->construct(s_idx_buffer + left_idx, _num - left_idx, s_fcenters, s_fboxes, s_current);
			}

			delete[] s_idx_buffer;
			s_idx_buffer = nullptr;
			delete[] s_fcenters;
			s_fcenters = nullptr;

			refit(s_fboxes);
			//delete[] s_fboxes;
		}

		void reorder(); // for breath-first refit

		void resetParents();
		
		~bvh() {
			if (_nodes)
				delete[] _nodes;
		}

		bvh_node *root() { return _nodes; }

		void refit(std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &ms);

		void push2GPU(bool);

		void collide(bvh *other, front_list &f);

		void collide(bvh *other, std::vector<TrianglePair> &ret);

		void self_collide(front_list &f, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &c); // hxl

		void self_collide(front_list& f, std::vector<CollisionMesh*>& c);
	};
}