#include "CollisionBVH.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include <vector>
#include <queue>
using namespace std;
namespace PhysIKA {
	
	void bvh_node::setParent(int p) { _parent = p; }
	bvh_node::bvh_node() {
		_child = 0;
		_parent = 0;
	}
	bvh_node::~bvh_node() {
		NULL;
	}

	void bvh_node::collide(bvh_node *other, std::vector<TrianglePair> &ret) {
		if (!_box.overlaps(other->box())) {
			return;
		}
		
		if (isLeaf() && other->isLeaf()) {
			/*
			float * dbv0 = _box.v0.getDataPtr();
			float * dbv1 = _box.v1.getDataPtr();
			float * odbv0 = other->box().v0.getDataPtr();
			float * odbv1 = other->box().v1.getDataPtr();
			printf("self [%f %f %f], [%f %f %f] \n", dbv0[0], dbv0[1], dbv0[2], dbv1[0], dbv1[1], dbv1[2]);
			printf("other [%f %f %f], [%f %f %f] \n", odbv0[0], odbv0[1], odbv0[2], odbv1[0], odbv1[1], odbv1[2]);
			*/
			
			ret.push_back(TrianglePair(this->triID(), other->triID()));
			return;
		}

		if (isLeaf()) {
			assert(other->left() > other);
			assert(other->right() > other);
			collide(other->left(), ret);
			collide(other->right(), ret);
		}
		else {
			left()->collide(other, ret);
			right()->collide(other, ret);
		}
	}
	void  bvh_node::collide(bvh_node *other, front_list &f, int level, int ptr)
	{
		if (isLeaf() && other->isLeaf()) {
			if (!CollisionManager::covertex(this->triID(), other->triID()))
				f.push_back(front_node(this, other, ptr));

			return;
		}

		if (!_box.overlaps(other->box()) || level > 100) {
			f.push_back(front_node(this, other, ptr));
			return;
		}

		if (isLeaf()) {
			collide(other->left(), f, level++, ptr);
			collide(other->right(), f, level++, ptr);
		}
		else {
			left()->collide(other, f, level++, ptr);
			right()->collide(other, f, level++, ptr);
		}
	}
	void bvh_node::self_collide(front_list &lst, bvh_node *r) {
		if (isLeaf())
			return;

		left()->self_collide(lst, r);
		right()->self_collide(lst, r);
		left()->collide(right(), lst, 0, this - r);
	}
	FORCEINLINE void bvh_node::getLevel(int current, int &max_level) {
		if (current > max_level)
			max_level = current;

		if (isLeaf()) return;
		left()->getLevel(current + 1, max_level);
		right()->getLevel(current + 1, max_level);
	}
	FORCEINLINE void bvh_node::getLevelIdx(int current, unsigned int *idx) {
		idx[current]++;

		if (isLeaf()) return;
		left()->getLevelIdx(current + 1, idx);
		right()->getLevelIdx(current + 1, idx);
	}
	void front_node::update(front_list &append, vector<TrianglePair> &ret)
	{
		if (_flag != 0)
			return;

		if (_left->isLeaf() && _right->isLeaf()) {
			if (!CollisionManager::covertex(_left->triID(), _right->triID()) &&
				_left->box().overlaps(_right->box()))
				ret.push_back(TrianglePair(_left->triID(), _right->triID()));

			return;
		}

		if (!_left->box().overlaps(_right->box()))
			return;

		// need to be spouted
		_flag = 1; // set to be invalid

		if (_left->isLeaf()) {
			_left->sprouting(_right->left(), append, ret);
			_left->sprouting(_right->right(), append, ret);
		}
		else {
			_left->left()->sprouting(_right, append, ret);
			_left->right()->sprouting(_right, append, ret);
		}
	}
	front_node::front_node(bvh_node *l, bvh_node *r, unsigned int ptr) {
		_left = l, _right = r, _flag = 0;
		_ptr = ptr;
	}

	void bvh::collide(bvh *other, front_list &f) {
		f.clear();

		std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> c;
		CollisionManager::self_mesh(c);

		if (other)
			root()->collide(other->root(), f, 0, -1);
	}
	void bvh::collide(bvh *other, std::vector<TrianglePair> &ret)
	{
		root()->collide(other->root(), ret);
	}
	void bvh::self_collide(front_list &f, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &c) {
		f.clear();

		CollisionManager::self_mesh(c);
		root()->self_collide(f, root());
	}
	void
		bvh_node::construct(unsigned int id, TAlignedBox3D<float>*s_fboxes)
	{
		_child = id;
		_box = s_fboxes[id];
	}
	void
		bvh_node::construct(unsigned int *lst, unsigned int num, vec3f *s_fcenters, TAlignedBox3D<float>*s_fboxes, bvh_node* &s_current)
	{
		for (unsigned int i = 0; i < num; i++)
			_box += s_fboxes[lst[i]];

		if (num == 1) {
			//s_current += 1;
			_child = lst[0];
			return;
		}

		// try to split them
		_child = int(((long long)this - (long long) s_current )/ sizeof(bvh_node));
		s_current += 2;

		if (num == 2) {
			left()->construct(lst[0], s_fboxes);
			right()->construct(lst[1], s_fboxes);
			return;
		}

		aap pln(_box);
		unsigned int left_idx = 0, right_idx = num - 1;
		for (unsigned int t = 0; t < num; t++) {
			int i = lst[left_idx];

			if (pln.inside(s_fcenters[i]))
				left_idx++;
			else {// swap it
				unsigned int tmp = lst[left_idx];
				lst[left_idx] = lst[right_idx];
				lst[right_idx--] = tmp;
			}
		}

		int half = num / 2;

		if (left_idx == 0 || left_idx == num) {
			left()->construct(lst, half, s_fcenters, s_fboxes, s_current);
			right()->construct(lst + half, num - half, s_fcenters, s_fboxes, s_current);
		}
		else {
			left()->construct(lst, left_idx, s_fcenters, s_fboxes, s_current);
			right()->construct(lst + left_idx, num - left_idx, s_fcenters, s_fboxes, s_current);
		}
	}
	void
		bvh_node::refit(TAlignedBox3D<float>*s_fboxes)
	{
		if (isLeaf()) {
			_box = s_fboxes[_child];

		}
		else {
			left()->refit(s_fboxes);
			right()->refit(s_fboxes);

			_box = left()->_box + right()->_box;
		}
	}
	void bvh::refit(TAlignedBox3D<float>*s_fboxes)
	{
		root()->refit(s_fboxes);
	}

	void bvh::reorder()
	{
		if (true)
		{
			std::queue<bvh_node *> q;

			// We need to perform a breadth-first traversal to fill the ids

			// the first pass get idx for each node ...
			int *buffer = new int[_num * 2 - 1];
			int idx = 0;
			q.push(root());
			while (!q.empty()) {
				bvh_node *node = q.front();
				//int(((long long)node->left() - (long long) _nodes )/ sizeof(bvh_node))
				buffer[((long long)node - (long long)_nodes) / sizeof(bvh_node)] = idx++;
				q.pop();

				if (!node->isLeaf()) {
					q.push(node->left());
					q.push(node->right());
				}
			}

			// the 2nd pass, get right nodes ...
			bvh_node *new_nodes = new bvh_node[_num * 2 - 1];
			idx = 0;
			q.push(root());
			while (!q.empty()) {
				bvh_node *node = q.front();
				q.pop();

				new_nodes[idx] = *node;
				if (!node->isLeaf()) {
					int loc = int(((long long)node->left() - (long long)_nodes) / sizeof(bvh_node));
					new_nodes[idx]._child = idx - buffer[loc];
				}
				idx++;

				if (!node->isLeaf()) {
					q.push(node->left());
					q.push(node->right());
				}
			}

			delete[] buffer;
			delete[] _nodes;
			_nodes = new_nodes;
		}
	}
	void
		bvh_node::resetParents(bvh_node *root)
	{
		if (this == root)
			setParent(-1);

		if (isLeaf())
			return;

		left()->resetParents(root);
		right()->resetParents(root);

		left()->setParent(this - root);
		right()->setParent(this - root);
	}
	void bvh::resetParents()
	{
		root()->resetParents(root());
	}
	void bvh::refit(std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &ms)
	{
		assert(s_fboxes);

		int tri_idx = 0;

		for (int i = 0; i < ms.size(); i++) {
			for (int j = 0; j < ms[i]->_num_tri; j++) {
				TopologyModule::Triangle &f = ms[i]->triangleSet->getHTriangles()[j];
				Vector3f &p1 = ms[i]->triangleSet->gethPoints()[f[0]];
				Vector3f &p2 = ms[i]->triangleSet->gethPoints()[f[1]];
				Vector3f &p3 = ms[i]->triangleSet->gethPoints()[f[2]];

				*&s_fboxes[tri_idx] = p1;
				*&s_fboxes[tri_idx] += p2;
				*&s_fboxes[tri_idx] += p3;

				tri_idx++;
			}
		}

		refit(s_fboxes);
	}
	void
		bvh_node::sprouting(bvh_node *other, front_list &append, vector<TrianglePair> &ret)
	{
		if (isLeaf() && other->isLeaf()) {

			if (!CollisionManager::covertex(triID(), other->triID())) {
				append.push_back(front_node(this, other, 0));

				if (_box.overlaps(other->_box))
					ret.push_back(TrianglePair(triID(), other->triID()));
			}

			return;
		}

		if (!_box.overlaps(other->_box)) {
			append.push_back(front_node(this, other, 0));
			return;
		}

		if (isLeaf()) {
			sprouting(other->left(), append, ret);
			sprouting(other->right(), append, ret);
		}
		else {
			left()->sprouting(other, append, ret);
			right()->sprouting(other, append, ret);
		}
	}
}