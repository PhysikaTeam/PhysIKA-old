#pragma once

#include <float.h>
#include <stdlib.h>
#include "real.hpp"
#include "vec3f.h"
#include "cmesh.h"
#include "box.h"

#include <vector>
using namespace std;

#define MAX(a,b)	((a) > (b) ? (a) : (b))
#define MIN(a,b)	((a) < (b) ? (a) : (b))

class bvh;
class bvh_node;
class front_list;


class tri_pair {
	unsigned int _id[2];

public:
	unsigned int id0() const { return _id[0]; }
	unsigned int id1() const { return _id[1]; }

	tri_pair(unsigned int id1, unsigned int id2)
	{
			_id[0] = id1;
			_id[1] = id2;
	}

	void get(unsigned int &id1, unsigned int &id2)
	{
		id1 = _id[0];
		id2 = _id[1];
	}

	bool operator < (const tri_pair &other) const {
		if (_id[0] == other._id[0])
			return _id[1] < other._id[1];
		else
			return _id[0] < other._id[0];
	}
};


class front_node {
public:
	bvh_node *_left, *_right;
	unsigned int _flag; // vailid or not
	unsigned int _ptr; // self-coliding parent;

	FORCEINLINE front_node(bvh_node *l, bvh_node *r, unsigned int ptr)
	{
		_left = l, _right = r, _flag = 0;
		_ptr = ptr;
	}

	void update(front_list &appended, vector<tri_pair> &ret);
};


bool covertex(int tri1, int tri2);
void self_mesh(vector<mesh *> &);

class front_list : public vector<front_node> {
public:
	void propogate(vector<mesh *> &c, vector<tri_pair> &ret);
	void push2GPU(bvh_node *r1, bvh_node *r2 = NULL);
};

class bvh_node {
	BOX _box;
	int _child; // >=0 leaf with tri_id, <0 left & right
	int _parent;

	void setParent(int p) { _parent = p; }

public:
	bvh_node() {
		_child = 0;
		_parent = 0;
	}

	~bvh_node() {
		NULL;
	}

	void collide(bvh_node *other, front_list &f, int level, int ptr)
	{
		if (isLeaf() && other->isLeaf()) {
			if (!covertex(this->triID(), other->triID()) )
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
		} else {
			left()->collide(other, f, level++, ptr);
			right()->collide(other, f, level++, ptr);
		}
	}

	void self_collide(front_list &lst, bvh_node *r) {
		if (isLeaf())
			return;

		left()->self_collide(lst, r);
		right()->self_collide(lst, r);
		left()->collide(right(), lst, 0, this-r);
	}

	void construct(unsigned int id);
	void construct(unsigned int *lst, unsigned int num);

	void visualize(int level);
	void refit();
	void resetParents(bvh_node *root);

	FORCEINLINE BOX &box() { return _box; }
	FORCEINLINE bvh_node *left() { return this - _child; }
	FORCEINLINE bvh_node *right() { return this - _child + 1; }
	FORCEINLINE int triID() { return _child; }
	FORCEINLINE int isLeaf() { return _child >= 0; }
	FORCEINLINE int parentID() { return _parent; }

	FORCEINLINE void getLevel(int current, int &max_level) {
		if (current > max_level)
			max_level = current;

		if (isLeaf()) return;
		left()->getLevel(current+1, max_level);
		right()->getLevel(current+1, max_level);
	}

	FORCEINLINE void getLevelIdx(int current, unsigned int *idx) {
		idx[current]++;

		if (isLeaf()) return;
		left()->getLevelIdx(current+1, idx);
		right()->getLevelIdx(current+1, idx);
	}

	void sprouting(bvh_node *other, front_list &append, vector<tri_pair> &ret);

	friend class bvh;
};

class mesh;

class bvh {
	int _num; // all face num
	bvh_node *_nodes;

	void construct(std::vector<mesh*> &);
	void refit();
	void reorder(); // for breath-first refit
	void resetParents();

public:
	bvh(std::vector<mesh*> &ms);

	~bvh() {
		if (_nodes)
			delete [] _nodes;
	}
	
	bvh_node *root() { return _nodes; }

	void refit(std::vector<mesh*> &ms);

	void push2GPU(bool);

	void collide(bvh *other, front_list &f) {
		f.clear();

		vector<mesh *> c;
		self_mesh(c);
		
		if (other)
		root()->collide(other->root(), f, 0, -1);
	}

	void self_collide(front_list &f, vector<mesh *> &c) {
		f.clear();

		self_mesh(c);
		root()->self_collide(f, root());
	}

	void visualize(int);
};