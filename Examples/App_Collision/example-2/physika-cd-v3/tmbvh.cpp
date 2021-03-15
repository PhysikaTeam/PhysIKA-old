#if defined(_WIN32)
#include <Windows.h>
#endif

//#include <GL/gl.h>

#include <stdlib.h>
#include <assert.h>
#include <stdio.h>

#include "tmbvh.hpp"
#include "cmesh.h"
#include <climits>
#include <utility>
using namespace std;

static vector<mesh *>  *ptCloth;

void self_mesh(vector<mesh *> &meshes)
{
	ptCloth = &meshes;
}

class aap {
public:
	char _xyz;
	float _p;

	FORCEINLINE aap(const BOX &total) {
		vec3f center = total.center();
		char xyz = 2;

		if (total.width() >= total.height() && total.width() >= total.depth()) {
			xyz = 0;
		} else
			if (total.height() >= total.width() && total.height() >= total.depth()) {
				xyz = 1;
			}

			_xyz = xyz;
			_p = center[xyz];
	}

	FORCEINLINE bool inside(const vec3f &mid) const {
		return mid[_xyz]>_p;
	}
};

bvh::bvh(std::vector<mesh*> &ms)
{
	_num = 0;
	_nodes = NULL;

	construct(ms);
	reorder();
	resetParents(); //update the parents after reorder ...
}

static vec3f *s_fcenters;
static BOX *s_fboxes;
static unsigned int *s_idx_buffer;
static bvh_node *s_current;

void bvh::construct(std::vector<mesh*> &ms)
{
	BOX total;

	for (int i=0; i<ms.size(); i++)
		for (int j = 0; j<ms[i]->_num_vtx; j++) {
			total += ms[i]->_vtxs[j];
		}

	_num = 0;
	for (int i=0; i<ms.size(); i++)
		_num += ms[i]->_num_tri;

	s_fcenters = new vec3f[_num];
	s_fboxes = new BOX[_num];

	int tri_idx = 0;
	int vtx_offset = 0;

	for (int i = 0; i < ms.size(); i++) {
		for (int j = 0; j < ms[i]->_num_tri; j++) {
			tri3f &f = ms[i]->_tris[j];
			vec3f &p1 = ms[i]->_vtxs[f.id0()];
			vec3f &p2 = ms[i]->_vtxs[f.id1()];
			vec3f &p3 = ms[i]->_vtxs[f.id2()];

			s_fboxes[tri_idx] += p1;
			s_fboxes[tri_idx] += p2;
			s_fboxes[tri_idx] += p3;

			s_fcenters[tri_idx] = (p1 + p2 + p3) / REAL(3.0);
			tri_idx++;
		}
		vtx_offset += ms[i]->_num_vtx;
	}

	aap pln(total);
	s_idx_buffer = new unsigned int[_num];
	unsigned int left_idx = 0, right_idx = _num;

	tri_idx = 0;
	for (int i=0; i<ms.size(); i++)
		for (int j = 0; j<ms[i]->_num_tri; j++) {
		if (pln.inside(s_fcenters[tri_idx]))
			s_idx_buffer[left_idx++] = tri_idx;
		else
			s_idx_buffer[--right_idx] = tri_idx;

		tri_idx++;
	}

	_nodes = new bvh_node[_num*2-1];
	_nodes[0]._box = total;
	s_current = _nodes+3;

	if (_num == 1)
		_nodes[0]._child = 0;
	else {
		_nodes[0]._child = -1;

		if (left_idx == 0 || left_idx == _num)
			left_idx = _num/2;

		_nodes[0].left()->construct(s_idx_buffer, left_idx);
		_nodes[0].right()->construct(s_idx_buffer+left_idx, _num-left_idx);
	}

	delete [] s_idx_buffer;
	delete [] s_fcenters;

	refit();
	//delete[] s_fboxes;
}

void bvh::refit(std::vector<mesh*> &ms)
{
	assert(s_fboxes);

	int tri_idx = 0;

	for (int i = 0; i < ms.size(); i++) {
		for (int j = 0; j < ms[i]->_num_tri; j++) {
			tri3f &f = ms[i]->_tris[j];
			vec3f &p1 = ms[i]->_vtxs[f.id0()];
			vec3f &p2 = ms[i]->_vtxs[f.id1()];
			vec3f &p3 = ms[i]->_vtxs[f.id2()];

			s_fboxes[tri_idx] = p1;
			s_fboxes[tri_idx] += p2;
			s_fboxes[tri_idx] += p3;

			tri_idx++;
		}
	}

	refit();
}

void bvh::resetParents()
{
	root()->resetParents(root());
}

void bvh::refit()
{
	root()->refit();
}

#include <queue>
using namespace std;

void bvh::reorder()
{
	if (true) 
	{
		queue<bvh_node *> q;

		// We need to perform a breadth-first traversal to fill the ids

		// the first pass get idx for each node ...
		int *buffer = new int[_num*2-1];
		int idx = 0;
		q.push(root());
		while (!q.empty()) {
			bvh_node *node = q.front();
			buffer[node-_nodes] = idx++;
			q.pop();

			if (!node->isLeaf()) {
				q.push(node->left());
				q.push(node->right());
			}
		}

		// the 2nd pass, get right nodes ...
		bvh_node *new_nodes = new bvh_node[_num*2-1];
		idx=0;
		q.push(root());
		while (!q.empty()) {
			bvh_node *node = q.front();
			q.pop();

			new_nodes[idx] = *node;
			if (!node->isLeaf()) {
				int loc = node->left()-_nodes;
				new_nodes[idx]._child = idx-buffer[loc];
			}
			idx++;

			if (!node->isLeaf()) {
				q.push(node->left());
				q.push(node->right());
			}
		}

		delete [] buffer;
		delete [] _nodes;
		_nodes = new_nodes;
	}
}

void
bvh_node::refit()
{
	if (isLeaf()) {
		_box = s_fboxes[_child];

	} else {
		left()->refit();
		right()->refit();

		_box = left()->_box + right()->_box;
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


void
bvh_node::construct(unsigned int id)
{
	_child = id;
	_box = s_fboxes[id];
}

void
bvh_node::construct(unsigned int *lst, unsigned int num)
{
	for (unsigned int i=0; i<num; i++)
		_box += s_fboxes[lst[i]];

	if (num == 1) {
		_child = lst[0];
		return;
	}

	// try to split them
	_child = int(this-s_current);
	s_current += 2;

	if (num == 2) {
		left()->construct(lst[0]);
		right()->construct(lst[1]);
		return;
	}

	aap pln(_box);
	unsigned int left_idx=0, right_idx=num-1;
	for (unsigned int t=0; t<num; t++) {
		int i=lst[left_idx];

		if (pln.inside( s_fcenters[i]))
			left_idx++;
		else {// swap it
			unsigned int tmp=lst[left_idx];
			lst[left_idx] = lst[right_idx];
			lst[right_idx--] = tmp;
		}
	}

	int half = num/2;

	if (left_idx == 0 || left_idx == num) {
		left()->construct(lst, half);
		right()->construct(lst+half, num-half);
	} else {
		left()->construct(lst, left_idx);
		right()->construct(lst+left_idx, num-left_idx);
	}
}


void
bvh_node::sprouting(bvh_node *other, front_list &append, vector<tri_pair> &ret)
{
	if (isLeaf() && other->isLeaf()) {

		if (!covertex(triID(), other->triID())) {
			append.push_back(front_node(this, other, 0));

			if (_box.overlaps(other->_box))
				ret.push_back(tri_pair(triID(), other->triID()));
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

void mesh_id(int id, vector<mesh *> &m, int &mid, int &fid)
{
	fid = id;
	for (mid=0; mid<m.size(); mid++)
		if (fid < m[mid]->_num_tri) {
			return;
		} else {
			fid -= m[mid]->_num_tri;
		}

	assert(false);
	fid = -1;
	mid = -1;
	printf("mesh_id error!!!!\n");
	abort();
}

bool covertex(int id1, int id2)
{
	if ((*ptCloth).empty())
		return false;

	int mid1, fid1, mid2, fid2;

	mesh_id(id1, *ptCloth, mid1, fid1);
	mesh_id(id2, *ptCloth, mid2, fid2);

	if (mid1 != mid2)
		return false;

	tri3f &f1 = (*ptCloth)[mid1]->_tris[fid1];
	tri3f &f2 = (*ptCloth)[mid2]->_tris[fid2];

	for (int i=0; i<3; i++)
		for (int j=0; j<3; j++)
			if (f1.id(i) == f2.id(2))
				return true;

	return false;
}

void
front_list::propogate(vector<mesh *> &c, vector<tri_pair> &ret)
{
	self_mesh(c);

	front_list append;

	for (vector<front_node>::iterator it = begin(); it != end(); it++) {
		(*it).update(append, ret);
	}

	insert(end(), append.begin(), append.end());
}

void front_node::update(front_list &append, vector<tri_pair> &ret)
 {
	 if (_flag != 0)
		 return;

	 if (_left->isLeaf() && _right->isLeaf()) {
		 if (!covertex(_left->triID(), _right->triID()) &&
			 _left->box().overlaps(_right->box()))
			 ret.push_back(tri_pair(_left->triID(), _right->triID()));

		 return;
	 }

	 if (!_left->box().overlaps(_right->box()))
		 return;

	 // need to be spouted
	 _flag = 1; // set to be invalid

	 if (_left->isLeaf()) {
		 _left->sprouting(_right->left(), append, ret);
		 _left->sprouting(_right->right(), append, ret);
	 } else {
		 _left->left()->sprouting(_right, append, ret);
		 _left->right()->sprouting(_right, append, ret);
	 }
 }


extern void refitBVH(bool);
extern void pushBVH(unsigned int length, int *ids, bool isCloth);
extern void pushBVHLeaf(unsigned int length, int *idf, bool isCloth);
extern void pushBVHIdx(int max_level, unsigned int *level_idx, bool isCloth);

void bvh::push2GPU(bool isCloth)
{
	unsigned int length = _num*2-1;
	int *ids = new int[length*2];

	for (unsigned int i=0; i<length; i++) {
		ids[i] = (root()+i)->triID();
		ids[length+i] = (root()+i)->parentID();
	}

	pushBVH(length, ids, isCloth);
	delete [] ids;

	unsigned int leafNum = 0;
	int *idf = new int[_num];
	for (unsigned int i = 0; i < length; i++) {
		if ((root() + i)->isLeaf()) {
			int idx = (root() + i)->triID();
			idf[idx] = i;
			leafNum++;
		}
	}
	assert(leafNum == _num);
	pushBVHLeaf(leafNum, idf, isCloth);
	delete []idf;

	{// push information for refit
		int max_level = 0;
		root()->getLevel(0, max_level);
		max_level++;

		unsigned int *level_idx = new unsigned int [max_level];
		unsigned int *level_buffer = new unsigned int [max_level];
		for (int i=0; i<max_level; i++)
			level_idx[i] = level_buffer[i] = 0;

		root()->getLevelIdx(0, level_buffer);
		for (int i=1; i<max_level; i++)
			for (int j=0; j<i; j++)
				level_idx[i] += level_buffer[j];

		delete [] level_buffer;
		pushBVHIdx(max_level, level_idx, isCloth);
		delete [] level_idx;
	}

	refitBVH(isCloth);
}

extern void pushFront(bool, int, unsigned int *);

void 
front_list::push2GPU(bvh_node *r1, bvh_node *r2 )
{
	bool self = (r2 == NULL);

	if (r2 == NULL)
		r2 = r1;

	int num = size();
	if (num) {
		int idx = 0;
		unsigned int *buffer = new unsigned int [num*4];
		for (vector<front_node>::iterator it=begin();
			it != end(); it++)
		{
			front_node n = *it;
			buffer[idx++] = n._left - r1;
			buffer[idx++] = n._right-r2;
			buffer[idx++] = 0;
			buffer[idx++] = n._ptr;
		}

		pushFront(self, num, buffer);
		printf("%d\n", num);
		delete [] buffer;
	} else
		pushFront(self, 0, NULL);
}


void
bvh_node::visualize(int level)
{
	/*
	if (isLeaf()) {
		_box.visualize();
	}
	else
		if ((level > 0)) {
			if (level == 1) {
				_box.visualize();
			}
			else {
				if (left()) left()->visualize(level - 1);
				if (right()) right()->visualize(level - 1);
			}
		}

		*/
}
