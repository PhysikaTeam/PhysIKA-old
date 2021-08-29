#include "CollisionBVH.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include <vector>
#include <queue>
using namespace std;

extern void refitBVH(bool);
extern void pushBVH(unsigned int length, int* ids, bool isCloth);
extern void pushBVHLeaf(unsigned int length, int* idf, bool isCloth);
extern void pushBVHIdx(int max_level, unsigned int* level_idx, bool isCloth);
extern void pushFront(bool, int, unsigned int*);

static vector<PhysIKA::CollisionMesh*>* ptCloth;

static void mesh_id(int id, vector<PhysIKA::CollisionMesh*>& m, int& mid, int& fid)
{
    fid = id;
    for (mid = 0; mid < m.size(); mid++)
        if (fid < m[mid]->_num_tri)
        {
            return;
        }
        else
        {
            fid -= m[mid]->_num_tri;
        }

    assert(false);
    fid = -1;
    mid = -1;
    printf("mesh_id error!!!!\n");
    abort();
}

static bool covertex(int id1, int id2)
{

    std::cout << "covertex" << std::endl;
    if ((*ptCloth).empty())
        return false;

    int mid1, fid1, mid2, fid2;

    mesh_id(id1, *ptCloth, mid1, fid1);
    mesh_id(id2, *ptCloth, mid2, fid2);

    if (mid1 != mid2)
        return false;

    PhysIKA::CollisionMesh::tri3f& f1 = (*ptCloth)[mid1]->_tris[fid1];
    PhysIKA::CollisionMesh::tri3f& f2 = (*ptCloth)[mid2]->_tris[fid2];

    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (f1.id(i) == f2.id(2))
                return true;

    return false;
}

namespace PhysIKA {
void bvh_node::setParent(int p)
{
    _parent = p;
}
bvh_node::bvh_node()
{
    _child  = 0;
    _parent = 0;
}
bvh_node::~bvh_node()
{
    NULL;
}

void bvh_node::collide(bvh_node* other, std::vector<TrianglePair>& ret)
{
    if (!_box.overlaps(other->box()))
    {
        return;
    }

    if (isLeaf() && other->isLeaf())
    {
        ret.push_back(TrianglePair(this->triID(), other->triID()));
        return;
    }

    if (isLeaf())
    {
        assert(other->left() > other);
        assert(other->right() > other);
        collide(other->left(), ret);
        collide(other->right(), ret);
    }
    else
    {
        left()->collide(other, ret);
        right()->collide(other, ret);
    }
}

void bvh_node::collide(bvh_node* other, front_list& f, int level, int ptr)
{
    if (isLeaf() && other->isLeaf())
    {
        if (!CollisionManager::covertex(this->triID(), other->triID()))
            f.push_back(front_node(this, other, ptr));

        return;
    }

    if (!_box.overlaps(other->box()) || level > 100)
    {
        f.push_back(front_node(this, other, ptr));
        return;
    }

    if (isLeaf())
    {
        collide(other->left(), f, level++, ptr);
        collide(other->right(), f, level++, ptr);
    }
    else
    {
        left()->collide(other, f, level++, ptr);
        right()->collide(other, f, level++, ptr);
    }
}

void bvh_node::self_collide(front_list& lst, bvh_node* r)
{
    if (isLeaf())
        return;

    left()->self_collide(lst, r);
    right()->self_collide(lst, r);
    left()->collide(right(), lst, 0, this - r);
}

FORCEINLINE void bvh_node::getLevel(int current, int& max_level)
{
    if (current > max_level)
        max_level = current;

    if (isLeaf())
        return;
    left()->getLevel(current + 1, max_level);
    right()->getLevel(current + 1, max_level);
}

FORCEINLINE void bvh_node::getLevelIdx(int current, unsigned int* idx)
{
    idx[current]++;

    if (isLeaf())
        return;
    left()->getLevelIdx(current + 1, idx);
    right()->getLevelIdx(current + 1, idx);
}

front_node::front_node(bvh_node* l, bvh_node* r, unsigned int ptr)
{
    _left = l, _right = r, _flag = 0;
    _ptr = ptr;
}

void bvh::collide(bvh* other, front_list& f)
{
    f.clear();

    std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> c;
    CollisionManager::self_mesh(c);

    if (other)
        root()->collide(other->root(), f, 0, -1);
}

void bvh::collide(bvh* other, std::vector<TrianglePair>& ret)
{
    root()->collide(other->root(), ret);
}

void bvh::self_collide(front_list& f, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>>& c)
{
    f.clear();

    CollisionManager::self_mesh(c);
    root()->self_collide(f, root());
}

void bvh_node::construct(unsigned int id, TAlignedBox3D<float>* s_fboxes)
{
    _child = id;
    _box   = s_fboxes[id];
}

void bvh_node::construct(unsigned int* lst, unsigned int num, vec3f* s_fcenters, TAlignedBox3D<float>* s_fboxes, bvh_node*& s_current)
{
    for (unsigned int i = 0; i < num; i++)
        _box += s_fboxes[lst[i]];

    if (num == 1)
    {
        //s_current += 1;
        _child = lst[0];
        return;
    }

    // try to split them
    _child = int((( long long )this - ( long long )s_current) / sizeof(bvh_node));
    s_current += 2;

    if (num == 2)
    {
        left()->construct(lst[0], s_fboxes);
        right()->construct(lst[1], s_fboxes);
        return;
    }

    aap          pln(_box);
    unsigned int left_idx = 0, right_idx = num - 1;
    for (unsigned int t = 0; t < num; t++)
    {
        int i = lst[left_idx];

        if (pln.inside(s_fcenters[i]))
            left_idx++;
        else
        {  // swap it
            unsigned int tmp = lst[left_idx];
            lst[left_idx]    = lst[right_idx];
            lst[right_idx--] = tmp;
        }
    }

    int half = num / 2;

    if (left_idx == 0 || left_idx == num)
    {
        left()->construct(lst, half, s_fcenters, s_fboxes, s_current);
        right()->construct(lst + half, num - half, s_fcenters, s_fboxes, s_current);
    }
    else
    {
        left()->construct(lst, left_idx, s_fcenters, s_fboxes, s_current);
        right()->construct(lst + left_idx, num - left_idx, s_fcenters, s_fboxes, s_current);
    }
}

void bvh_node::refit(TAlignedBox3D<float>* s_fboxes)
{
    if (isLeaf())
    {
        _box = s_fboxes[_child];
    }
    else
    {
        left()->refit(s_fboxes);
        right()->refit(s_fboxes);

        _box = left()->_box + right()->_box;
    }
}

void bvh::refit(TAlignedBox3D<float>* s_fboxes)
{
    root()->refit(s_fboxes);
}

void bvh::reorder()
{
    if (true)
    {
        std::queue<bvh_node*> q;

        // We need to perform a breadth-first traversal to fill the ids

        // the first pass get idx for each node ...
        int* buffer = new int[_num * 2 - 1];
        int  idx    = 0;
        q.push(root());
        while (!q.empty())
        {
            bvh_node* node = q.front();
            //int(((long long)node->left() - (long long) _nodes )/ sizeof(bvh_node))
            buffer[(( long long )node - ( long long )_nodes) / sizeof(bvh_node)] = idx++;
            q.pop();

            if (!node->isLeaf())
            {
                q.push(node->left());
                q.push(node->right());
            }
        }

        // the 2nd pass, get right nodes ...
        bvh_node* new_nodes = new bvh_node[_num * 2 - 1];
        idx                 = 0;
        q.push(root());
        while (!q.empty())
        {
            bvh_node* node = q.front();
            q.pop();

            new_nodes[idx] = *node;
            if (!node->isLeaf())
            {
                int loc               = int((( long long )node->left() - ( long long )_nodes) / sizeof(bvh_node));
                new_nodes[idx]._child = idx - buffer[loc];
            }
            idx++;

            if (!node->isLeaf())
            {
                q.push(node->left());
                q.push(node->right());
            }
        }

        delete[] buffer;
        delete[] _nodes;
        _nodes = new_nodes;
    }
}

void bvh_node::resetParents(bvh_node* root)
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

void bvh::refit(std::vector<std::shared_ptr<TriangleMesh<DataType3f>>>& ms)
{
    assert(s_fboxes);

    int tri_idx = 0;

    for (int i = 0; i < ms.size(); i++)
    {
        for (int j = 0; j < ms[i]->_num_tri; j++)
        {
            TopologyModule::Triangle& f  = ms[i]->triangleSet->getHTriangles()[j];
            Vector3f&                 p1 = ms[i]->triangleSet->gethPoints()[f[0]];
            Vector3f&                 p2 = ms[i]->triangleSet->gethPoints()[f[1]];
            Vector3f&                 p3 = ms[i]->triangleSet->gethPoints()[f[2]];

            *&s_fboxes[tri_idx] = p1;
            *&s_fboxes[tri_idx] += p2;
            *&s_fboxes[tri_idx] += p3;

            tri_idx++;
        }
    }

    refit(s_fboxes);
}

void bvh_node::sprouting2(bvh_node* other, front_list& append, vector<TrianglePair>& ret)
{
    if (isLeaf() && other->isLeaf())
    {

        if (!CollisionManager::covertex(triID(), other->triID()))
        {
            append.push_back(front_node(this, other, 0));

            if (_box.overlaps(other->_box))
                ret.push_back(TrianglePair(triID(), other->triID()));
        }

        return;
    }

    if (!_box.overlaps(other->_box))
    {
        append.push_back(front_node(this, other, 0));
        return;
    }

    if (isLeaf())
    {
        sprouting2(other->left(), append, ret);
        sprouting2(other->right(), append, ret);
    }
    else
    {
        left()->sprouting2(other, append, ret);
        right()->sprouting2(other, append, ret);
    }
}

void bvh_node::sprouting(bvh_node* other, front_list& append, std::vector<TrianglePair>& ret)
{
    if (isLeaf() && other->isLeaf())
    {

        if (!covertex(triID(), other->triID()))
        {
            append.push_back(front_node(this, other, 0));

            if (_box.overlaps(other->_box))
                ret.push_back(TrianglePair(triID(), other->triID()));
        }

        return;
    }

    if (!_box.overlaps(other->_box))
    {
        append.push_back(front_node(this, other, 0));
        return;
    }

    if (isLeaf())
    {
        sprouting(other->left(), append, ret);
        sprouting(other->right(), append, ret);
    }
    else
    {
        left()->sprouting(other, append, ret);
        right()->sprouting(other, append, ret);
    }
}

void bvh::self_collide(front_list& f, std::vector<CollisionMesh*>& c)
{
    f.clear();

    ptCloth = &c;
    root()->self_collide(f, root());
}

bvh::bvh(const std::vector<CollisionMesh*>& ms)
{
    _num   = 0;
    _nodes = nullptr;

    TAlignedBox3D<float> total;

    for (int i = 0; i < ms.size(); i++)
    {
        for (int j = 0; j < ms[i]->_num_vtx; j++)
        {
            total += Vector3f(ms[i]->_vtxs[j].x, ms[i]->_vtxs[j].y, ms[i]->_vtxs[j].z);
        }
    }

    _num = 0;
    for (int i = 0; i < ms.size(); i++)
    {
        _num += ms[i]->_num_tri;
        std::cout << ms[i]->_num_tri << std::endl;
    }

    std::cout << _num << std::endl;

    s_fcenters = new vec3f[_num];
    s_fboxes   = new TAlignedBox3D<float>[_num];

    int tri_idx    = 0;
    int vtx_offset = 0;

    for (int i = 0; i < ms.size(); i++)
    {
        for (int j = 0; j < ms[i]->_num_tri; j++)
        {
            const auto& f  = ms[i]->_tris[j];
            vec3f&      p1 = ms[i]->_vtxs[f.id0()];
            vec3f&      p2 = ms[i]->_vtxs[f.id1()];
            vec3f&      p3 = ms[i]->_vtxs[f.id2()];

            //printf("%d, %d, %d\n", f.id0(), f.id1(), f.id2());

            s_fboxes[tri_idx] += Vector3f(p1.x, p1.y, p1.z);
            s_fboxes[tri_idx] += Vector3f(p2.x, p2.y, p2.z);
            s_fboxes[tri_idx] += Vector3f(p3.x, p3.y, p3.z);

            s_fcenters[tri_idx] = (p1 + p2 + p3) / 3.0f;
            tri_idx++;
        }
        vtx_offset += ms[i]->_num_vtx;
    }

    aap pln(total);
    s_idx_buffer          = new unsigned int[_num];
    unsigned int left_idx = 0, right_idx = _num;

    tri_idx = 0;
    for (int i = 0; i < ms.size(); i++)
        for (int j = 0; j < ms[i]->_num_tri; j++)
        {
            if (pln.inside(s_fcenters[tri_idx]))
                s_idx_buffer[left_idx++] = tri_idx;
            else
                s_idx_buffer[--right_idx] = tri_idx;

            tri_idx++;
        }

    _nodes         = new bvh_node[_num * 2 - 1];
    _nodes[0]._box = total;

    bvh_node* s_current = _nodes + 3;
    if (_num == 1)
        _nodes[0]._child = 0;
    else
    {
        _nodes[0]._child = -1;

        if (left_idx == 0 || left_idx == _num)
            left_idx = _num / 2;
        _nodes[0].left()->construct(s_idx_buffer, left_idx, s_fcenters, s_fboxes, s_current);
        _nodes[0].right()->construct(s_idx_buffer + left_idx, _num - left_idx, s_fcenters, s_fboxes, s_current);
    }

    delete[] s_idx_buffer;
    delete[] s_fcenters;

    refit(s_fboxes);
    reorder();
    resetParents();
}

void bvh::push2GPU(bool isCloth)
{
    unsigned int length = _num * 2 - 1;
    int*         ids    = new int[length * 2];

    for (unsigned int i = 0; i < length; i++)
    {
        ids[i]          = (root() + i)->triID();
        ids[length + i] = (root() + i)->parentID();
    }

    ::pushBVH(length, ids, isCloth);
    delete[] ids;

    unsigned int leafNum = 0;
    int*         idf     = new int[_num];
    for (unsigned int i = 0; i < length; i++)
    {
        if ((root() + i)->isLeaf())
        {
            int idx  = (root() + i)->triID();
            idf[idx] = i;
            leafNum++;
        }
    }
    assert(leafNum == _num);
    ::pushBVHLeaf(leafNum, idf, isCloth);
    delete[] idf;

    {  // push information for refit
        int max_level = 0;
        root()->getLevel(0, max_level);
        max_level++;

        unsigned int* level_idx    = new unsigned int[max_level];
        unsigned int* level_buffer = new unsigned int[max_level];
        for (int i = 0; i < max_level; i++)
            level_idx[i] = level_buffer[i] = 0;

        root()->getLevelIdx(0, level_buffer);
        for (int i = 1; i < max_level; i++)
            for (int j = 0; j < i; j++)
                level_idx[i] += level_buffer[j];

        delete[] level_buffer;
        ::pushBVHIdx(max_level, level_idx, isCloth);
        delete[] level_idx;
    }

    ::refitBVH(isCloth);
}

void front_list::push2GPU(bvh_node* r1, bvh_node* r2)
{
    bool self = (r2 == NULL);

    if (r2 == NULL)
        r2 = r1;

    int num = size();
    if (num)
    {
        int           idx    = 0;
        unsigned int* buffer = new unsigned int[num * 4];
        for (vector<front_node>::iterator it = begin();
             it != end();
             it++)
        {
            front_node n  = *it;
            buffer[idx++] = n._left - r1;
            buffer[idx++] = n._right - r2;
            buffer[idx++] = 0;
            buffer[idx++] = n._ptr;
        }

        ::pushFront(self, num, buffer);
        printf("%d\n", num);
        delete[] buffer;
    }
    else
        ::pushFront(self, 0, NULL);
}
}  // namespace PhysIKA