#pragma once

#include <vector>
#include <memory>
#include "Dynamics/RigidBody/TriangleMesh.h"
#include "Framework/Collision/CollidableTriangle.h"
#include "Framework/Collision/CollidableTriangleMesh.h"
#include "Framework/Framework/ModuleTopology.h"
namespace PhysIKA {
	
	class vec3f {
	public:
		union {
			struct {
				double x, y, z;
			};
			struct {
				double v[3];
			};
		};

		FORCEINLINE vec3f()
		{
			x = 0; y = 0; z = 0;
		}

		FORCEINLINE vec3f(const vec3f &v)
		{
			x = v.x;
			y = v.y;
			z = v.z;
		}

		FORCEINLINE vec3f(const double *v)
		{
			x = v[0];
			y = v[1];
			z = v[2];
		}
		FORCEINLINE vec3f(float *v)
		{
			x = v[0];
			y = v[1];
			z = v[2];
		}

		FORCEINLINE vec3f(double x, double y, double z)
		{
			this->x = x;
			this->y = y;
			this->z = z;
		}

		FORCEINLINE double operator [] (int i) const { return v[i]; }
		FORCEINLINE double &operator [] (int i) { return v[i]; }

		FORCEINLINE vec3f &operator += (const vec3f &v) {
			x += v.x;
			y += v.y;
			z += v.z;
			return *this;
		}

		FORCEINLINE vec3f &operator -= (const vec3f &v) {
			x -= v.x;
			y -= v.y;
			z -= v.z;
			return *this;
		}

		FORCEINLINE vec3f &operator *= (double t) {
			x *= t;
			y *= t;
			z *= t;
			return *this;
		}

		FORCEINLINE vec3f &operator /= (double t) {
			x /= t;
			y /= t;
			z /= t;
			return *this;
		}

		FORCEINLINE void negate() {
			x = -x;
			y = -y;
			z = -z;
		}

		FORCEINLINE vec3f operator - () const {
			return vec3f(-x, -y, -z);
		}

		FORCEINLINE vec3f operator+ (const vec3f &v) const
		{
			return vec3f(x + v.x, y + v.y, z + v.z);
		}

		FORCEINLINE vec3f operator- (const vec3f &v) const
		{
			return vec3f(x - v.x, y - v.y, z - v.z);
		}

		FORCEINLINE vec3f operator *(double t) const
		{
			return vec3f(x*t, y*t, z*t);
		}

		FORCEINLINE vec3f operator /(double t) const
		{
			return vec3f(x / t, y / t, z / t);
		}

		// cross product
		FORCEINLINE const vec3f cross(const vec3f &vec) const
		{
			return vec3f(y*vec.z - z * vec.y, z*vec.x - x * vec.z, x*vec.y - y * vec.x);
		}

		FORCEINLINE double dot(const vec3f &vec) const {
			return x * vec.x + y * vec.y + z * vec.z;
		}

		FORCEINLINE void normalize()
		{
			double sum = x * x + y * y + z * z;
			if (sum > double(10e-12)) {
				double base = double(1.0 / sqrt(sum));
				x *= base;
				y *= base;
				z *= base;
			}
		}

		FORCEINLINE double length() const {
			return double(sqrt(x*x + y * y + z * z));
		}

		FORCEINLINE vec3f getUnit() const {
			return (*this) / length();
		}
		inline bool isEqual(double a, double b, double tol = double(10e-6)) const
		{
			return fabs(a - b) < tol;
		}
		FORCEINLINE bool isUnit() const {
			return isEqual(squareLength(), 1.f);
		}

		//! max(|x|,|y|,|z|)
		FORCEINLINE double infinityNorm() const
		{
			return fmax(fmax(fabs(x), fabs(y)), fabs(z));
		}

		FORCEINLINE vec3f & set_value(const double &vx, const double &vy, const double &vz)
		{
			x = vx; y = vy; z = vz; return *this;
		}

		FORCEINLINE bool equal_abs(const vec3f &other) {
			return x == other.x && y == other.y && z == other.z;
		}

		FORCEINLINE double squareLength() const {
			return x * x + y * y + z * z;
		}

		static vec3f zero() {
			return vec3f(0.f, 0.f, 0.f);
		}

		//! Named constructor: retrieve vector for nth axis
		static vec3f axis(int n) {
			assert(n < 3);
			switch (n) {
			case 0: {
				return xAxis();
			}
			case 1: {
				return yAxis();
			}
			case 2: {
				return zAxis();
			}
			}
			return vec3f();
		}

		//! Named constructor: retrieve vector for x axis
		static vec3f xAxis() { return vec3f(1.f, 0.f, 0.f); }
		//! Named constructor: retrieve vector for y axis
		static vec3f yAxis() { return vec3f(0.f, 1.f, 0.f); }
		//! Named constructor: retrieve vector for z axis
		static vec3f zAxis() { return vec3f(0.f, 0.f, 1.f); }

	};

	inline vec3f operator * (double t, const vec3f &v) {
		return vec3f(v.x*t, v.y*t, v.z*t);
	}

	inline vec3f interp(const vec3f &a, const vec3f &b, double t)
	{
		return a * (1 - t) + b * t;
	}

	inline vec3f vinterp(const vec3f &a, const vec3f &b, double t)
	{
		return a * t + b * (1 - t);
	}

	inline vec3f interp(const vec3f &a, const vec3f &b, const vec3f &c, double u, double v, double w)
	{
		return a * u + b * v + c * w;
	}

	inline double clamp(double f, double a, double b)
	{
		return fmax(a, fmin(f, b));
	}

	inline double vdistance(const vec3f &a, const vec3f &b)
	{
		return (a - b).length();
	}


	inline std::ostream& operator<<(std::ostream&os, const vec3f &v) {
		os << "(" << v.x << ", " << v.y << ", " << v.z << ")" << std::endl;
		return os;
	}

#define CLAMP(a, b, c)		if((a)<(b)) (a)=(b); else if((a)>(c)) (a)=(c)


	FORCEINLINE void
		vmin(vec3f &a, const vec3f &b)
	{
		a.set_value(
			fmin(a[0], b[0]),
			fmin(a[1], b[1]),
			fmin(a[2], b[2]));
	}

	FORCEINLINE void
		vmax(vec3f &a, const vec3f &b)
	{
		a.set_value(
			fmax(a[0], b[0]),
			fmax(a[1], b[1]),
			fmax(a[2], b[2]));
	}

	FORCEINLINE vec3f lerp(const vec3f &a, const vec3f &b, float t)
	{
		return a + t * (b - a);
	}
	class front_node;
	class bvh_node;
	
	static vec3f *s_fcenters;
	class front_list : public std::vector<front_node> {
	public:
		void propogate(std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &c, vector<TrianglePair> &ret);
		void push2GPU(bvh_node *r1, bvh_node *r2 = NULL);
	};

	class bvh_node {
		TAlignedBox3D<float> _box;
		static bvh_node* s_current;
		int _child; // >=0 leaf with tri_id, <0 left & right
		int _parent;

		void setParent(int p);

	public:
		bvh_node();

		~bvh_node();

		void collide(bvh_node *other, std::vector<TrianglePair> &ret);

		void collide(bvh_node *other, front_list &f, int level, int ptr);

		void self_collide(front_list &lst, bvh_node *r);
		void
			construct(unsigned int id, TAlignedBox3D<float>*s_fboxes);
		void
			construct(unsigned int *lst, unsigned int num, vec3f *s_fcenters, TAlignedBox3D<float>*s_fboxes,
				bvh_node*& s_current);

		void visualize(int level);
		void refit(TAlignedBox3D<float>*s_fboxes);
		void resetParents(bvh_node *root);

		FORCEINLINE TAlignedBox3D<float> &box() { return _box; }
		FORCEINLINE bvh_node *left() { return this - _child; }
		FORCEINLINE bvh_node *right() { return this - _child + 1; }
		FORCEINLINE int triID() { return _child; }
		FORCEINLINE int isLeaf() { return _child >= 0; }
		FORCEINLINE int parentID() { return _parent; }

		FORCEINLINE void getLevel(int current, int &max_level);

		FORCEINLINE void getLevelIdx(int current, unsigned int *idx);
		void sprouting(bvh_node *other, front_list &append, std::vector < TrianglePair > &ret);

		friend class bvh;
	};
	class front_node {
	public:
		bvh_node *_left, *_right;
		unsigned int _flag; // vailid or not
		unsigned int _ptr; // self-coliding parent;

		front_node(bvh_node *l, bvh_node *r, unsigned int ptr);

		void update(front_list &appended, std::vector<TrianglePair> &ret);
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
		int _num; // all face num
		bvh_node *_nodes;
		TAlignedBox3D<float>*s_fboxes;

		unsigned int *s_idx_buffer;

	public:
		bvh() {};
		template<typename T>
		bvh(std::vector<std::shared_ptr<TriangleMesh<T>>> &ms) {
			_num = 0;
			_nodes = NULL;

			construct<T>(ms);
			reorder();
			resetParents(); //update the parents after reorder ...
		}
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
					s_fcenters[tri_idx] /= double(3.0);
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
			delete[] s_fcenters;

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

		void self_collide(front_list &f, std::vector<std::shared_ptr<TriangleMesh<DataType3f>>> &c);

		void visualize(int);
	};

	
}