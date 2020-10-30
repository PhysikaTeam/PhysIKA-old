#pragma once

#include <iostream>
#include <glm/vec3.hpp>
#include "Core/Platform.h"
#include "Dynamics/RigidBody/SpatialVector.h"

#include "Core/Matrix/matrix_base.h"
#include <glm/gtx/norm.hpp>


namespace PhysIKA
{
	template<typename T>
	class JointSpaceBase :public MatrixBase
	{
	public:
		virtual unsigned int rows() const { return 6; }
		//virtual unsigned int cols() const { return Dof; }

		
		virtual const SpatialVector<T> operator*(const VectorBase<T>& q)const { return SpatialVector<T>(); }
		virtual const SpatialVector<T> mul(const T* q)const { return SpatialVector<T>(); }

		virtual void transposeMul(const SpatialVector<T>& v, T* res)const{}

		virtual int dof()const { return 0; }
		virtual const SpatialVector<T>* getBases()const = 0;
		virtual SpatialVector<T>* getBases() = 0;
		
		virtual T& operator() (unsigned int, unsigned int) = 0;
		virtual const T operator() (unsigned int, unsigned int) const = 0;
	};

	template<typename T, unsigned int Dof>
	class JointSpace: public JointSpaceBase<T>//public MatrixBase<T>
	{
	public:
		JointSpace() {}

		//virtual unsigned int rows() const { return 6; }
		virtual unsigned int cols() const { return Dof; }

		virtual T& operator() (unsigned int i, unsigned int j);
		virtual const T operator() (unsigned int i, unsigned int j) const;

		
		virtual const SpatialVector<T> operator*(const VectorBase<T>& q)const;
		virtual const SpatialVector<T> mul(const T* q)const;

		/// transpose multiply. Transform a vector from 6d space to  joint space.
		virtual void transposeMul(const SpatialVector<T>& v, T* res)const;

		virtual int dof()const { return Dof; }
		virtual const SpatialVector<T>* getBases()const { return m_data; }
		virtual SpatialVector<T>* getBases() { return m_data; }

		//const 
	private:
		SpatialVector<T> m_data[Dof];
	};



	template<typename T, unsigned int Dof>
	inline T & JointSpace<T, Dof>::operator()(unsigned int i, unsigned int j)
	{
		return m_data[j][i];
	}
	template<typename T, unsigned int Dof>
	inline const T  JointSpace<T, Dof>::operator()(unsigned int i, unsigned int j) const
	{
		// TODO: insert return statement here
		return m_data[j][i];
	}

	template<typename T, unsigned int Dof>
	inline const SpatialVector<T> JointSpace<T, Dof>::operator*(const VectorBase<T>& q) const
	{
		SpatialVector<T> vj(0, 0, 0,  0, 0, 0);
		int n = q.size();
		for (int i = 0; i < n; ++i)
		{
			vj += (m_data[i] * q[i]);
		}
		return vj;
	}
	template<typename T, unsigned int Dof>
	inline const SpatialVector<T> JointSpace<T, Dof>::mul(const T * q) const
	{
		SpatialVector<T> vj(0, 0, 0, 0, 0, 0);
		for (int i = 0; i < Dof; ++i)
		{
			vj += (m_data[i] * (q[i]));
		}
		return vj;
	}
	template<typename T, unsigned int Dof>
	inline void JointSpace<T, Dof>::transposeMul(const SpatialVector<T>& v, T * res) const
	{
		for (int i = 0; i < Dof; ++i)
		{
			res[i] = (this->m_data[i]) * v;
		}
	}
}