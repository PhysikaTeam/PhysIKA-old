#pragma once
#include <assert.h>
#include <cuda_runtime.h>
#include <cstring>
#include "Core/Platform.h"

namespace PhysIKA {

#define INVALID -1

	template<typename T, DeviceType deviceType = DeviceType::GPU>
	class Array3D
	{
	public:
		Array3D() 
			: m_nx(0)
			, m_ny(0)
			, m_nz(0)
			, m_nxy(0)
			, m_totalNum(0)
			, m_data(NULL)
		{};

		Array3D(int nx, int ny, int nz)
			: m_nx(nx)
			, m_ny(ny)
			, m_nz(nz)
			, m_nxy(nx*ny)
			, m_totalNum(nx*ny*nz)
			, m_data(NULL)
		{
			AllocMemory();
		};

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array3D() { };

		void Resize(int nx, int ny, int nz);

		void Reset();

		void Release();

		inline T*		GetDataPtr() { return m_data; }
		void			SetDataPtr(T* _data) { m_data = _data; }

		COMM_FUNC inline int Nx() { return m_nx; }
		COMM_FUNC inline int Ny() { return m_ny; }
		COMM_FUNC inline int Nz() { return m_nz; }
		
		COMM_FUNC inline T operator () (const int i, const int j, const int k) const
		{
			return m_data[i + j*m_nx + k*m_nxy];
		}

		COMM_FUNC inline T& operator () (const int i, const int j, const int k)
		{
			return m_data[i + j*m_nx + k*m_nxy];
		}

		COMM_FUNC inline int Index(const int i, const int j, const int k)
		{
			return i + j*m_nx + k*m_nxy;
		}

		COMM_FUNC inline T operator [] (const int id) const
		{
			return m_data[id];
		}

		COMM_FUNC inline T& operator [] (const int id)
		{
			return m_data[id];
		}

		COMM_FUNC inline int Size() { return m_totalNum; }
		COMM_FUNC inline bool IsCPU() { return deviceType == DeviceType::CPU; }
		COMM_FUNC inline bool IsGPU() { return deviceType == DeviceType::GPU; }

	public:
		void AllocMemory();

	private:
		int m_nx;
		int m_ny;
		int m_nz;
		int m_nxy;
		int m_totalNum;
		T*	m_data;
	};

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::Resize(int nx, int ny, int nz)
	{
		if (NULL != m_data) Release();
		m_nx = nx;	m_ny = ny;	m_nz = nz;	m_nxy = m_nx*m_ny;	m_totalNum = m_nxy*m_nz;
		AllocMemory();
	}

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::Reset()
	{
		switch (deviceType)
		{
		case CPU:
			memset((void*)m_data, 0, m_totalNum * sizeof(T));
			break;
		case GPU:
			cudaMemset(m_data, 0, m_totalNum * sizeof(T));
			break;
		default:
			break;
		}
	}

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::Release()
	{
		if (m_data != NULL)
		{
			switch (deviceType)
			{
			case CPU:
				delete[]m_data;
				break;
			case GPU:
				(cudaFree(m_data));
				break;
			default:
				break;
			}
		}

		m_data = NULL;
		m_nx = 0;
		m_ny = 0;
		m_nz = 0;
		m_nxy = 0;
		m_totalNum = 0;
	}

	template<typename T, DeviceType deviceType>
	void Array3D<T, deviceType>::AllocMemory()
	{
		switch (deviceType)
		{
		case CPU:
			m_data = new T[m_totalNum];
			break;
		case GPU:
			(cudaMalloc((void**)&m_data, m_totalNum * sizeof(T)));
			break;
		default:
			break;
		}

		Reset();
	}

	template<typename T>
	using HostArray3D = Array3D<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArray3D = Array3D<T, DeviceType::GPU>;

	typedef DeviceArray3D<float>	Grid1f;
	typedef DeviceArray3D<float3> Grid3f;
	typedef DeviceArray3D<bool> Grid1b;
}
