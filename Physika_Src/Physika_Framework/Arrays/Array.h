#pragma once
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include "Platform.h"

namespace Physika {

	/*!
	*	\class	Array
	*	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
	*/
	template<typename T, DeviceType deviceType = DeviceType::GPU>
	class Array
	{
	public:
		Array()
			: m_data(NULL)
			, m_totalNum(0)
		{
		};

		Array(int num) 
			: m_data(NULL)
			, m_totalNum(num)
		{
			AllocMemory();
		}

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array() {};

		void Resize(int n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void Reset();

		/*!
		*	\brief	Free allocated memory.	Should be called before the object is deleted.
		*/
		void release();

		inline T*		GetDataPtr() { return m_data; }
		inline void		SetDataPtr(T* data) { m_data = data; }

		DeviceType		GetDeviceType() { return deviceType; }

		void Swap(Array<T, deviceType>& arr)
		{
			assert(m_totalNum == arr.Size());
			T* tp = arr.GetDataPtr();
			arr.SetDataPtr(m_data);
			m_data = tp;
		}

		HYBRID_FUNC inline T& operator [] (unsigned int id)
		{
			return m_data[id];
		}

		HYBRID_FUNC inline T operator [] (unsigned int id) const
		{
			return m_data[id];
		}

		HYBRID_FUNC inline int Size() { return m_totalNum; }
		HYBRID_FUNC inline bool IsCPU() { return deviceType == DeviceType::CPU; }
		HYBRID_FUNC inline bool IsGPU() { return deviceType == DeviceType::GPU; }

	protected:
		void AllocMemory();
		
	private:
		T* m_data;
		int m_totalNum;
	};

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::Resize(const int n)
	{
		if (NULL != m_data) release();
		m_totalNum = n;
		AllocMemory();
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::release()
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
		m_totalNum = 0;
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::AllocMemory()
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

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::Reset()
	{
		switch (deviceType)
		{
		case CPU:
			memset((void*)m_data, 0, m_totalNum*sizeof(T));
			break;
		case GPU:
			cudaMemset(m_data, 0, m_totalNum * sizeof(T));
			break;
		default:
			break;
		}
	}

	template<typename T>
	using HostArray = Array<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArray = Array<T, DeviceType::GPU>;
}
