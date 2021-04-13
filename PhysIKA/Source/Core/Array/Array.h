#pragma once
#include <cassert>
#include <vector>
#include <cuda_runtime.h>
#include <memory>
#include "Core/Platform.h"
#include "MemoryManager.h"

namespace PhysIKA {

	/*!
	*	\class	Array
	*	\brief	This class is designed to be elegant, so it can be directly passed to GPU as parameters.
	*/
	template<typename T, DeviceType deviceType = DeviceType::GPU>
	class Array
	{
	public:
		Array(const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
			: m_data(NULL)
			, m_totalNum(0)
			, m_alloc(alloc)
		{
		};

		Array(int num, const std::shared_ptr<MemoryManager<deviceType>> alloc = std::make_shared<DefaultMemoryManager<deviceType>>())
			: m_data(NULL)
			, m_totalNum(num)
			, m_alloc(alloc)
		{
			allocMemory();
		}

		/*!
		*	\brief	Should not release data here, call Release() explicitly.
		*/
		~Array() {};

		void resize(int n);

		/*!
		*	\brief	Clear all data to zero.
		*/
		void reset();

		/*!
		*	\brief	Free allocated memory.	Should be called before the object is deleted.
		*/
		void release();

		inline T*		getDataPtr() { return m_data; }

		DeviceType		getDeviceType() { return deviceType; }

		void Swap(Array<T, deviceType>& arr)
		{
			assert(m_totalNum == arr.Size());
// 			T* tp = arr.GetDataPtr();
// 			arr.SetDataPtr(m_data);
			T* tp = arr.m_data;
			arr.m_data = m_data;
			m_data = tp;
		}

		COMM_FUNC inline T& operator [] (unsigned int id)
		{
			return m_data[id];
		}

		COMM_FUNC inline T operator [] (unsigned int id) const
		{
			return m_data[id];
		}

		COMM_FUNC inline int size() { return m_totalNum; }
		COMM_FUNC inline bool isCPU() { return deviceType == DeviceType::CPU; }
		COMM_FUNC inline bool isGPU() { return deviceType == DeviceType::GPU; }
		COMM_FUNC inline bool isEmpty() { return m_data == NULL; }

	protected:
		void allocMemory();
		
	private:
		T* m_data;
		int m_totalNum;
		std::shared_ptr<MemoryManager<deviceType>> m_alloc;
	};

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::resize(const int n)
	{
//		assert(n >= 1);
		if (NULL != m_data) release();
		if (n <= 0)
		{
			m_totalNum = 0;
		}
		else
		{
			m_totalNum = n;
			allocMemory();
		}
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::release()
	{
// 		if (m_data != NULL)
// 		{
// 			switch (deviceType)
// 			{
// 			case CPU:
// 				delete[]m_data;
// 				break;
// 			case GPU:
// 				(cudaFree(m_data));
// 				break;
// 			default:
// 				break;
// 			}
// 		}
		if (m_data != NULL)
		{
			m_alloc->releaseMemory((void**)&m_data);
		}
		
		m_data = NULL;
		m_totalNum = 0;
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::allocMemory()
	{
// 		switch (deviceType)
// 		{
// 		case CPU:
// 			m_data = new T[m_totalNum];
// 			break;
// 		case GPU:
// 			(cudaMalloc((void**)&m_data, m_totalNum * sizeof(T)));
// 			break;
// 		default:
// 			break;
// 		}

		printf("%d %d\n", m_totalNum, sizeof(T));
		m_alloc->allocMemory1D((void**)&m_data, m_totalNum, sizeof(T));

		reset();
	}

	template<typename T, DeviceType deviceType>
	void Array<T, deviceType>::reset()
	{
// 		switch (deviceType)
// 		{
// 		case CPU:
// 			memset((void*)m_data, 0, m_totalNum*sizeof(T));
// 			break;
// 		case GPU:
// 			cudaMemset(m_data, 0, m_totalNum * sizeof(T));
// 			break;
// 		default:
// 			break;
// 		}

		m_alloc->initMemory((void*)m_data, 0, m_totalNum*sizeof(T));
	}

	template<typename T>
	using HostArray = Array<T, DeviceType::CPU>;

	template<typename T>
	using DeviceArray = Array<T, DeviceType::GPU>;
}
