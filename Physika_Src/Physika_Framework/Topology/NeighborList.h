#pragma once
#include "Physika_Core/Platform.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include <thrust/scan.h>

namespace Physika
{
	template<typename ElementType>
	class NeighborList
	{
	public:
		NeighborList()
			: m_maxNum(0)
		{
		};

		~NeighborList() {};

		COMM_FUNC int size() { return m_index.size(); }

		GPU_FUNC int getNeighborSize(int i)
		{ 
			if (!isLimited())
			{
				if (i >= m_index.size() - 1)
				{
					return m_elements.size() - m_index[i];
				}
				return m_index[i + 1] - m_index[i];
			}
			else
			{
				return m_index[i];
			}
		}

		COMM_FUNC int getNeighborLimit()
		{
			return m_maxNum;
		}

		GPU_FUNC void setNeighborLimit(int i, int num)
		{
			if (isLimited())
				m_index[i] = num;
		}

		GPU_FUNC ElementType getElement(int i, int j) {
			if (!isLimited())
				return m_elements[m_index[i] + j];
			else
				return m_elements[m_maxNum*i + j];
		};

		GPU_FUNC void setElement(int i, int j, ElementType elem) {
			if (!isLimited())
				m_elements[m_index[i] + j] = elem;
			else
				m_elements[m_maxNum*i + j] = elem;
		}

		COMM_FUNC bool isLimited()
		{
			return m_maxNum > 0;
		}

		void resize(int n) {
			m_index.resize(n);
		}
		
		void release()
		{
			m_elements.release();
			m_index.release();
		}

		void setNeighborLimit(int nbrMax)
		{
			m_maxNum = nbrMax;
			m_elements.resize(m_maxNum*m_index.size());
		}

		void setDynamic()
		{
			m_maxNum = 0;
		}

		DeviceArray<int>& getIndex() { return m_index; }
		DeviceArray<ElementType>& getElements() { return m_elements; }

	private:

		int m_maxNum;
		DeviceArray<ElementType> m_elements;
		DeviceArray<int> m_index;
	};
}