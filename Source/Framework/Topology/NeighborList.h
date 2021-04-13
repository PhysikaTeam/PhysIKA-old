#pragma once
#include "Core/Platform.h"
#include "Core/Array/Array.h"
#include "Core/Utility.h"

namespace PhysIKA
{
	template<typename ElementType>
	class NeighborList
	{
	public:
		NeighborList()
			: m_maxNum(0)
		{
		};

		NeighborList(int n, int maxNbr)
			: m_maxNum(maxNbr)
		{
			resize(n, maxNbr);
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

		GPU_FUNC void setNeighborSize(int i, int num)
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

		void resize(int n, int maxNbr = 0) {
			m_index.resize(n);
			if (maxNbr != 0)
			{
				setNeighborLimit(maxNbr);
			}
			else
			{
				setDynamic();
			}
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

		void copyFrom(NeighborList<ElementType>& neighborlist)
		{
			m_maxNum = neighborlist.m_maxNum;
			if (m_elements.size() != neighborlist.m_elements.size())
				m_elements.resize(neighborlist.m_elements.size());

			Function1Pt::copy(m_elements, neighborlist.m_elements);

			if (m_index.size() != neighborlist.m_index.size())
				m_index.resize(neighborlist.m_index.size());

			Function1Pt::copy(m_index, neighborlist.m_index);
			
		}

		DeviceArray<int>& getIndex() { return m_index; }
		DeviceArray<ElementType>& getElements() { return m_elements; }

	private:

		int m_maxNum;
		DeviceArray<ElementType> m_elements;
		DeviceArray<int> m_index;
	};
}