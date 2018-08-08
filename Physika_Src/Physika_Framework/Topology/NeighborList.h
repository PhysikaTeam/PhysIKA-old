#pragma once
#include "Platform.h"
namespace Physika
{
	template<class ElementType>
	class NeighborList
	{
	public:
		NeighborList() {};
		~NeighborList() {};

		GPU_FUNC virtual int getSize() { return m_start.size(); }

		GPU_FUNC virtual int getNeighborSize(int i) { return m_end[i] - m_start[i]; }

		GPU_FUNC virtual ElementType getNeighborElement(int i, int j) {
			return m_index[m_start[i] + j];
		};

		GPU_FUNC virtual void setNeighborElement(int i, int j, ElementType elem) {
			m_index[m_start[i] + j] = elem;
		}

	private:
		DeviceArray<ElementType> m_index;
		DeviceArray<int> m_start;
		DeviceArray<int> m_end;
	};
}