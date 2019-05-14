#include "Field.h"

namespace Physika
{
	void Field::setParent(Base* owner)
	{
		m_owner = owner;
	}

	Physika::Base* Field::getParent()
	{
		return m_owner;
	}

	bool Field::isDerived()
	{
		return m_derived;
	}

	bool Field::isAutoDestroyable()
	{
		return m_autoDestroyable;
	}

	void Field::setAutoDestroy(bool autoDestroy)
	{
		m_autoDestroyable = autoDestroy;
	}

	void Field::setDerived(bool derived)
	{
		m_derived = derived;
	}

}

