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

	void Field::setSource(Field* source)
	{
		m_source = source;
		if (source != nullptr)
		{
			m_derived = true;
		}
	}

	Field* Field::getSource()
	{
		return m_source;
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

