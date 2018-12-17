#include "Base.h"
#include "Physika_Framework/Framework/Field.h"

namespace Physika {

bool Base::addField(std::shared_ptr<Field> data)
{
	return addField(data->getObjectName(), data);
}

bool Base::addField(FieldID name, std::shared_ptr<Field> data)
{
	if (findField(data) == NULL)
	{
		m_field.push_back(data);
	}
	else
	{
		std::cout << "Data field " << name
			<< " already exists in this class !"
			<< std::endl;
		return false;
	}

	addFieldAlias(name, data);

	return true;
}

bool Base::addFieldAlias(FieldID name, std::shared_ptr<Field> data)
{
	if (findFieldAlias(name) == NULL)
	{
		m_fieldAlias.insert(std::make_pair(name, data));
	}
	else
	{
		if (data != getField(name))
		{
			std::cout << "Field name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return false;
		}

	}

}

bool Base::addFieldAlias(FieldID name, std::shared_ptr<Field> data, MapPtr<Field>& fieldAlias)
{
	if (findFieldAlias(name, fieldAlias) == NULL)
	{
		fieldAlias.insert(std::make_pair(name, data));
	}
	else
	{
		if (data != getField(name))
		{
			std::cout << "Field name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return false;
		}

	}
}

bool Base::findField(std::shared_ptr<Field> data)
{
	VectorPtr<Field>::iterator result = find(m_field.begin(), m_field.end(), data);
	// return false if no field is found!
	if (result == m_field.end())
	{
		return false;
	}
	return true;
}

bool Base::findFieldAlias(const FieldID name)
{
	MapPtr<Field>::iterator result = m_fieldAlias.find(name);
	// return false if no alias is found!
	if (result == m_fieldAlias.end())
	{
		return false;
	}
	return true;
}

bool Base::findFieldAlias(const FieldID name, MapPtr<Field>& fieldAlias)
{
	MapPtr<Field>::iterator result = fieldAlias.find(name);
	// return false if no alias is found!
	if (result == fieldAlias.end())
	{
		return false;
	}
	return true;
}

bool Base::removeField(std::shared_ptr<Field> data)
{
	VectorPtr<Field>::iterator result = find(m_field.begin(), m_field.end(), data);
	if (result == m_field.end())
	{
		return false;
	}

	m_field.erase(result);

	MapPtr<Field>::iterator iter;
	for (iter = m_fieldAlias.begin(); iter != m_fieldAlias.end();)
	{
		if (iter->second == data)
		{
			m_fieldAlias.erase(iter++);
		}
		else
		{
			++iter;
		}
	}

	return true;
}

bool Base::removeFieldAlias(const FieldID name)
{
	return removeFieldAlias(name, m_fieldAlias);
}

bool Base::removeFieldAlias(const FieldID name, MapPtr<Field>& fieldAlias)
{
	MapPtr<Field>::iterator iter = fieldAlias.find(name);
	if (iter != fieldAlias.end())
	{
		std::shared_ptr<Field> data = iter->second;

		fieldAlias.erase(iter);

		if (getFieldAliasCount(data) == 0)
		{
			removeField(data);
		}
		return true;
	}

	return false;
}

std::shared_ptr<Physika::Field> Base::getField(const FieldID name)
{
	MapPtr<Field>::iterator iter = m_fieldAlias.find(name);
	if (iter != m_fieldAlias.end())
	{
		return iter->second;
	}
	return nullptr;
}

std::vector<std::string> Base::getFieldAlias(std::shared_ptr<Field> field)
{
	std::vector<FieldID> names;
	MapPtr<Field>::iterator iter;
	for (iter = m_fieldAlias.begin(); iter != m_fieldAlias.end(); iter++)
	{
		if (iter->second == field)
		{
			names.push_back(iter->first);
		}
	}
	return names;
}

int Base::getFieldAliasCount(std::shared_ptr<Field> data)
{
	int num = 0;
	MapPtr<Field>::iterator iter;
	for (iter = m_fieldAlias.begin(); iter != m_fieldAlias.end(); iter++)
	{
		if (iter->second == data)
		{
			num++;
		}
	}
	return num;
}

}