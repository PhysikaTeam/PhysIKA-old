#pragma once
#include <iostream>
#include "Physika_Framework/Framework/Field.h"
#include "Physika_Framework/Framework/Object.h"

namespace Physika {
/**
*  \brief Base class for modules
*
*  This class contains all functionality shared by every module in Physika.
*  It defines how to retrieve information about an class (name, type, data fields).
*
*/

typedef std::string FieldID;

class Base : public Object
{
public:
	typedef std::vector<std::shared_ptr<Field>> FieldVector;
	typedef std::map<FieldID, std::shared_ptr<Field>> FieldMap;

	Base() : Object() {};
	~Base() override {};

	bool addField(std::shared_ptr<Field> data);
	bool addField(FieldID name, std::shared_ptr<Field> data);
	bool addFieldAlias(FieldID name, std::shared_ptr<Field> data);
	bool addFieldAlias(FieldID name, std::shared_ptr<Field> data, MapPtr<Field>& fieldAlias);

	bool findField(std::shared_ptr<Field> data);
	bool findFieldAlias(const FieldID name);
	bool findFieldAlias(const FieldID name, MapPtr<Field>& fieldAlias);

	bool removeField(std::shared_ptr<Field> data);
	bool removeFieldAlias(const FieldID name);
	bool removeFieldAlias(const FieldID name, MapPtr<Field>& fieldAlias);

	std::shared_ptr<Field>	getField(const FieldID name);

	template<typename T>
	std::shared_ptr< T > getField(FieldID name)
	{
		MapPtr<Field>::iterator iter = m_fieldAlias.find(name);
		if (iter != m_fieldAlias.end())
		{
			return std::dynamic_pointer_cast<T>(iter->second);
		}
		return nullptr;
	}


	std::vector<FieldID>	getFieldAlias(std::shared_ptr<Field> data);
	int				getFieldAliasCount(std::shared_ptr<Field> data);

private:
	FieldVector m_field;
	FieldMap m_fieldAlias;
};

}
