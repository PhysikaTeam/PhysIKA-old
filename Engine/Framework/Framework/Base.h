#pragma once
#include <iostream>
#include "Framework/Framework/Field.h"
#include "Framework/Framework/Object.h"

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
	typedef std::vector<Field*> FieldVector;
	typedef std::map<FieldID, Field*> FieldMap;

	Base() : Object() {};
	~Base() override {};

	bool addField(Field* data);
	bool addField(FieldID name, Field* data);
	bool addFieldAlias(FieldID name, Field* data);
	bool addFieldAlias(FieldID name, Field* data, FieldMap& fieldAlias);

	bool findField(Field* data);
	bool findFieldAlias(const FieldID name);
	bool findFieldAlias(const FieldID name, FieldMap& fieldAlias);

	bool removeField(Field* data);
	bool removeFieldAlias(const FieldID name);
	bool removeFieldAlias(const FieldID name, FieldMap& fieldAlias);

	Field*	getField(const FieldID name);

	bool attachField(Field* field, std::string name, std::string desc, bool autoDestroy = true);

	template<typename T>
	T* getField(FieldID name)
	{
		FieldMap::iterator iter = m_fieldAlias.find(name);
		if (iter != m_fieldAlias.end())
		{
			return dynamic_cast<T*>(iter->second);
		}
		return nullptr;
	}

	bool isAllFieldsReady();

	std::vector<FieldID>	getFieldAlias(Field* data);
	int				getFieldAliasCount(Field* data);

private:
	FieldVector m_field;
	FieldMap m_fieldAlias;
};

}
