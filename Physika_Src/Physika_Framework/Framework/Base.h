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

class Base : public Object
{
public:
	typedef std::vector<std::shared_ptr<Field>> FieldVector;
	typedef std::map<std::string, std::shared_ptr<Field>> FieldMap;

	Base() : Object() {};
	virtual ~Base() {};

	bool addField(std::shared_ptr<Field> data);
	bool addField(std::string name, std::shared_ptr<Field> data);
	bool addFieldAlias(std::string name, std::shared_ptr<Field> data);
	bool addFieldAlias(std::string name, std::shared_ptr<Field> data, MapPtr<Field>& fieldAlias);

	bool findField(std::shared_ptr<Field> data);
	bool findFieldAlias(const std::string name);
	bool findFieldAlias(const std::string name, MapPtr<Field>& fieldAlias);

	bool removeField(std::shared_ptr<Field> data);
	bool removeFieldAlias(const std::string name);
	bool removeFieldAlias(const std::string name, MapPtr<Field>& fieldAlias);

	std::shared_ptr<Field>	getField(const std::string name);

	template<typename T>
	std::shared_ptr< T > getField(std::string name)
	{
		MapPtr<Field>::iterator iter = m_fieldAlias.find(name);
		if (iter != m_fieldAlias.end())
		{
			return std::dynamic_pointer_cast<T>(iter->second);
		}
		return nullptr;
	}


	std::vector<std::string>	getFieldAlias(std::shared_ptr<Field> data);
	int				getFieldAliasCount(std::shared_ptr<Field> data);

/*	template<typename T>
	std::shared_ptr< HostVariable<T> > allocHostVariable(std::string name, std::string description)
	{
		return allocVariable<T, DeviceType::CPU>(name, description);
	}

	template<typename T>
	std::shared_ptr< HostVariable<T> > allocHostVariable(std::string name, std::string description, T value)
	{
		return allocVariable<T, DeviceType::CPU>(name, description, value);
	}

	template<typename T>
	std::shared_ptr< HostBuffer<T> > allocHostBuffer(std::string name, std::string description, int num)
	{
		return allocArrayBuffer<T, DeviceType::CPU>(name, description, num);
	}

	template<typename T>
	std::shared_ptr< HostVariable<T> > getHostVariable(std::string name)
	{
		return getVariable<T, DeviceType::CPU>(name);
	}

	template<typename T>
	std::shared_ptr< HostBuffer<T> > getHostBuffer(std::string name)
	{
		return getArrayBuffer<T, DeviceType::CPU>(name);
	}

	//#define CONTEXT_ADD_SEPCIAL_DATA()

protected:

	/// Allocate variables in context
	template<typename T, DeviceType deviceType>
	std::shared_ptr< Variable<T, deviceType> >
		allocVariable(std::string name, std::string description)
	{
		std::shared_ptr<Field> ret = this->getField(name);
		if (nullptr != ret)
		{
			std::cout << "Variable " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = TypeInfo::New<Variable<T, deviceType>>(name, description);//Variable<T, deviceType>::create(name, description);
		this->addField(name, TypeInfo::CastPointerUp<Field>(var));
		return var;
	}

	/// Allocate variables in context
	template<typename T, DeviceType deviceType>
	std::shared_ptr< Variable<T, deviceType> >
		allocVariable(std::string name, std::string description, T value)
	{
		std::shared_ptr<Field> ret = this->getField(name);
		if (nullptr != ret)
		{
			std::cout << "Variable " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = TypeInfo::New<Variable<T, deviceType>>(name, description);//Variable<T, deviceType>::create(name, description);
		var->setValue(value);
		this->addField(name, TypeInfo::CastPointerUp<Field>(var));
		return var;
	}

	/// Allocate one dimensional array buffer in context
	template<typename T, DeviceType deviceType>
	std::shared_ptr< ArrayBuffer<T, deviceType> >
		allocArrayBuffer(std::string name, std::string description, int num)
	{
		std::shared_ptr<Field> ret = this->getField(name);
		if (nullptr != ret)
		{
			std::cout << "ArrayBuffer " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = TypeInfo::New<ArrayBuffer<T, deviceType>>(name, description, num);// ArrayBuffer<T, deviceType>::create(name, description, num);
		this->addField(name, TypeInfo::CastPointerUp<Field>(var));
		return var;
	}

	/// Return the variable by name
	template<typename T, DeviceType deviceType>
	std::shared_ptr< Variable<T, deviceType> >
		getVariable(std::string name)
	{
		typedef Variable<T, deviceType> FieldType;
		std::shared_ptr<Field> ret = this->getField(name);
		if (nullptr == ret)
		{
			std::cout << "Variable " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		return TypeInfo::CastPointerDown<FieldType>(ret);
	}

	/// Return the array buffer by name
	template<typename T, DeviceType deviceType>
	std::shared_ptr< ArrayBuffer<T, deviceType> >
		getArrayBuffer(std::string name)
	{
		typedef ArrayBuffer<T, deviceType> FieldType;
		std::shared_ptr<Field> ret = this->getField(name);
		if (nullptr == ret)
		{
			std::cout << "ArrayBuffer " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		return TypeInfo::CastPointerDown<FieldType>(ret);
	}*/

private:
	FieldVector m_field;
	FieldMap m_fieldAlias;
};

}
