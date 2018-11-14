#pragma once
#include "Platform.h"
#include <typeinfo>
#include <string>
#include <cuda_runtime.h>
#include "Typedef.h"

namespace Physika {
	class Base;
/*!
*	\class	Variable
*	\brief	Interface for all variables.
*/
class Field
{
public:
	Field() : m_name("default"), m_description("") {};
	Field(std::string name, std::string description) { m_name = name; m_description = description; }
	virtual ~Field() {};

	virtual size_t size() { return 0; }
	virtual const std::string getTemplateName() { return std::string(""); }
	virtual const std::string getClassName() { return std::string("Field"); }

	std::string	getObjectName() { return m_name; }
	std::string	getDescription() { return m_description; }
	virtual DeviceType	getDeviceType() { return DeviceType::UNDEFINED; }

	void	setObjectName(std::string name) { m_name = name; }
	void	setDescription(std::string description) { m_description = description; }

	virtual void reset() {};

public:
/*		template<typename T>
	static T* CastPointerDown(Field* b)
	{
		T* ptr = dynamic_cast<T*>(b);
		return ptr;
	}

	template<typename T>
	static Field* CastPointerUp(T* b)
	{
		Field* ptr = dynamic_cast<Field*>(b);
		return ptr;
	}

	template<typename T>
	static SPtr<T> CastPointerDown(SPtr<Field> b)
	{
		SPtr<T> ptr = std::dynamic_pointer_cast<T>(b);
		return ptr;
	}

	template<typename T>
	static SPtr<Field> CastPointerUp(SPtr<T> b)
	{
		SPtr<Field> ptr = std::dynamic_pointer_cast<Field>(b);
		return ptr;
	}*/

private:
	std::string m_name;
	std::string m_description;
};

}