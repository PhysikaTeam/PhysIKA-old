#pragma once
#include "Platform.h"
#include <typeinfo>
#include <string>
#include <cuda_runtime.h>
#include "Typedef.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Cuda_Array/MemoryManager.h"

using namespace Physika;
namespace Physika {
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

class Base;

/*!
*	\class	LocalVariable
*	\brief	Variables of build-in data types.
*/
template<typename T, DeviceType deviceType = DeviceType::CPU>
class Variable : public Field
{
public:
	Variable(std::string name, std::string description);

public:
	~Variable() override;

	size_t size() override { return 1; }
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("Variable"); }
	DeviceType getDeviceType() override { return deviceType; }
		
	T getValue();
	void setValue(T val);

	inline T* getDataPtr() { return m_data; }
// 
// 	static std::shared_ptr< Variable<T, deviceType> > 	create(std::string name, std::string description)
// 	{
// 		return TypeInfo::New<Variable<T, deviceType>>(name, description);
// 	}

private:
	Variable() {};
	T* m_data;
	std::shared_ptr<MemoryManager<deviceType>> m_alloc;
};

template<typename T, DeviceType deviceType>
Variable<T, deviceType>::Variable(std::string name, std::string description)
	: Field(name, description)
	, m_data(NULL)
	, m_alloc(std::make_shared<DefaultMemoryManager<deviceType>>())
{
//	m_alloc->allocMemory1D((void**)&m_data, 1, sizeof(T));

	switch (deviceType)
	{
	case GPU:	(cudaMalloc((void**)&m_data, 1 * sizeof(T)));	break;
	case CPU:	m_data = new T;	break;
	default:	break;
	}
}

template<typename T, DeviceType deviceType>
Variable<T, deviceType>::~Variable()
{
	if (m_data != NULL)
	{
		m_alloc->releaseMemory((void**)&m_data);
// 		switch (deviceType)
// 		{
// 		case GPU:	(cudaFree(m_data));	break;
// 		case CPU:	delete m_data;	break;
// 		default:
// 			break;
// 		}
	}
};

template<typename T, DeviceType deviceType>
void Variable<T, deviceType>::setValue(T val)
{
//	m_alloc->initMemory((void*)m_data, val, 1 * sizeof(T));

	switch (deviceType)
	{
	case GPU:	(cudaMemcpy(m_data, &val, sizeof(T), cudaMemcpyHostToDevice));	break;
	case CPU:	*m_data = val;	break;
	default:
		break;
	}
}

template<typename T, DeviceType deviceType>
T Variable<T, deviceType>::getValue()
{
	T val;
	switch (deviceType)
	{
	case GPU:	(cudaMemcpy(&val, m_data, sizeof(T), cudaMemcpyDeviceToHost));	break;
	case CPU:	val = *m_data;	break;
	default:	break;
	}
	return val;
}

template<typename T>
using HostVariable = Variable<T, DeviceType::CPU>;

template<typename T>
using DeviceVariable = Variable<T, DeviceType::GPU>;

template<typename T>
using HostVariablePtr = std::shared_ptr< HostVariable<T> >;

template<typename T>
using DeviceVariablePtr = std::shared_ptr< DeviceVariable<T> >;

template<typename T, DeviceType deviceType>
class ArrayBuffer : public Field
{
public:
	ArrayBuffer(std::string name, std::string description, int num = 1);
	~ArrayBuffer() override;

	size_t size() override { return m_data->size(); }
	void resize(int num);
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("ArrayBuffer"); }
	DeviceType getDeviceType() override { return deviceType; }

	Array<T>* getDataPtr() { return m_data; }

	Array<T>& getValue() { return *m_data; }

public:
	static ArrayBuffer* create(int num) { return new ArrayBuffer<T, deviceType>("default", "default", num); }
// 
// 	static std::shared_ptr< ArrayBuffer<T, deviceType> > 
// 		create(std::string name, std::string description, int num)
// 	{
// 		return TypeInfo::New<ArrayBuffer<T, deviceType>>(name, description, num);//SPtr< ArrayBuffer<T, deviceType> > var( new ArrayBuffer<T, deviceType>(name, description) );
// 	}

private:
	ArrayBuffer() {};

	Array<T>* m_data;
};

// template<typename T, DeviceType deviceType>
// ArrayBuffer<T, deviceType>::ArrayBuffer(int num)
// 	: Field()
// 	, m_data(NULL)
// {
// 	m_data = new Array<T, deviceType>(num);
// }

template<typename T, DeviceType deviceType>
ArrayBuffer<T, deviceType>::ArrayBuffer(std::string name, std::string description, int num)
	: Field(name, description)
	, m_data(NULL)
{
	m_data = new Array<T, deviceType>(num);
}


template<typename T, DeviceType deviceType>
void ArrayBuffer<T, deviceType>::resize(int num)
{
	m_data->resize(num);
}

template<typename T, DeviceType deviceType>
ArrayBuffer<T, deviceType>::~ArrayBuffer()
{
	if (m_data != NULL)
	{
		m_data->release();
		delete m_data;
	}
}


template<typename T>
using HostBuffer = ArrayBuffer<T, DeviceType::CPU>;

template<typename T>
using DeviceBuffer = ArrayBuffer<T, DeviceType::GPU>;
}