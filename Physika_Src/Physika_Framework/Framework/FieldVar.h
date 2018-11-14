#pragma once
#include "Typedef.h"
#include "Field.h"
#include "Base.h"
#include "Physika_Core/Cuda_Array/MemoryManager.h"

namespace Physika {

/*!
*	\class	Variable
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

	void reset() override;

// 	static std::shared_ptr< Variable<T, deviceType> > 	create(std::string name, std::string description)
// 	{
// 		return TypeInfo::New<Variable<T, deviceType>>(name, description);
// 	}

	static std::shared_ptr< Variable<T, deviceType> >
		createField(Base* module, std::string name, std::string description, T value)
	{
		std::shared_ptr<Field> ret = module->getField(name);
		if (nullptr != ret)
		{
			std::cout << "Variable " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = TypeInfo::New<Variable<T, deviceType>>(name, description);//Variable<T, deviceType>::create(name, description);
		var->setValue(value);
		module->addField(name, TypeInfo::CastPointerUp<Field>(var));
		return var;
	}

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

template<typename T, DeviceType deviceType /*= DeviceType::CPU*/>
void Physika::Variable<T, deviceType>::reset()
{
	T value = T(0);
	switch (deviceType)
	{
	case GPU:	(cudaMemcpy(m_data, &value, sizeof(T), cudaMemcpyHostToDevice));	break;
	case CPU:	*m_data = value;	break;
	default:
		break;
	}
}

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

}