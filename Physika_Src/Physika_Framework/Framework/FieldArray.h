#pragma once
#include "Typedef.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Cuda_Array/MemoryManager.h"
#include "Field.h"
#include "Base.h"

namespace Physika {

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

	void reset() override { m_data->reset(); }

public:
	static ArrayBuffer* create(int num) { return new ArrayBuffer<T, deviceType>("default", "default", num); }

// 	static std::shared_ptr< ArrayBuffer<T, deviceType> >
// 		create(std::string name, std::string description, int num)
// 	{
// 		return TypeInfo::New<ArrayBuffer<T, deviceType>>(name, description, num);//SPtr< ArrayBuffer<T, deviceType> > var( new ArrayBuffer<T, deviceType>(name, description) );
// 	}

	static std::shared_ptr< ArrayBuffer<T, deviceType> >
		createField(Base* module, std::string name, std::string description, int num)
	{
		std::shared_ptr<Field> ret = module->getField(name);
		if (nullptr != ret)
		{
			std::cout << "Array name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = TypeInfo::New<ArrayBuffer<T, deviceType>>(name, description, num);//SPtr< ArrayBuffer<T, deviceType> > var( new ArrayBuffer<T, deviceType>(name, description) );
		module->addField(name, var);
		return var;
	}

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