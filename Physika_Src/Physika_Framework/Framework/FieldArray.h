#pragma once
#include "Physika_Core/Typedef.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Cuda_Array/MemoryManager.h"
#include "Physika_Core/Utilities/Function1Pt.h"
#include "Field.h"
#include "Base.h"
#include "Physika_Framework/Framework/Log.h"

namespace Physika {

template<typename T, DeviceType deviceType>
class ArrayField : public Field
{
public:
	typedef T VarType;
	typedef Array<T, deviceType> FieldType;

	ArrayField();
	ArrayField(int num);
	ArrayField(std::string name, std::string description, int num = 1);
	~ArrayField() override;

	size_t getElementCount() override { return m_data->size(); }
	void setElementCount(size_t num);
//	void resize(int num);
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("ArrayBuffer"); }
//	DeviceType getDeviceType() override { return deviceType; }

	std::shared_ptr<Array<T, deviceType>> getReference() { return m_data; }

	Array<T, deviceType>& getValue() const { return *m_data; }
	void setValue(std::vector<T>& vals);

//	void reset() override { m_data->reset(); }

	bool isEmpty() override {
		return m_data == nullptr;
	}

	bool connect(ArrayField<T, deviceType>& field2);

public:
	static ArrayField* create(int num) { return new ArrayField<T, deviceType>("default", "default", num); }

	static ArrayField<T, deviceType>*
		createField(Base* module, std::string name, std::string description, int num)
	{
		Field* ret = module->getField(name);
		if (nullptr != ret)
		{
			std::cout << "Array name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = new ArrayField<T, deviceType>(name, description, num);//SPtr< ArrayBuffer<T, deviceType> > var( new ArrayBuffer<T, deviceType>(name, description) );
		module->addField(name, var);
		return var;
	}

private:
	std::shared_ptr<Array<T, deviceType>> m_data;
};

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField()
	: Field("", "")
	, m_data(nullptr)
{
//	m_data = std::make_shared<Array<T, deviceType>>();
}

// template<typename T, DeviceType deviceType>
// ArrayBuffer<T, deviceType>::ArrayBuffer(int num)
// 	: Field()
// 	, m_data(NULL)
// {
// 	m_data = new Array<T, deviceType>(num);
// }


template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField(int num)
	: Field("", "")
{
	if (num < 1)
	{
		std::runtime_error(std::string("Array size should be larger than 1"));
	}
	m_data = std::make_shared<Array<T, deviceType>>(num);
}


template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::ArrayField(std::string name, std::string description, int num)
	: Field(name, description)
{
	if (num < 1)
	{
		std::runtime_error(std::string("Array size should be larger than 1"));
	}
	m_data = std::make_shared<Array<T, deviceType>>(num);
}


// template<typename T, DeviceType deviceType>
// void ArrayField<T, deviceType>::resize(int num)
// {
// 	if (m_data == nullptr)
// 		m_data = std::make_shared<Array<T, deviceType>>();
// 
// 	m_data->resize(num);
// }

template<typename T, DeviceType deviceType>
ArrayField<T, deviceType>::~ArrayField()
{
	if (m_data.use_count() == 1)
	{
		m_data->release();
	}
}

template<typename T, DeviceType deviceType>
void ArrayField<T, deviceType>::setElementCount(size_t num)
{
	if (m_data != nullptr)
	{
		m_data->resize(num);
	}
	else
	{
		m_data = std::make_shared<Array<T, deviceType>>(num);
	}
}

template<typename T, DeviceType deviceType>
bool ArrayField<T, deviceType>::connect(ArrayField<T, deviceType>& field2)
{
	if (this->isEmpty())
	{
		Log::sendMessage(Log::Warning, "The parent field " + this->getObjectName() + " is empty!");
		return false;
	}
	field2.setDerived(true);
	if (field2.m_data.use_count() == 1)
	{
		field2.m_data->release();
	}
	field2.m_data = m_data;
	return true;
}

template<typename T, DeviceType deviceType>
void ArrayField<T, deviceType>::setValue(std::vector<T>& vals)
{
	if (m_data == nullptr)
	{
		m_data = std::make_shared<Array<T, deviceType>>();
		m_data->resize(vals.size());
		Function1Pt::copy(*m_data, vals);
		return;
	}
	
	if (vals.size() != m_data->size())
	{
		Log::sendMessage(Log::Error, "The input array size is not equal to Field " + this->getObjectName());
	}
	else
	{
		Function1Pt::copy(*m_data, vals);
	}
}


template<typename T>
using HostArrayField = ArrayField<T, DeviceType::CPU>;

template<typename T>
using DeviceArrayField = ArrayField<T, DeviceType::GPU>;
}