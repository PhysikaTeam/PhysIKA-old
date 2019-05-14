#pragma once
#include "Physika_Core/Typedef.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Cuda_Array/MemoryManager.h"
#include "Physika_Framework/Framework/Field.h"
#include "Physika_Framework/Framework/Base.h"
#include "Physika_Framework/Topology/NeighborList.h"

namespace Physika {

	template<typename TDataType>
	class TPair
	{
	public:
		typedef typename TDataType::Coord Coord;

		int j;
		Coord pos;
	};

template<typename T>
class NeighborField : public Field
{
public:
	typedef T VarType;
	typedef NeighborList<T> FieldType;

	NeighborField();
	NeighborField(int num, int nbrSize = 0);
	NeighborField(std::string name, std::string description, int num = 1, int nbrSize = 0);
	~NeighborField() override;

	size_t getElementCount() override { return m_data->size(); }
	void setElementCount(int num, int nbrSize = 0);
//	void resize(int num);
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("NeighborField"); }
//	DeviceType getDeviceType() override { return DeviceType::GPU; }

	std::shared_ptr<NeighborList<T>> getReference() { return m_data; }

	NeighborList<T>& getValue() const { return *m_data; }

	bool isEmpty() override {
		return m_data == nullptr;
	}

	bool connect(NeighborField<T>& field2);

public:
	static NeighborField<T>*
		createField(Base* module, std::string name, std::string description, int num, int nbrSize = 0)
	{
		Field* ret = module->getField(name);
		if (nullptr != ret)
		{
			std::cout << "Neighbor name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = new NeighborField<T>(name, description, num, nbrSize);
		module->addField(name, var);
		return var;
	}

private:
	std::shared_ptr<NeighborList<T>> m_data;
};


template<typename T>
NeighborField<T>::NeighborField()
	:Field("", "")
	, m_data(nullptr)
{
}


template<typename T>
NeighborField<T>::NeighborField(int num, int nbrSize /*= 0*/)
	:Field("", "")
{
	m_data = std::make_shared<NeighborList<T>>();
	m_data->resize(num);
	if (nbrSize != 0)
	{
		m_data->setNeighborLimit(nbrSize);
	}
	else
	{
		m_data->setDynamic();
	}
}

template<typename T>
void NeighborField<T>::setElementCount(int num, int nbrSize /*= 0*/)
{
	if (m_data == nullptr)
		m_data = std::make_shared<NeighborList<T>>();
	
	m_data->resize(num);
	if (nbrSize != 0)
	{
		m_data->setNeighborLimit(nbrSize);
	}
	else
	{
		m_data->setDynamic();
	}
}

template<typename T>
NeighborField<T>::NeighborField(std::string name, std::string description, int num, int nbrSize)
	: Field(name, description)
{
	m_data = std::make_shared<NeighborList<T>>();
	m_data->resize(num);
	if (nbrSize != 0)
	{
		m_data->setNeighborLimit(nbrSize);
	}
	else
	{
		m_data->setDynamic();
	}
}


// template<typename T>
// void NeighborField<T>::resize(int num)
// {
// 	m_data->resize(num);
// }

template<typename T>
NeighborField<T>::~NeighborField()
{
	if (m_data.use_count() == 1)
	{
		m_data->release();
	}
}

template<typename T>
bool NeighborField<T>::connect(NeighborField<T>& field2)
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

}