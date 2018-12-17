#pragma once
#include "Physika_Core/Typedef.h"
#include "Physika_Core/Cuda_Array/Array.h"
#include "Physika_Core/Cuda_Array/MemoryManager.h"
#include "Physika_Framework/Framework/Field.h"
#include "Physika_Framework/Framework/Base.h"
#include "Physika_Framework/Topology/NeighborList.h"

namespace Physika {

template<typename T>
class NeighborField : public Field
{
public:
	typedef T VarType;

	NeighborField(std::string name, std::string description, int num = 1, int nbrSize = 0);
	~NeighborField() override;

	size_t size() override { return m_data->size(); }
	void resize(int num);
	const std::string getTemplateName() override { return std::string(typeid(T).name()); }
	const std::string getClassName() override { return std::string("NeighborField"); }
	DeviceType getDeviceType() override { return DeviceType::GPU; }

	NeighborList<T>* getDataPtr() { return m_data; }

	NeighborList<T>& getValue() { return *m_data; }

public:
	static std::shared_ptr< NeighborField<T> >
		createField(Base* module, std::string name, std::string description, int num, int nbrSize = 0)
	{
		std::shared_ptr<Field> ret = module->getField(name);
		if (nullptr != ret)
		{
			std::cout << "Neighbor name " << name
				<< " conflicts with existing fields!"
				<< std::endl;
			return nullptr;
		}

		auto var = TypeInfo::New<NeighborField<T>>(name, description, num, nbrSize);
		module->addField(name, var);
		return var;
	}

private:
	NeighborField() {};

	NeighborList<T>* m_data;
};


template<typename T>
NeighborField<T>::NeighborField(std::string name, std::string description, int num, int nbrSize)
	: Field(name, description)
	, m_data(NULL)
{
	m_data = new NeighborList<T>();
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
void NeighborField<T>::resize(int num)
{
	m_data->resize(num);
}

template<typename T>
NeighborField<T>::~NeighborField()
{
	if (m_data != NULL)
	{
		m_data->release();
		delete m_data;
	}
}

}