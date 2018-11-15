#pragma once
#include "Physika_Core/Platform.h"
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "Framework/Base.h"
#include "Framework/Log.h"
#include "Physika_Core/Typedef.h"
#include "Physika_Core/DataTypes.h"

namespace Physika
{
class Node;

class BaseSlot
{
public:
	BaseSlot() { m_initialized = false; }
	bool isInitialized() { return m_initialized; }

	void setName(std::string name) { m_name = name; }
	void setDescription(std::string desc) { m_description = desc; }

	std::string getName() { return m_name; }
	std::string getDesccription() { return m_description; }

protected:
	bool m_initialized;

private:
	std::string m_name;
	std::string m_description;
};

template < class T >
class Slot : public BaseSlot
{
public:
	Slot() : BaseSlot() {};

	void setField(std::shared_ptr<T> value) { m_value = value; m_initialized = true; }
	T& getField() {
		if (!m_value)
		{
			Log::sendMessage(Log::Error, std::string("Field ") + getName() + std::string(" is not set!"));
			exit(0);
		}
		
		return *m_value;
	}

	std::shared_ptr<Field> getFieldPtr() { return TypeInfo::CastPointerUp<Field>(m_value); }

private:
	std::shared_ptr<T> m_value;
};


class Module : public Base
{
public:
	Module();

	virtual ~Module(void);

	bool initialize();

	virtual bool execute() { return false; }

	virtual bool updateStates() { return true; }

	void setParent(Node* node) {
		m_node = node;
	}

	Node* getParent() {
		if (m_node == NULL)
		{
			Log::sendMessage(Log::Error, "Parent node is not set!");
		}
		return m_node; 
	}

	template<typename T>
	bool connect(std::shared_ptr<Field>& var, Slot<T>& slot)
	{
		std::shared_ptr<T> derived = TypeInfo::CastPointerDown<T>(var);
		if (!derived)
		{
			Log::sendMessage(Log::Error, std::string("Field ")+ slot.getName() + std::string(" does not match the required data type!"));
			return false;
		}

		slot.setField(derived);
		return true;
	}

// 	template<class T>
// 	void Connect(Arg<T>& ref, T& value)
// 	{
// 		ref.setValue(value);
// 	}

	bool isArgumentComplete();
	bool isInitialized();

	virtual std::string getModuleType() { return "Module"; }

protected:
	void initArgument(BaseSlot* arg, std::string name, std::string desc);
	virtual bool initializeImpl() { return m_initialized; }

private:
	Node* m_node;
	bool m_initialized;
	std::list<BaseSlot*> m_arguments;
};
}