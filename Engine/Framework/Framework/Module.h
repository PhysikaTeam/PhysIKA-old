#pragma once
#include "Core/Platform.h"
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "Base.h"
#include "Log.h"
#include "Core/Typedef.h"
#include "Core/DataTypes.h"
#include "FieldVar.h"

namespace Physika
{
class Node;

class Module : public Base
{
public:
	Module(std::string name = "default");

	~Module(void) override;

	bool initialize();

	virtual void begin() {};

	virtual bool execute() { return false; }

	virtual void end() {};

	void setName(std::string name);
	void setParent(Node* node);

	std::string getName();

	Node* getParent()
	{
		if (m_node == NULL)
		{
			Log::sendMessage(Log::Error, "Parent node is not set!");
		}
		return m_node;
	}

	bool isInitialized();

	virtual std::string getModuleType() { return "Module"; }

protected:
	/// \brief Initialization function for each module
	/// 
	/// This function is used to initialize internal variables for each module
	/// , it is called after all fields are set.
	virtual bool initializeImpl();

private:
	Node* m_node;
	std::string m_module_name;
	bool m_initialized;
};
}