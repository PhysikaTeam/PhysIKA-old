#pragma once
#include "Physika_Core/Platform.h"
#include <memory>
#include <vector>
#include <cassert>
#include <iostream>
#include "Physika_Framework/Framework/Base.h"
#include "Physika_Framework/Framework/Log.h"
#include "Physika_Core/Typedef.h"
#include "Physika_Core/DataTypes.h"

namespace Physika
{
class Node;

class Module : public Base
{
public:
	Module();

	~Module(void) override;

	bool initialize();

	virtual void beforeExecution() {};

	virtual bool execute() { return false; }

	virtual void afterExecution() {};

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

	bool isInitialized();

	virtual std::string getModuleType() { return "Module"; }

protected:
	/// \brief Initialization function for each module
	/// 
	/// This function is used to initialize internal variables for each module
	/// , it is called after all fields are set.
	virtual bool initializeImpl() { return m_initialized; }

private:
	Node* m_node;
	bool m_initialized;
};
}