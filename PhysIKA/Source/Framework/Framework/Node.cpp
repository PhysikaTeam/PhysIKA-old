#include "Node.h"
#include "NodeIterator.h"

#include "Framework/Action/Action.h"


namespace PhysIKA
{
IMPLEMENT_CLASS(Node)

Node::Node(std::string name)
	: Base()
	, m_parent(NULL)
	, m_node_name(name)
	, m_dt(0.001f)
	, m_mass(1.0f)
{
	this->varScale()->setValue(Vector3f(1, 1, 1));
	this->varScale()->setMin(0.01);
	this->varScale()->setMax(100.0f);
}


Node::~Node()
{
	m_render_list.clear();
	m_module_list.clear();
}

void Node::setName(std::string name)
{
	m_node_name = name;
}

std::string Node::getName()
{
	return m_node_name;
}


Node* Node::getChild(std::string name)
{
	for (ListPtr<Node>::iterator it = m_children.begin(); it != m_children.end(); ++it)
	{
		if ((*it)->getName() == name)
			return it->get();
	}
	return NULL;
}

Node* Node::getParent()
{
	return m_parent;
}

Node* Node::getRoot()
{
	Node* root = this;
	while (root->getParent() != NULL)
	{
		root = root->getParent();
	}
	return root;
}

bool Node::isControllable()
{
	return m_controllable;
}

void Node::setControllable(bool con)
{
	m_controllable = con;
}

bool Node::isActive()
{
	return this->varActive()->getValue();
}

void Node::setActive(bool active)
{
	this->varActive()->setValue(active);
}

bool Node::isVisible()
{
	return this->varVisible()->getValue();
}

void Node::setVisible(bool visible)
{
	this->varVisible()->setValue(visible);
}

float Node::getDt()
{
	return m_dt;
}

void Node::setDt(Real dt)
{
	m_dt = dt;
}

void Node::setMass(Real mass)
{
	m_mass = mass;
}

Real Node::getMass()
{
	return m_mass;
}

bool Node::hasChild(std::shared_ptr<Node> child)
{
	auto it = find(m_children.begin(), m_children.end(), child);

	return it == m_children.end() ? false : true;
}

// NodeIterator Node::begin()
// {
// 	return NodeIterator(this);
// }
// 
// NodeIterator Node::end()
// {
// 	return NodeIterator();
// }

void Node::removeChild(std::shared_ptr<Node> child)
{
	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); )
	{
		if (*iter == child)
		{
			m_children.erase(iter++);
		}
		else
		{
			++iter;
		}
	}
}

void Node::removeAllChildren()
{
	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); )
	{
		m_children.erase(iter++);
	}
}

void Node::advance(Real dt)
{
	auto nModel = this->getNumericalModel();
	if (nModel == NULL)
	{
		Log::sendMessage(Log::Warning, this->getName() + ": No numerical model is set!");
	}
	else
	{
		nModel->step(this->getDt());
	}
}

std::shared_ptr<DeviceContext> Node::getContext()
{
	if (m_context == nullptr)
	{
		m_context = TypeInfo::New<DeviceContext>();
		m_context->setParent(this);
		addModule(m_context);
	}
	return m_context;
}

void Node::setContext(std::shared_ptr<DeviceContext> context)
{
	if (m_context != nullptr)
	{
		deleteModule(m_context);
	}

	m_context = context; 
	addModule(m_context);
}

std::shared_ptr<MechanicalState> Node::getMechanicalState()
{
	if (m_mechanical_state == nullptr)
	{
		m_mechanical_state = TypeInfo::New<MechanicalState>();
		m_mechanical_state->setParent(this);
		addModule(m_mechanical_state);
	}
	return m_mechanical_state;
}

void Node::setMechanicalState(std::shared_ptr<MechanicalState> state)
{
	if (m_mechanical_state != nullptr)
	{
		deleteModule(m_mechanical_state);
	}

	m_mechanical_state = state; 
	addModule(state);
}

std::unique_ptr<AnimationController>& Node::getAnimationPipeline()
{
	if (m_animation_pipeline == nullptr)
	{
		m_animation_pipeline = std::make_unique<AnimationController>();
	}
	return m_animation_pipeline;
}

std::unique_ptr<RenderController>& Node::getRenderPipeline()
{
	if (m_render_pipeline == nullptr)
	{
		m_render_pipeline = std::make_unique<RenderController>();
	}
	return m_render_pipeline;
}

/*
std::shared_ptr<MechanicalState> Node::getMechanicalState()
{
	if (m_mechanical_state == nullptr)
	{
		m_mechanical_state = TypeInfo::New<MechanicalState>();
		m_mechanical_state->setParent(this);
	}
	return m_mechanical_state;
}*/
/*
bool Node::addModule(std::string name, Module* module)
{
	if (getContext() == nullptr || module == NULL)
	{
		std::cout << "Context or module does not exist!" << std::endl;
		return false;
	}

	std::map<std::string, Module*>::iterator found = m_modules.find(name);
	if (found != m_modules.end())
	{
		std::cout << "Module name already exists!" << std::endl;
		return false;
	}
	else
	{
		m_modules[name] = module;
		m_module_list.push_back(module);

//		module->insertToNode(this);
	}

	return true;
}
*/
bool Node::addModule(std::shared_ptr<Module> module)
{
	bool ret = true;
	ret &= addToModuleList(module);

	std::string mType = module->getModuleType();
	if (std::string("TopologyModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<TopologyModule>(module);
		m_topology = downModule;
	}
	else if (std::string("NumericalModel").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<NumericalModel>(module);
		m_numerical_model = downModule;
	}
	else if (std::string("NumericalIntegrator").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<NumericalIntegrator>(module);
		m_numerical_integrator = downModule;
	}
	else if (std::string("ForceModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<ForceModule>(module);
		this->addToForceModuleList(downModule);
	}
	else if (std::string("ConstraintModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<ConstraintModule>(module);
		this->addToConstraintModuleList(downModule);
	}
	else if (std::string("ComputeModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<ComputeModule>(module);
		this->addToComputeModuleList(downModule);
	}
	else if (std::string("CollisionModel").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<CollisionModel>(module);
		this->addToCollisionModelList(downModule);
	}
	else if (std::string("VisualModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<VisualModule>(module);
		this->addToVisualModuleList(downModule);
	}
	else if (std::string("TopologyMapping").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<TopologyMapping>(module);
		this->addToTopologyMappingList(downModule);
	}

	return ret;
}

bool Node::deleteModule(std::shared_ptr<Module> module)
{
	bool ret = true;

	ret &= deleteFromModuleList(module);

	std::string mType = module->getModuleType();

	if (std::string("TopologyModule").compare(mType) == 0)
	{
		m_topology = nullptr;
	}
	else if (std::string("NumericalModel").compare(mType) == 0)
	{
		m_numerical_model = nullptr;
	}
	else if (std::string("NumericalIntegrator").compare(mType) == 0)
	{
		m_numerical_integrator = nullptr;
	}
	else if (std::string("ForceModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<ForceModule>(module);
		this->deleteFromForceModuleList(downModule);
	}
	else if (std::string("ConstraintModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<ConstraintModule>(module);
		this->deleteFromConstraintModuleList(downModule);
	}
	else if (std::string("ComputeModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<ComputeModule>(module);
		this->deleteFromComputeModuleList(downModule);
	}
	else if (std::string("CollisionModel").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<CollisionModel>(module);
		this->deleteFromCollisionModelList(downModule);
	}
	else if (std::string("VisualModule").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<VisualModule>(module);
		this->deleteFromVisualModuleList(downModule);
	}
	else if (std::string("TopologyMapping").compare(mType) == 0)
	{
		auto downModule = TypeInfo::CastPointerDown<TopologyMapping>(module);
		this->deleteFromTopologyMappingList(downModule);
	}
		
	return ret;
}

void Node::doTraverseBottomUp(Action* act)
{
	act->start(this);

	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); iter++)
	{
		(*iter)->traverseBottomUp(act);
	}

	act->process(this);

	act->end(this);
}

void Node::doTraverseTopDown(Action* act)
{
	act->start(this);
	act->process(this);

	ListPtr<Node>::iterator iter = m_children.begin();
	for (; iter != m_children.end(); iter++)
	{
		(*iter)->doTraverseTopDown(act);
	}

	act->end(this);
}

void Node::traverseBottomUp(Action* act)
{
	doTraverseBottomUp(act);
}

void Node::traverseTopDown(Action* act)
{
	doTraverseTopDown(act);
}

bool Node::attachField(Field* field, std::string name, std::string desc, bool autoDestroy /*= true*/)
{
	field->setParent(this);
	field->setObjectName(name);
	field->setDescription(desc);
	field->setAutoDestroy(autoDestroy);

	bool ret = false;
	
	auto fType = field->getFieldType();
	switch (field->getFieldType())
	{
	case FieldType::Current:
		ret = this->getMechanicalState()->addOutputField(field);
		break;

	case FieldType::Param:
		ret = this->addField(field);

	default:
		break;
	}
	

	if (!ret)
	{
		Log::sendMessage(Log::Error, std::string("The field ") + name + std::string(" already exists!"));
	}
	return ret;
}

bool Node::addNodePort(NodePort* port)
{
	m_node_ports.push_back(port);

	return true;
}

void Node::setAsCurrentContext()
{
	getContext()->enable();
}

// void Node::setTopologyModule(std::shared_ptr<TopologyModule> topology)
// {
// 	if (m_topology != nullptr)
// 	{
// 		deleteModule(m_topology);
// 	}
// 	m_topology = topology;
// 	addModule(topology);
// }
// 
// void Node::setNumericalModel(std::shared_ptr<NumericalModel> numerical)
// {
// 	if (m_numerical_model != nullptr)
// 	{
// 		deleteModule(m_numerical_model);
// 	}
// 	m_numerical_model = numerical;
// 	addModule(numerical);
// }
// 
// void Node::setCollidableObject(std::shared_ptr<CollidableObject> collidable)
// {
// 	if (m_collidable_object != nullptr)
// 	{
// 		deleteModule(m_collidable_object);
// 	}
// 	m_collidable_object = collidable;
// 	addModule(collidable);
// }

std::shared_ptr<Module> Node::getModule(std::string name)
{
	std::shared_ptr<Module> base = nullptr;
	std::list<std::shared_ptr<Module>>::iterator iter;
	for (iter = m_module_list.begin(); iter != m_module_list.end(); iter++)
	{
		if ((*iter)->getName() == name)
		{
			base = *iter;
			break;
		}
	}
	return base;
}

bool Node::hasModule(std::string name)
{
	if (getModule(name) == nullptr)
		return false;

	return true;
}

/*Module* Node::getModule(std::string name)
{
	std::map<std::string, Module*>::iterator result = m_modules.find(name);
	if (result == m_modules.end())
	{
		return NULL;
	}

	return result->second;
}*/


bool Node::addToModuleList(std::shared_ptr<Module> module)
{
	auto found = std::find(m_module_list.begin(), m_module_list.end(), module);
	if (found == m_module_list.end())
	{
		m_module_list.push_back(module);
		module->setParent(this);
		return true;
	}

	return false;
}

bool Node::deleteFromModuleList(std::shared_ptr<Module> module)
{
	auto found = std::find(m_module_list.begin(), m_module_list.end(), module);
	if (found != m_module_list.end())
	{
		m_module_list.erase(found);
		return true;
	}

	return true;
}

}