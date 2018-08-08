#pragma once
#include "Base.h"
#include "Typedef.h"

namespace Physika {
class Module;
class DeviceContext;
class Action;
class TopologyModule;
class ForceModule;
class ConstraintModule;
class CollisionModule;
class VisualModule;
class NumericalModel;
class RenderController;
class AnimationController;

class Node : public Base
{
public:
	Node(std::string name);
	virtual ~Node();

	void setName(std::string name);
	std::string getName();

	Node* getChild(std::string name);
	Node* getParent();
	Node* getRoot();

	/// Check the state of dynamics
	virtual bool isActive();

	/// Set the state of dynamics
	virtual void setActive(bool active);

	/// Check the visibility of context
	virtual bool isVisible();

	/// Set the visibility of context
	virtual void setVisible(bool visible);

	/// Simulation time
	virtual float getTime();

	/// Simulation timestep
	virtual float getDt();

	template<class TNode>
	std::shared_ptr<TNode> createChild(std::string name)
	{
		return addChild(TypeInfo::New<TNode>(name));
	}

	std::shared_ptr<Node> addChild(std::shared_ptr<Node> child) { 
		m_children.push_back(child);
		return child; 
	}

	void removeChild(std::shared_ptr<Node> child);

	ListPtr<Node> getChildren() { return m_children; }

	virtual bool initialize() { return false; }

	virtual void draw() {};
	virtual void advance(float dt) {};
	virtual void takeOneFrame() {};

	inline std::shared_ptr<DeviceContext> getContext();

	void setContext(std::shared_ptr<DeviceContext> context) { m_context = context; }

	
	bool addModule(std::string name, Module* module);

	bool addModule(Module* module);
	bool deleteModule(Module* module);

	virtual void doTraverse(Action* act);
	void traverse(Action* act);
	template<class Act>
	void traverse() {
		Act action;
		doTraverse(&action);
	}

	virtual void setAsCurrentContext();

	void setTopologyModule(TopologyModule* topology) {
		m_topology = topology;
		addModule((Module*)topology);
	}
	void setNumericalModel(NumericalModel* numerical){
		m_numerical_model = numerical;
		addModule((Module*)numerical);
	}

	NumericalModel* getNumericalModel() { return m_numerical_model; }
	TopologyModule* getTopologyModule() { return m_topology; }

	void setRenderController(RenderController* controller) { m_render_controller = controller; }
	RenderController* getRenderController() { return m_render_controller; }

	void setAnimationController(AnimationController* controller) { m_animation_controller = controller; }
	AnimationController* getAnimationController() { return m_animation_controller; }

	Module* getModule(std::string name);

	template<class TModule>
	TModule* getModule(std::string name)
	{
		Module* module = getModule(name);
		TModule* t_module = TypeInfo::CastPointerDown<TModule>(module);
		if (t_module == NULL)
		{
			return NULL;
		}

		return t_module;
	}

	template<class TModule>
	TModule* getModule()
	{
// 		TModule* tmp = new TModule;
// 		Module* base = NULL;
// 		std::list<Module*>::iterator iter;
// 		for (iter = m_module_list.begin(); iter != m_module_list.end(); iter++)
// 		{
// 			if ((*iter)->getClassInfo() == tmp->getClassInfo())
// 			{
// 				base = *iter;
// 				break;
// 			}
// 		}
// 		delete tmp;
// 		return TypeInfo::CastPointerDown<TModule>(base);
		return;
	}

	std::list<Module*>& getModuleList() { return m_module_list; }

	bool execute(std::string name);

	virtual void updateModules() {};

#define NODE_ADD_SPECIAL_MODULE( CLASSNAME, SEQUENCENAME ) \
	virtual void add##CLASSNAME( CLASSNAME* module) { SEQUENCENAME.push_back(module); addModule((Module*)module);} \
	virtual void delete##CLASSNAME( CLASSNAME* module) { SEQUENCENAME.remove(module); deleteModule((Module*)module); } \
	std::list<CLASSNAME*>& get##CLASSNAME##List(){ return SEQUENCENAME;}

	NODE_ADD_SPECIAL_MODULE(ForceModule, m_force_list)
	NODE_ADD_SPECIAL_MODULE(ConstraintModule, m_constraint_list)
	NODE_ADD_SPECIAL_MODULE(CollisionModule, m_collision_list)
	NODE_ADD_SPECIAL_MODULE(VisualModule, m_render_list)

private:
	float m_dt;

	HostVariablePtr<bool> m_active;
	HostVariablePtr<bool> m_visible;
	HostVariablePtr<float> m_time;

	HostVariablePtr<std::string> m_node_name;

	std::list<Module*> m_module_list;
	std::map<std::string, Module*> m_modules;

	TopologyModule* m_topology;
	NumericalModel* m_numerical_model;

	RenderController* m_render_controller;
	AnimationController* m_animation_controller;

	std::list<ForceModule*> m_force_list;
	std::list<ConstraintModule*> m_constraint_list;
	std::list<CollisionModule*> m_collision_list;
	std::list<VisualModule*> m_render_list;

	std::shared_ptr<DeviceContext> m_context;

	ListPtr<Node> m_children;

	Node* m_parent;
};
}
