/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-06-08
 * @description: Declaration of Node class, a tree node is a primitive in scene graph that generally represents
 *               an object attached with properties and actions.
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-08-03
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include <memory>

#include "Core/Typedef.h"
#include "Core/Platform.h"
#include "Core/Vector/vector_3d.h"
#include "Base.h"
#include "FieldVar.h"
#include "DeclareModuleField.h"
#include "NodePort.h"
#include "NumericalIntegrator.h"
#include "ModuleCompute.h"
#include "NumericalModel.h"
#include "ModuleForce.h"
#include "ModuleConstraint.h"
#include "ModuleCustom.h"
#include "CollisionModel.h"
#include "CollidableObject.h"
#include "ModuleVisual.h"
#include "MechanicalState.h"
#include "ModuleTopology.h"
#include "TopologyMapping.h"

namespace PhysIKA {
class Action;
class DeviceContext;
class AnimationController;
class RenderController;

/**
 * Node, a primitive in scene graph.
 * A scene graph is composed of node hierarchies.
 * Node class is the base class of all nodes.
 *
 * A node is designed to be an object with properties and actions.
 * However, this philosophy is not strictly enforced by PhysIKA developers.
 * There're subclasses of Node that are implementation of algorithms on objects in scene, e.g., SFINode.
 * Just let it be...
 * Nevertheless, we should follow the design philosophy as much as we can while writing new code.
 *
 * Node class models object properties as member variables or Fields.
 * The algorithms (e.g., simulation pipelines) are modeled as ordered execution of Modules.
 * Please refer to corresponding headers for description of these concepts.
 *
 * Usage:
 * 1. Define a node instance
 * 2. Add the node to scene graph by calling SceneGraph API
 * 3. Attach some modules (simulation algorithm, rendering action, etc.) to the node
 * 4. Enter simulation loop
 *
 * For subclass implementers:
 * 1. Inherit from Node class
 * 2. Override inheritted API
 * 3. Add member variables/methods if needed
 */
class Node : public Base
{
    DECLARE_CLASS(Node)
public:
    /**
     * construct a node with name
     *
     * @param[in] name    name of the node
     */
    Node(std::string name = "default");
    virtual ~Node();

    /**
     * set the name of the node
     *
     * @param[in] name    name of the node
     */
    void setName(std::string name);

    /**
     * get the name of the node
     *
     * @return    a string representing name of the node
     */
    std::string getName() const;

    /**
     * get child node by name
     *
     * @param[in] name    name of the child
     *
     * @return    Pointer to the child, return nullptr if there's no child with the name
     */
    Node* getChild(std::string name);

    /**
     * get parent node
     *
     * @return    Pointer to the parent, return nullptr if the node is scene graph root
     */
    Node* getParent();

    /**
     * get root node of the host scene graph
     *
     * @return    Pointer to the scene graph
     */
    Node* getRoot();

    /**
     * Query the controllable state of the node
     * The design of controllable state seems to be deprecated.
     *
     * @return    true if the node is controllable, false otherwise
     */
    bool isControllable() const;

    /**
     * Set the controllable state of the node
     *
     * @param[in] con    the expected controllable state
     */
    void setControllable(bool con);

    /**
     * Query the simulation state of the node.
     * A node's state will be updated in simulation pipeline if it is active.
     *
     * @return    simulation state of the node
     */
    virtual bool isActive();

    /**
     * Set the simulation state of the node
     *
     * @param[in] active    the expected state of the node
     */
    virtual void setActive(bool active);

    /**
     * Query the visibility of the node
     * A node will not be rendered if it is not visible
     *
     * @return    visibility of the node
     */
    virtual bool isVisible();

    /** Set visibility of the node
     *
     * @param[in] visible    the expected visibility of the node
     *
     */
    virtual void setVisible(bool visible);

    /**
     * Query the time step of simulation
     *
     * @return    time step of the simulation
     */
    virtual Real getDt() const;

    /**
     * Set the simulation time step
     *
     * @param[in] dt    time step of simulation
     */
    void setDt(Real dt);

    /**
     * Set the mass of the node
     *
     * @param[in] mass    the specified node mass, must be positive
     *                    no validity check
     */
    void setMass(Real mass);

    /**
     * Query the mass of the node
     *
     * @return    mass of the node
     */
    Real getMass() const;

    /**
     * Create a child node by name
     *
     * @tparam TNode    type of the child node
     *
     * @param[in] name    name of the child node
     *
     * @return    pointer to the created child node
     *            return nullptr if name is aleady used. （TODO: seems not true).
     */
    template <class TNode>
    std::shared_ptr<TNode> createChild(std::string name)
    {
        return addChild(TypeInfo::New<TNode>(name));
    }

    /**
     * add a node as child
     *
     * @param[in] child    pointer to the child node, must be nullptr
     *                     access violation occurs if child is nullptr
     *
     * @return    pointer to the child
     */
    std::shared_ptr<Node> addChild(std::shared_ptr<Node> child)
    {
        m_children.push_back(child);
        child->setParent(this);
        return child;
    }

    /**
     * Check if a given node is child node of current node
     *
     * @param[in] child    the child node to be queried
     *
     * @return    true if given node is a child otherwise return false
     */
    bool hasChild(std::shared_ptr<Node> child);

    /**
     * Remove given node from child node list
     * Nothing happens if given node is not a child
     *
     * @param[in] child    the node to be removed from child list
     *
     */
    void removeChild(std::shared_ptr<Node> child);

    /**
     * Clear child node list
     */
    void removeAllChildren();

    /**
     * Return child node list
     *
     * @return list of child nodes
     */
    ListPtr<Node>& getChildren()
    {
        return m_children;
    }

    /**
     * Device context management
     * A DeviceContext contains device info on which simulation of current node runs
     *
     * Note: The design of DeviceContext seems to be incomplete, and it is kind of deprecated.
     * TODO(Zhu Fei): fix it in future versions.
     */
    std::shared_ptr<DeviceContext> getContext();
    void                           setContext(std::shared_ptr<DeviceContext> context);
    virtual void                   setAsCurrentContext();

    /**
     * Return pointer to the MechanicalState Module attached to the node.
     * Create one and attach it to the node if there's none.
     * A MechanicalState is a Module that contains common mechanical state fields.
     *
     * @return pointer to the MechanicalState Module.
     */
    std::shared_ptr<MechanicalState> getMechanicalState();

    /**
     * Set MechanicalState Module. The old one is replaced if there is one.
     *
     * @param[in] state    pointer to the new MechanicalState
     */
    void setMechanicalState(std::shared_ptr<MechanicalState> state);

    /**
     * Add a module to module list of current node
     *
     * @param[in] module    pointer to the module to be added
     *
     * @return    true if successfullly added false if module already exists in module list
     */
    bool addModule(std::shared_ptr<Module> module);

    /**
     * Remove the module from module list of current node
     *
     * @param[in] module    pointer to the module to be deleted
     *
     * @return    always return true
     */
    bool deleteModule(std::shared_ptr<Module> module);

    /**
     * Add&&Remove modules that are subclass of Module.
     * @tparam TModule    the module that is to be added or removed
     *                    the class type must be subclass of Module
     */
    template <class TModule>
    bool addModule(std::shared_ptr<TModule> tModule)
    {
        std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
        return addModule(module);
    }
    template <class TModule>
    bool deleteModule(std::shared_ptr<TModule> tModule)
    {
        std::shared_ptr<Module> module = std::dynamic_pointer_cast<Module>(tModule);
        return deleteModule(module);
    }

    /**
     * Return reference to the module list
     */
    std::list<std::shared_ptr<Module>>& getModuleList()
    {
        return m_module_list;
    }

    /**
     * Query if a module is in module list by name
     *
     * @param[in] name    name of the module to be queried
     *
     * @return    true if the module is in module list, false otherwise
     */
    bool hasModule(std::string name);

    /**
     * Get a module by its name
     *
     * @param name     Module name
     *
     * @return nullptr is no module is found, otherwise return the first found module
     */
    std::shared_ptr<Module> getModule(std::string name);

    /**
     * Get the Module by the module class name
     *
     * @tparam TModule     Module class name
     *
     * @return nullptr is no module is found, otherwise return the first found module
     */
    template <class TModule>
    std::shared_ptr<TModule> getModule()
    {
        TModule*                                     tmp = new TModule;
        std::shared_ptr<Module>                      base;
        std::list<std::shared_ptr<Module>>::iterator iter;
        for (iter = m_module_list.begin(); iter != m_module_list.end(); iter++)
        {
            if ((*iter)->getClassInfo() == tmp->getClassInfo())
            {
                base = *iter;
                break;
            }
        }
        delete tmp;
        return TypeInfo::cast<TModule>(base);
    }

    /**
     * Get a module by its name and cast it to specified type
     *
     * @tparam TModule the expected type of the module
     *
     * @param name     Module name
     *
     * @return         nullptr is no module is found or the module with given name is not specified type,
     *                 otherwise return the first found module
     */
    template <class TModule>
    std::shared_ptr<TModule> getModule(std::string name)
    {
        std::shared_ptr<Module> base = getModule(name);

        return TypeInfo::cast<TModule>(base);
    }

/**
 * The macro to define a method that sets a special module for the node by name
 * E.G., NODE_SET_SPECICAL_MODULE(TopologyModule) expands to definition of a method
 * with the name setTopologyModule, where the parameter is the module name
 *
 * @param[in] name    name of the module to be set
 *
 * @return    nullptr if a module with the name already exists in the module list
 *            otherwise return pointer to the module
 */
#define NODE_SET_SPECIAL_MODULE(CLASSNAME)                                                                 \
    template <class TModule>                                                                               \
    std::shared_ptr<TModule> set##CLASSNAME(std::string name)                                              \
    {                                                                                                      \
        if (hasModule(name))                                                                               \
        {                                                                                                  \
            Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!")); \
            return nullptr;                                                                                \
        }                                                                                                  \
                                                                                                           \
        std::shared_ptr<TModule> module = std::make_shared<TModule>();                                     \
        module->setName(name);                                                                             \
        this->set##CLASSNAME(module);                                                                      \
                                                                                                           \
        return module;                                                                                     \
    }

    /**
     * definition of setTopologyModule(std::string name)
     * A TopologyModule manages geometric representation of the node
     */
    NODE_SET_SPECIAL_MODULE(TopologyModule)

    /**
     * definition of setNumericalModel(std::string name)
     * A NumericalModel is generally a numerical simulation method, e.g., PBD, PDM, SPH, etc.
     */
    NODE_SET_SPECIAL_MODULE(NumericalModel)

    /**
     * definition of setCollidableObject(std::string name)
     * A CollidableObject is a kinematic object in scene that can collide with simulated objects
     */
    NODE_SET_SPECIAL_MODULE(CollidableObject)

    /**
     * definition of setNumericalIntegrator(std::string name)
     * A NumericalIntegrator perform classical integration algorithm to update the state of host node
     */
    NODE_SET_SPECIAL_MODULE(NumericalIntegrator)

    /**
     * The macro to define a method that sets a special module for the node
     * E.G., NODE_SET_SPEICIAL_MODULE(TopologyModule, m_topology) expands to definition of
     * a method setTopologyModule(std::shared_ptr<TopologyModule> module). The module sets
     * m_topology member variable of the node, and add module to module list.
     */
#undef NODE_SET_SPECIAL_MODULE
#define NODE_SET_SPECIAL_MODULE(CLASSNAME, MODULENAME)             \
    virtual void set##CLASSNAME(std::shared_ptr<CLASSNAME> module) \
    {                                                              \
        if (MODULENAME != nullptr)                                 \
            deleteFromModuleList(MODULENAME);                      \
        MODULENAME = module;                                       \
        addToModuleList(module);                                   \
    }

    NODE_SET_SPECIAL_MODULE(TopologyModule, m_topology)                   // definition of setTopologyModule(std::shared_ptr<TopologyModule> module)
    NODE_SET_SPECIAL_MODULE(NumericalModel, m_numerical_model)            // definition of setNumericalModel(std::shared_ptr<NumericalModel> module)
    NODE_SET_SPECIAL_MODULE(CollidableObject, m_collidable_object)        // definition of setCollidableObject(std::shared_ptr<CollidableObject> module)
    NODE_SET_SPECIAL_MODULE(NumericalIntegrator, m_numerical_integrator)  // definition of setNumericalIntegrator(std::shared_ptr<NumericalIntegrator> module)

    /**
     * Return pointers to queried special modules
     */
    std::shared_ptr<CollidableObject> getCollidableObject()
    {
        return m_collidable_object;
    }
    std::shared_ptr<NumericalModel> getNumericalModel()
    {
        return m_numerical_model;
    }
    std::shared_ptr<TopologyModule> getTopologyModule()
    {
        return m_topology;
    }
    std::shared_ptr<NumericalIntegrator> getNumericalIntegrator()
    {
        return m_numerical_integrator;
    }

    /**
     * AnimationController&&RenderController was designed to decouple simulation&&render
     * workflow from node. However, the design is incomplete and not used.
     *
     */
    std::unique_ptr<AnimationController>& getAnimationPipeline();
    std::unique_ptr<RenderController>&    getRenderPipeline();

    /**
     * add a module with given name and type to the module list of the node
     *
     * @tparam TModule    type of the module to be added
     *
     * @param[in] name    name of the module to be added
     *
     * @return    pointer to the newly added module, return nullptr if there's already a module with given name in
     *            module list
     */
    template <class TModule>
    std::shared_ptr<TModule> addModule(std::string name)
    {
        if (hasModule(name))
        {
            Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!"));
            return nullptr;
        }
        std::shared_ptr<TModule> module = std::make_shared<TModule>();
        module->setName(name);
        this->addModule(module);

        return module;
    }

    /**
     * The macro to define a method that add a special module with given name and type to the module list
     * There could exist multiple modules with the same type in the module list
     * E.G., NODE_CREATE_SPECIALMODULE(ConstraintModule) expands to definition of  a method addConstraintModule(std::string name)
     *
     * @tparam TModule    type of the module to be added
     *
     * @param[in] name    name of the module to be added
     *
     * @return    pointer to the newly added module, return nullptr if a module with given name already exits
     */
#define NODE_CREATE_SPECIAL_MODULE(CLASSNAME)                                                              \
    template <class TModule>                                                                               \
    std::shared_ptr<TModule> add##CLASSNAME(std::string name)                                              \
    {                                                                                                      \
        if (hasModule(name))                                                                               \
        {                                                                                                  \
            Log::sendMessage(Log::Error, std::string("Module ") + name + std::string(" already exists!")); \
            return nullptr;                                                                                \
        }                                                                                                  \
        std::shared_ptr<TModule> module = std::make_shared<TModule>();                                     \
        module->setName(name);                                                                             \
        this->add##CLASSNAME(module);                                                                      \
                                                                                                           \
        return module;                                                                                     \
    }

    /**
     * The macro to define 3 methods for managing special modules.
     * E.G., NODE_ADD_SPECIAL_MODULE(ForceModule, m_force_list) expands to 3 methods, namely
     * addForceModule(std::shared_ptr<ForceModule> module)
     * deleteForceModule(std::shared_ptr<ForceModule> module)
     * getForceModuleList()
     *
     */
#define NODE_ADD_SPECIAL_MODULE(CLASSNAME, SEQUENCENAME)              \
    virtual void add##CLASSNAME(std::shared_ptr<CLASSNAME> module)    \
    {                                                                 \
        SEQUENCENAME.push_back(module);                               \
        addToModuleList(module);                                      \
    }                                                                 \
    virtual void delete##CLASSNAME(std::shared_ptr<CLASSNAME> module) \
    {                                                                 \
        SEQUENCENAME.remove(module);                                  \
        deleteFromModuleList(module);                                 \
    }                                                                 \
    std::list<std::shared_ptr<CLASSNAME>>& get##CLASSNAME##List()     \
    {                                                                 \
        return SEQUENCENAME;                                          \
    }

    /**
     * Definition of addForceModule, deleteForceModule, getForceModuleList
     * A ForceModule is computation of a force formulation, e.g., gravity
     */
    NODE_ADD_SPECIAL_MODULE(ForceModule, m_force_list)

    /**
     * Deinition of addConstraintModule, deleteConstraintModule, getConstraintModuleList
     * A ConstraintModule is a mechanism to model computation as constraints
     */
    NODE_ADD_SPECIAL_MODULE(ConstraintModule, m_constraint_list)

    /**
     * Definition of addCollisionModel, deleteCollisionModel, getCollisionModelList
     * A CollisionModel handles collision detection and resolution between host node and collidable objects
     */
    NODE_ADD_SPECIAL_MODULE(CollisionModel, m_collision_list)

    /**
     * Definition of addVisualModule, deleteVisualModule, getVisualModuleList
     * A VisualModule renders the host node data
     */
    NODE_ADD_SPECIAL_MODULE(VisualModule, m_render_list)

    /**
     * Definition of addTopologyMapping, deleteTopologyMapping, getTopologyMappingList
     * A TopologyMapping manages correspondence between different data representations, e.g., particles && meshes
     */
    NODE_ADD_SPECIAL_MODULE(TopologyMapping, m_topology_mapping_list)

    /**
     * Definition of addComputeModule, deleteComputeModule, getComputeModuleList
     * A ComputeModule performs some general purpose computations
     */
    NODE_ADD_SPECIAL_MODULE(ComputeModule, m_compute_list)

    /**
     * Definition of addCustomModule, deleteCustomModule, getCustomModuleList
     * A CustomModule is designed for subclass implementers to implement custom functionalities
     */
    NODE_ADD_SPECIAL_MODULE(CustomModule, m_custom_list)

    NODE_CREATE_SPECIAL_MODULE(ForceModule)       //definition of addForceModule(std::string name)
    NODE_CREATE_SPECIAL_MODULE(ConstraintModule)  //definition of addConstraintModule(std::string name)
    NODE_CREATE_SPECIAL_MODULE(CollisionModel)    //definition of addCollisionModel(std::string name)
    NODE_CREATE_SPECIAL_MODULE(VisualModule)      //definition of addVisualModule(std::string name)
    NODE_CREATE_SPECIAL_MODULE(TopologyMapping)   //definition of addTopologyMapping(std::string name)
    NODE_CREATE_SPECIAL_MODULE(ComputeModule)     //definition of addComputeModule(std::string name)
    NODE_CREATE_SPECIAL_MODULE(CustomModule)      //definition of addCustomModule(std::string name)

    /**
     * Initialize the node after construction
     *
     * @return    the initialization status
     */
    virtual bool initialize()
    {
        return true;
    }

    /**
     * Deprecated
     */
    virtual void draw() {}

    /**
     * advance the simulation with specified time step
     *
     * @param[in] dt    the time step
     */
    virtual void advance(Real dt);

    /**
     * advance the simulation for one frame
     * One frame might be composed of several time steps, depending on the specified framerate
     * Never used in subclasses
     */
    virtual void takeOneFrame() {}

    /**
     * update the states of the modules?
     * Never used in subclasses
     */
    virtual void updateModules() {}

    /**
     * update the topology representation of the node
     * Note: the design motivation of this method is unclear, and it is implemented for different
     * purposes by subclass developers.
     */
    virtual void updateTopology() {}

    /**
     * Reset simulation state to intial state (very much like initialize...)
     *
     * @return    true if succeed, false otherwise
     */
    virtual bool resetStatus()
    {
        return true;
    }

    /**
     * update the simulation state according to some rule during simulation?
     * Never used in subclasses
     */
    virtual void updateStatus() {}

    /**
     * apply the topology mapping modules of the node
     */
    virtual void applyTopologyMappings();

    /**
     * Perform action on the node hierarchies rooted at current node using
     * a depth-first bottom-up approach
     *
     * @param[in] act  the action to perform on nodes
     */
    void traverseBottomUp(Action* act);

    /**
     * Perform action with arguments on the node hierarchies rooted at current node using
     * a depth-first bottom-up approach
     *
     * @tparam Act    the action type
     * @tparam Args   the arguments types
     * @param[in] args    the arguments of the action
     */
    template <class Act, class... Args>
    void traverseBottomUp(Args&&... args)
    {
        Act action(std::forward<Args>(args)...);
        doTraverseBottomUp(&action);
    }

    /**
     * Perform action on the node hierarchies rooted at current node using
     * a depth-first top-down approach
     *
     * @param[in] act  the action to perform on nodes
     */
    void traverseTopDown(Action* act);

    /**
     * Perform action with arguments on the node hierarchies rooted at current node using
     * a depth-first top-down approach
     *
     * @tparam Act    the action type
     * @tparam Args   the arguments types
     * @param[in] args    the arguments of the action
     */
    template <class Act, class... Args>
    void traverseTopDown(Args&&... args)
    {
        Act action(std::forward<Args>(args)...);
        doTraverseTopDown(&action);
    }

    /**
     * attach a field to current node
     *
     * @param[in] field        pointer to the field to be attached
     * @param[in] name         name of the field to be attached
     * @param[in] description  description of the field
     * @param[in] autoDestroy  The field will be destroyed by node if true, otherwise, the field should be explicitly destroyed by its creator.
     *
     * @return   true if succeed, false if the name conflicts with existing field
     */
    bool attachField(Field* field, std::string name, std::string desc, bool autoDestroy = true) override;

    /**
     * add a node port to the port list of current node
     * TODO(Zhu Fei): find out what a port is designed for
     *
     * @param[in] port    pointer to the port, must not be nullptr
     *                    no validity check
     *
     * @return  always return true
     */
    bool addNodePort(NodePort* port);

    /**
     * Get all node ports of current node
     *
     * @return    reference to node port list
     */
    std::vector<NodePort*>& getAllNodePorts()
    {
        return m_node_ports;
    }

    /**
     * set the parent of current node
     *
     * @param[in] p    pointer to specified parent node
     */
    void setParent(Node* p)
    {
        m_parent = p;
    }

protected:
    virtual void doTraverseBottomUp(Action* act);  //implementation of traverseBottomUp
    virtual void doTraverseTopDown(Action* act);   //implementation of traverseTopDown

private:
    /**
     * add a module to module list
     *
     * @param[in] module    the module to be added
     *
     * @return    true if succeed, false if the module is already in module list
     */
    bool addToModuleList(std::shared_ptr<Module> module);
    /**
     * delete a module from module list
     *
     * @param[in] module    the module to be removed
     *
     * @return    always return true
     */
    bool deleteFromModuleList(std::shared_ptr<Module> module);

    /**
     * The macro defines 2 method to add/remove a module to/from the list of special modules
     * E.G., NODE_ADD_SPECIAL_MODULE_LIST(ForceModule, m_force_list) expands to definition of
     * addToForceModuleList(std::shared_ptr<ForceModule> module)
     * deleteFromForceModuleList(std::shared_ptr<ForceModule> module)
     *
     * Modules added/removed via these 2 apis are not managed by the global module list
     */
#define NODE_ADD_SPECIAL_MODULE_LIST(CLASSNAME, SEQUENCENAME)                   \
    virtual void addTo##CLASSNAME##List(std::shared_ptr<CLASSNAME> module)      \
    {                                                                           \
        SEQUENCENAME.push_back(module);                                         \
    }                                                                           \
    virtual void deleteFrom##CLASSNAME##List(std::shared_ptr<CLASSNAME> module) \
    {                                                                           \
        SEQUENCENAME.remove(module);                                            \
    }

    NODE_ADD_SPECIAL_MODULE_LIST(ForceModule, m_force_list)
    NODE_ADD_SPECIAL_MODULE_LIST(ConstraintModule, m_constraint_list)
    NODE_ADD_SPECIAL_MODULE_LIST(CollisionModel, m_collision_list)
    NODE_ADD_SPECIAL_MODULE_LIST(VisualModule, m_render_list)
    NODE_ADD_SPECIAL_MODULE_LIST(TopologyMapping, m_topology_mapping_list)
    NODE_ADD_SPECIAL_MODULE_LIST(ComputeModule, m_compute_list)
    NODE_ADD_SPECIAL_MODULE_LIST(CustomModule, m_custom_list)
protected:
    Node* m_parent;  //!< parent node
private:
    bool                           m_controllable = true;  //!< indicate whether the node is controllable
    Real                           m_dt;                   //!< simulation time step
    bool                           m_initalized;           //!< initialization state
    Real                           m_mass;                 //!< node mass
    std::string                    m_node_name;            //!< node name
    std::shared_ptr<DeviceContext> m_context;              //!< device info, design incomplete
    ListPtr<Node>                  m_children;             //!< list of child nodes
    std::vector<NodePort*>         m_node_ports;           //!< list of node ports

    DEF_VAR(Location, Vector3f, 0, "Node location");                                        //!< var_Location and varLocation()
    DEF_VAR(Rotation, Vector3f, 0, "Node rotation");                                        //!< var_Rotation and varRotation()
    DEF_VAR(Scale, Vector3f, 0, "Node scale");                                              //!< var_Scale and varScale()
    DEF_VAR(Active, bool, true, "Indicating whether the simulation is on for this node!");  //!< var_Active and varActive()
    DEF_VAR(Visible, bool, true, "Indicating whether the node is visible!");                //!< var_Visible and varVisible()

private:
    std::list<std::shared_ptr<Module>> m_module_list;  //!< list of all modules attached to the node
    // pointers to special modules, a node can only be attached by one module of the same type
    std::shared_ptr<TopologyModule>      m_topology;
    std::shared_ptr<NumericalModel>      m_numerical_model;
    std::shared_ptr<MechanicalState>     m_mechanical_state;
    std::shared_ptr<CollidableObject>    m_collidable_object;
    std::shared_ptr<NumericalIntegrator> m_numerical_integrator;
    // lists of special modules, a node can be attached by multiple modules of the same type
    std::list<std::shared_ptr<ForceModule>>      m_force_list;
    std::list<std::shared_ptr<ConstraintModule>> m_constraint_list;
    std::list<std::shared_ptr<ComputeModule>>    m_compute_list;
    std::list<std::shared_ptr<CollisionModel>>   m_collision_list;
    std::list<std::shared_ptr<VisualModule>>     m_render_list;
    std::list<std::shared_ptr<TopologyMapping>>  m_topology_mapping_list;
    std::list<std::shared_ptr<CustomModule>>     m_custom_list;

    // controllers, seems not used
    std::unique_ptr<AnimationController> m_animation_pipeline;
    std::unique_ptr<RenderController>    m_render_pipeline;
};
}  // namespace PhysIKA
