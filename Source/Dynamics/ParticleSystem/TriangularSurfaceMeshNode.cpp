/**
 * @author     : Chang Yue (changyue@buaa.edu.cn)
 * @date       : 2020-09-17
 * @description: Declaration of TriangularSurfaceMeshNode class
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-27
 * @description: poslish code
 * @version    : 1.1
 */

#include "TriangularSurfaceMeshNode.h"

#include "Core/Utility.h"
#include "Framework/Topology/TriangleSet.h"

namespace PhysIKA {
IMPLEMENT_CLASS_1(TriangularSurfaceMeshNode, TDataType)

template <typename TDataType>
TriangularSurfaceMeshNode<TDataType>::TriangularSurfaceMeshNode(std::string name)
    : Node(name)
{
    attachField(&m_vertex_position, MechanicalState::position(), "Storing vertex position!", false);
    attachField(&m_vertex_velocity, MechanicalState::velocity(), "Storing vertex velocity!", false);
    attachField(&m_vertex_force, MechanicalState::force(), "Storing vertex force!", false);

    m_triSet = std::make_shared<TriangleSet<TDataType>>();
    this->setTopologyModule(m_triSet);
}

template <typename TDataType>
TriangularSurfaceMeshNode<TDataType>::~TriangularSurfaceMeshNode()
{
}

template <typename TDataType>
bool TriangularSurfaceMeshNode<TDataType>::translate(Coord t)
{
    m_triSet->translate(t);

    return true;
}

template <typename TDataType>
bool TriangularSurfaceMeshNode<TDataType>::scale(Real s)
{
    m_triSet->scale(s);

    return true;
}

template <typename TDataType>
bool TriangularSurfaceMeshNode<TDataType>::initialize()
{
    return Node::initialize();
}

template <typename TDataType>
void TriangularSurfaceMeshNode<TDataType>::updateTopology()
{
    if (this->isActive())
    {
        auto pts = m_triSet->getPoints();
        Function1Pt::copy(pts, this->getVertexPosition()->getValue());
    }
}

template <typename TDataType>
bool TriangularSurfaceMeshNode<TDataType>::resetStatus()
{
    auto pts       = m_triSet->getPoints();
    auto triangles = m_triSet->getTriangles();

    m_vertex_position.setElementCount(pts.size());
    m_vertex_velocity.setElementCount(pts.size());
    m_vertex_force.setElementCount(pts.size());
    m_triangle_index.setElementCount(triangles->size());

    Function1Pt::copy(m_vertex_position.getValue(), pts);
    Function1Pt::copy(m_triangle_index.getValue(), *triangles);
    m_vertex_velocity.getReference()->reset();

    return Node::resetStatus();
}
}  // namespace PhysIKA