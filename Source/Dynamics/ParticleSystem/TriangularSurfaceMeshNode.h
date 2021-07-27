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

#pragma once

#include "Framework/Framework/Node.h"
namespace PhysIKA {
template <typename TDataType>
class TriangleSet;

/**
 * TriangularSurfaceMeshNode, a deformable surface mesh with reference configuration and deformed configuration
 */
template <typename TDataType>
class TriangularSurfaceMeshNode : public Node
{
    DECLARE_CLASS_1(ParticleSystem, TDataType)
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;

    TriangularSurfaceMeshNode(std::string name = "TriangularSurfaceMeshNode");
    virtual ~TriangularSurfaceMeshNode();

    /**
     * translate the reference configuration
     *
     * @param[in] t    the translation vector
     *
     * @return    true if succeed, false otherwise
     */
    virtual bool translate(Coord t);

    /**
     * scale the reference configuration
     *
     * @param[in] s    the scale factor, must be positive
     *
     * @return    true if succeed, false otherwise
     */
    virtual bool scale(Real s);

    /**
     * return pointer to current vertex position array
     */
    DeviceArrayField<Coord>* getVertexPosition()
    {
        return &m_vertex_position;
    }

    /**
     * return pointer to current vertex velocity array
     */
    DeviceArrayField<Coord>* getVertexVelocity()
    {
        return &m_vertex_velocity;
    }

    /**
     * return pointer to current vertex force array
     */
    DeviceArrayField<Coord>* getVertexForce()
    {
        return &m_vertex_force;
    }

    /**
     * return pointer to current triangle index array
     */
    DeviceArrayField<Triangle>* getTriangleIndex()
    {
        return &m_triangle_index;
    }

    /**
     * return reference triangle mesh
     */
    std::shared_ptr<TriangleSet<TDataType>> getTriangleSet()
    {
        return m_triSet;
    }

    /**
     * Use current configuration as the new reference configuration
     * Function ignored if current node is not active
     */
    void updateTopology() override;

    /**
     * reset current configuration using reference configuration
     */
    bool resetStatus() override;

public:
    bool initialize() override;

protected:
    DeviceArrayField<Coord>                 m_vertex_position;  //!< vertex position of current configuration
    DeviceArrayField<Coord>                 m_vertex_velocity;  //!< vertex velocity of current configuration
    DeviceArrayField<Coord>                 m_vertex_force;     //!< vertex force of current configuration
    DeviceArrayField<Triangle>              m_triangle_index;   //!< triangle index of current configuration
    std::shared_ptr<TriangleSet<TDataType>> m_triSet;           //!< reference configuration
};

#ifdef PRECISION_FLOAT
template class TriangularSurfaceMeshNode<DataType3f>;
#else
template class TriangularSurfaceMeshNode<DataType3d>;
#endif
}  // namespace PhysIKA