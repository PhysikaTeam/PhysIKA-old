/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of BoundaryConstraint class, SDF based collision handling constraint
 * @version    : 1.0
 *
 * @author     : Zhu Fei (feizhu@pku.edu.cn)
 * @date       : 2021-07-26
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once

#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldArray.h"

namespace PhysIKA {

template <typename TDataType>
class DistanceField3D;

/**
 * BoundaryConstraint, handling collision with objects represented as SDF
 * Usage:
 * 1. Define a BoundaryConstraint instance
 * 2. initialize the constraint by loading from sdf file or analytical representations
 * 3. setup friction coefficients if necessary
 * 4. Apply constraint with specified position&&velocity by calling constrain() when needed
 */
template <typename TDataType>
class BoundaryConstraint : public ConstraintModule
{
    DECLARE_CLASS_1(BoundaryConstraint, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    BoundaryConstraint();
    ~BoundaryConstraint() override;

    /**
     * handle collision between sdf and m_position&&m_velocity
     * m_position&&m_velocity need to be setup before calling this API
     * [Deprecated] Use constrain() with arguments instead
     *
     * @return true if no error occurs, false otherwise
     */
    bool constrain() override;

    /**
     * handle collision between sdf and specified position&&velocity
     *
     * @param[in&&out] position    array of simulated positions
     * @param[in&&out] velocity    array of simulated velocities
     * @param[in]      dt          time step
     *
     * @return true if no error occurs, false otherwise
     */
    bool constrain(DeviceArray<Coord>& position, DeviceArray<Coord>& velocity, Real dt);

    /**
     * load sdf from file
     *
     * @param[in] filename    file name of the SDF file
     * @param[in] inverted    whether to invert the sdf sign after the load
     */
    void load(std::string filename, bool inverted = false);

    /**
     * initialize sdf with a cuboid
     *
     * @param[in] lo          coordinate of the cuboid's lower corner
     * @param[in] hi          coordinate of the cuboid's higher corner
     * @param[in] distance    sampling distance of the sdf
     * @param[in] inverted    the outside of the cuboid is positive by default,
     *                        the sdf will be inverted if the argument is set to true
     */
    void setCube(Coord lo, Coord hi, Real distance, bool inverted = false);

    /**
     * initialize sdf with a sphere
     *
     * @param[in] center      center of the sphere
     * @param[in] r           radius of the sphere
     * @param[in] distance    sampling distance of the sdf
     * @param[in] invertedthe outside of the sphere is positive by default,
     *                        the sdf will be inverted if the argument is set to true
     */
    void setSphere(Coord center, Real r, Real distance, bool inverted = false);

public:
    DeviceArrayField<Coord> m_position;  //!< position vector used in no-argument version of constrain()
    DeviceArrayField<Coord> m_velocity;  //!< position vector used in no-argument version of constrain()

    Real m_normal_friction  = 0.95f;  //!< friction coefficient in normal direction, value in range [0, 1]
    Real m_tangent_friction = 0.0;    //!< friction coefficient in tangital direction, value in range [0, 1]

    std::shared_ptr<DistanceField3D<TDataType>> m_cSDF;  //!< sdf representation of the subject boundary
};

#ifdef PRECISION_FLOAT
template class BoundaryConstraint<DataType3f>;
#else
template class BoundaryConstraint<DataType3d>;
#endif

}  // namespace PhysIKA
