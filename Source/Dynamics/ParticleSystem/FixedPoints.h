/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2020-06-11
 * @description: Declaration of FixedPoints class, applying fix-point constraint
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-4
 * @description: add comments
 * @version    : 1.1
 */

#pragma once
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldArray.h"

#include <map>

namespace PhysIKA {

/**
 * FixedPoints, applying fix-point constraint
 * Usage:
 * 1. Add the IDs and positions of the fixed points using addFixedPoint
 * 2. Apply constraint with specified position&&velocity by calling constrain() when needed
 */

template <typename TDataType>
class FixedPoints : public ConstraintModule
{
    DECLARE_CLASS_1(FixedPoints, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    FixedPoints();
    ~FixedPoints() override;

    /**
     * add a fixed point constraint to the host array
     *
     * @param[in] id          center of the sphere
     * @param[in] pt          radius of the sphere
     */
    void addFixedPoint(int id, Coord pt);
    /**
     * remove a fixed point from host array according to ID
     *
     * @param[in] id          the ID of the point to be deleted
     * 
     */
    void removeFixedPoint(int id);

    void clear();

    /**
     * handle the fixed point constraints, positions are set to be constant and velocities to be zero
     * addFixedPoint should be called to initialize before calling this API
     *
     * @return true if no error occurs, false otherwise
     */
    bool constrain() override;

    void constrainPositionToPlane(Coord pos, Coord dir);

public:
    /**
        * @brief Particle position
        */
    DeviceArrayField<Coord> m_position;

    /**
        * @brief Particle velocity
        */
    DeviceArrayField<Coord> m_velocity;

protected:
    virtual bool initializeImpl() override;

    FieldID m_initPosID;

private:
    /**
     * Copy the host ID and positions to device array
     */
    void updateContext();

    bool bUpdateRequired = false;

    std::map<int, Coord> m_fixedPts;  //!< CPU vector, used to store fix IDs and positions to initialize

    std::vector<int>   m_bFixed_host;
    std::vector<Coord> m_fixed_positions_host;

    DeviceArray<int>   m_bFixed;           //!< GPU array
    DeviceArray<Coord> m_fixed_positions;  //!< GPU array of fixed positions
};

#ifdef PRECISION_FLOAT
template class FixedPoints<DataType3f>;
#else
template class FixedPoints<DataType3d>;
#endif

}  // namespace PhysIKA
