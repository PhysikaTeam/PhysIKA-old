/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of SimpleDamping class, which applies a simple damping on particle velocities
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-08
 * @description: poslish code
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldArray.h"

#include <map>

namespace PhysIKA {
/**
     * @brief A simple damping on particle velocities
     * 
     * @tparam TDataType 
     * 
     * Usage:
     * (1) initialize velocities by initializing m_velocity
     * (2) call setDampingCofficient() to set damping parameter if needed
     * (3) call constrain() when needed
     */
template <typename TDataType>
class SimpleDamping : public ConstraintModule
{
    DECLARE_CLASS_1(SimpleDamping, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    SimpleDamping();
    ~SimpleDamping() override;

    /**
    * apply damping 
    */
    bool constrain() override;

    /**
     * setup damping cofficient
     *
     * @param[in]      c          the damping cofficient
     *
     * @set the damping cofficient to be c
     */
    void setDampingCofficient(Real c);

public:
    /**
        * @brief Particle velocity
        */
    DeviceArrayField<Coord> m_velocity;

protected:
    virtual bool initializeImpl() override;

private:
    VarField<float> m_damping;
};

#ifdef PRECISION_FLOAT
template class SimpleDamping<DataType3f>;
#else
template class SimpleDamping<DataType3d>;
#endif

}  // namespace PhysIKA
