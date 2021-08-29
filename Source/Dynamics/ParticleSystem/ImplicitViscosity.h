/**
 * @author     : He Xiaowei (xiaowei@iscas.ac.cn)
 * @date       : 2020-10-07
 * @description: Declaration of ImplicitViscosity class, which implements a simpler XSPH style of damping noise
 *               For more details, please refer to [Schechter et al. 2012] "Ghost SPH for Animating Water"
 * @version    : 1.0
 * 
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-05
 * @description: poslish code
 * @version    : 1.1
 * 
 */
#pragma once
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Framework/FieldVar.h"
#include "Framework/Framework/FieldArray.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA {
template <typename TDataType>
class ImplicitViscosity : public ConstraintModule
{
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    ImplicitViscosity();
    ~ImplicitViscosity() override;

    /**
     * apply the viscosity, update the velocity device array
     *
     * m_position&&m_neighborhood need to be setup before calling this API
     * 
     * @return true(always)
     *                      
     */
    bool constrain() override;

    /**
     * set the number of iterations
     *
     * @param[in] n          the number of iterations
     *
     */
    void setIterationNumber(int n);

    /**
     * set the tunable parameter m_viscosity
     *
     * @param[in] mu          the value of viscosity
     * 
     */
    void setViscosity(Real mu);

protected:
    bool initializeImpl() override;

public:
    VarField<Real> m_viscosity;  //the tunable parameter in eq(2), default 0.05
    VarField<Real> m_smoothingLength;

    DeviceArrayField<Coord> m_velocity;  //input&output velocity array
    DeviceArrayField<Coord> m_position;  //input positio array

    NeighborField<int> m_neighborhood;  //input neighbor list

private:
    int m_maxInteration;

    DeviceArray<Coord> m_velOld;
    DeviceArray<Coord> m_velBuf;
};

#ifdef PRECISION_FLOAT
template class ImplicitViscosity<DataType3f>;
#else
template class ImplicitViscosity<DataType3d>;
#endif
}  // namespace PhysIKA