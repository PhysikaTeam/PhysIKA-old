/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of Helmholtz class, which implements the particle shifting in section 4.3
 *               introduced in the paper <A Variational Staggered Particle Framework for Incompressible Free-Surface Flows>
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: poslish code
 * @version    : 1.1
 */

#pragma once
#include "Core/Array/Array.h"
#include "Framework/Framework/ModuleConstraint.h"
#include "Framework/Topology/FieldNeighbor.h"

namespace PhysIKA {

/**
 * Helmholtz implements the particle shifting of the paper
 * <A Variational Staggered Particle Framework for Incompressible Free-Surface Flows>
 * It can be used in any fluid-related class
 */

template <typename TDataType>
class SummationDensity;

template <typename TDataType>
class Helmholtz : public ConstraintModule
{
    DECLARE_CLASS_1(Helmholtz, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    Helmholtz();
    ~Helmholtz() override;
    /**
     * used to apply particle shifting as well as surface tension for fluids
     * m_posID, m_velID and m_neighborhoodID should be set by calling the corresponding API before running this function 
     * 
     * @return true if no error occurs, false otherwise
     */
    bool constrain() override;

    /**
     * set position ID, used to get input position
     *
     * @param[in] id          the ID of the point position
     */
    void setPositionID(FieldID id)
    {
        m_posID = id;
    }
    /**
     * set velocity ID, used to get input velocity
     *
     * @param[in] id          the ID of the point velocity
     * 
     */
    void setVelocityID(FieldID id)
    {
        m_velID = id;
    }
    /**
     * set neighborhood ID, used to get input neighborhood
     *
     * @param[in] id          the ID of the neighborhood
     * 
     */
    void setNeighborhoodID(FieldID id)
    {
        m_neighborhoodID = id;
    }
    /**
     * set iteration number, used to set iteration number
     *
     * @param[in] n          the number of iterations
     * 
     */
    void setIterationNumber(int n)
    {
        m_maxIteration = n;
    }
    /**
     * used to set searching radius and to calculate kernel function of SPH
     *
     * @param[in] len          a positive real number, indicating the smooghing length
     * 
     */
    void setSmoothingLength(Real len)
    {
        m_smoothingLength = len;
    }

    void computeC(DeviceArray<Real>& c, DeviceArray<Coord>& pos, NeighborList<int>& neighbors);
    void computeGC();
    void computeLC(DeviceArray<Real>& lc, DeviceArray<Coord>& pos, NeighborList<int>& neighbors);

    void setReferenceDensity(Real rho)
    {
        m_referenceRho = rho;
    }

protected:
    bool initializeImpl() override;

protected:
    FieldID m_posID;
    FieldID m_velID;
    FieldID m_neighborhoodID;

private:
    bool m_bSetup;

    int  m_maxIteration;
    Real m_smoothingLength;
    Real m_referenceRho;

    Real m_scale;
    Real m_lambda;
    Real m_kappa;

    DeviceArray<Real>  m_c;
    DeviceArray<Real>  m_lc;
    DeviceArray<Real>  m_energy;
    DeviceArray<Coord> m_bufPos;
    DeviceArray<Coord> m_originPos;
};

#ifdef PRECISION_FLOAT
template class Helmholtz<DataType3f>;
#else
template class Helmholtz<DataType3d>;
#endif
}  // namespace PhysIKA