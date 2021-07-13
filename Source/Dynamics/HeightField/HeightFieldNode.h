/**
 * @author     : Ben Xu (xuben@mail.nankai.edu.cn) Kangrui Zhang(kangrui@nwafu.edu.cn)
 * @date       : 2021-02-01
 * @description: init the 2D height field and call shallow-water-equation model
 * @version    : 1.0
 */
#pragma once
#include "Framework/Framework/Node.h"
#include "ShallowWaterEquationModel.h"
namespace PhysIKA {
template <typename TDataType>
class HeightField;
/*!
    *    \class    HeightFieldNode
    *    \brief    A height field node.
    * 
    * Sample usage: 
    * std::shared_ptr<HeightFieldNode<DataType3f>> root = scene.createNewScene<HeightFieldNode<DataType3f>>();
    * root->loadParticles(Vector3f(0, 0, 0), Vector3f(2, 1.5, 2), 1024, 0.7, 1);
    * float dt = 1.0 / 60;
    * root->run(1, dt);
    */
template <typename TDataType>
class HeightFieldNode : public Node
{
    DECLARE_CLASS_1(HeightFieldNode, TDataType)
public:
    typedef typename TDataType::Real  Real;
    typedef typename TDataType::Coord Coord;

    HeightFieldNode(std::string name = "default");
    virtual ~HeightFieldNode();

    bool initialize() override;
    void advance(Real dt) override;
    void SWEconnect();
    /**
             * implementation of init square height field scene
             *
             * @param[in]    lo        the minimum coord of the scene
             * @param[in]    hi        the maximum coord of the scene
             * @param[in]    pixels    number of pixels along one dimension
             * @param[in]    slope   control the steepness of the mountain
             * @param[in]    relax    energy dissipation coefficient, between (0,1]
         */
    void loadParticles(Coord lo, Coord hi, int pixels, Real slope, Real relax);

    /**
             * implementation of init square height field scene
             *
             * @param[in]    filename1        the file path of terrain image 
             * @param[in]    filename2        the file path of river image
             * @param[in]    proportion        parameter mapping the terrain height from picture to scene, which is always greater than zero
             * @param[in]    relax            energy dissipation coefficient, between (0,1]
         */
    void loadParticlesFromImage(std::string filename1, std::string filename2, Real proportion, Real relax);
    /**
         * run swe model
         *
         * @param[in]   stepNum     the number of timesteps need to run
         * @param[in]    timestep    the length of a timestep
         */
    void run(int stepNum, float timestep);
    void init();
    /**
             * implementation of output the result
        */
    std::vector<Real>& outputDepth();
    std::vector<Real>& outputSolid();
    std::vector<Real>& outputUVel();
    std::vector<Real>& outputWVel();

    void updateTopology() override;

public:
    /**
         * @brief Particle position
         */
    DEF_EMPTY_CURRENT_ARRAY(Position, Coord, DeviceType::GPU, "Particle position");

    /**
         * @brief Particle velocity
         */
    DEF_EMPTY_CURRENT_ARRAY(Velocity, Coord, DeviceType::GPU, "Particle velocity");

private:
    void                    loadHeightFieldParticles(Coord lo, Coord hi, int pixels, Real slope, std::vector<Coord>& vertList);
    void                    loadHeightFieldFromImage(Coord lo, Coord hi, int pixels, Real slope, std::vector<Coord>& vertList);
    Real                    distance;  //!< distance between each neighbor grid point
    Real                    relax;     //!< energy dissipation coefficient, between(0, 1]
    DeviceArrayField<Real>  solid;     //!< solid terrain stored on GPU
    DeviceArrayField<Coord> normal;    //!< the normal direction stored on GPU
    DeviceArrayField<int>   isBound;   //!< mark whether the  node is a boundary node stored on GPU
    DeviceArrayField<Real>  h;         //!< the fluid height stored on GPU

    DeviceArrayField<Real> buffer;

    std::vector<Real> Solid;  //!< solid terrain stored on CPU
    std::vector<Real> Depth;  //!< the fluid depth stored on CPU
    std::vector<Real> UVel;   //!< fluid velocity along U direction stored on CPU
    std::vector<Real> WVel;   //!< fluid velocity along W direction stored on CPU

    int zcount = 0;
    int xcount = 0;

    int nx = 0;
    int nz = 0;

    std::shared_ptr<HeightField<TDataType>> m_height_field;
};

#ifdef PRECISION_FLOAT
template class HeightFieldNode<DataType3f>;
#else
template class HeightFieldNode<DataType3d>;
#endif
}  // namespace PhysIKA