/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of ParticleWriter class, which can write particle positions and corresponding scalar values into txt files
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-05
 * @description: poslish code
 * @version    : 1.1
 */
#pragma once
#include "Framework/Framework/ModuleIO.h"
#include "Framework/Framework/ModuleTopology.h"

#include <string>

/**
 * ParticleWriter, write particle information into txt files, usually used for future rendering
 * The output format could be recognized by Particle2RealFlow
 * 
 * Usage:
 * 1. Initialize m_position and m_color_mapping
 * 2. Call execute() in each frame
 *
 */

namespace PhysIKA {

template <typename TDataType>
class TriangleSet;
template <typename TDataType>
class ParticleWriter : public IOModule
{
    DECLARE_CLASS_1(ParticleWriter, TDataType)
public:
    typedef typename TDataType::Real          Real;
    typedef typename TDataType::Coord         Coord;
    typedef typename TopologyModule::Triangle Triangle;
    ParticleWriter();
    virtual ~ParticleWriter();

    /**
     * set prefix of the output file name
     *
     * @param[in] prefix    file name prefix of the output file, note that path should not be included
     */
    void setNamePrefix(std::string prefix);

    /**
     * set path of the output file name
     *
     * @param[in] path    path of the output file
     */
    void setOutputPath(std::string path);

    /**
    *  Write information of interest info files 
    *  The output index changes each frame while the name prefix stay unchanged
    *  @return true(always)
    */
    bool execute() override;

public:
    DeviceArrayField<Coord> m_position;       //position of particles
    DeviceArrayField<Real>  m_color_mapping;  //a scalar(e.g. pressure, norm of velocity) which you want to be demostrated by color mapping

    DeviceArrayField<Triangle> m_triangle_index;  //IDs of triangle meshes, reserved for triangle mesh writer
    DeviceArrayField<Coord>    m_triangle_pos;    //Positions for triangle vertexs, reserved for triangle mesh writer

private:
    int         m_output_index = 0;  //!<the index of current frame, plus one each time called
    std::string m_output_path;       //!< the path of output
    std::string m_name_prefix;       //!< the name of output file
};

#ifdef PRECISION_FLOAT
template class ParticleWriter<DataType3f>;
#else
template class ParticleWriter<DataType3d>;
#endif
}  // namespace PhysIKA
