/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Declaration of NeighborData, declare class TPair
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: poslish code
 * @version    : 1.1
 */
#pragma once
#include "Core/Platform.h"

/**
 * Define TPair to store the initial positions for deformable bodies
 *
 * Currently used in elastic-related classes(ElasticityModule, ElastoplasticityModule, FractureModule, GranularModule)
 */
namespace PhysIKA {
template <typename TDataType>
class TPair
{
public:
    typedef typename TDataType::Coord Coord;

    COMM_FUNC TPair(){};
    COMM_FUNC TPair(int id, Coord p)
    {
        index = id;
        pos   = p;
    }

    int   index;  //index of the neighboring particle
    Coord pos;    //init position of the neighboring particle
};

}  // namespace PhysIKA