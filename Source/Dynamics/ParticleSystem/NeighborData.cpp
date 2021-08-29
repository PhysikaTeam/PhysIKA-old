/**
 * @author     : He Xiaowei (Clouddon@sina.com)
 * @date       : 2019-05-14
 * @description: Implemendation of NeighborData, declare NeighborField for int and TPair
 * @version    : 1.0
 *
 * @author     : Chang Yue (yuechang@pku.edu.cn)
 * @date       : 2021-08-04
 * @description: poslish code
 * @version    : 1.1
 */
#include "NeighborData.h"
#include "Core/DataTypes.h"
#include "Framework/Topology/FieldNeighbor.h"
namespace PhysIKA {
template class NeighborField<int>;
template class NeighborField<TPair<DataType3f>>;
template class NeighborField<TPair<DataType3d>>;
}  // namespace PhysIKA