/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: linesearch method for optimization problem.
 * @version    : 1.0
 */
#ifndef PhysIKA_LINE_SEARCH
#define PhysIKA_LINE_SEARCH

#include <memory>

#include "Common/def.h"
#include "Common/data_str_core.h"

namespace PhysIKA {
template <typename T, size_t dim_>
T line_search(const T& val_init, const T& down, const std::shared_ptr<Functional<T, dim_>>& energy, std::shared_ptr<dat_str_core<T, dim_>>& data, const T* const xk, const T* const pk);

}  //namespace PhysIKA

#endif
