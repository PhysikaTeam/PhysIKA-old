/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: eigen extern.
 * @version    : 1.0
 */
#ifndef EIGEN_EXT
#define EIGEN_EXT
#include <Eigen/Dense>

namespace PhysIKA {

/**
 * indexing functor, for easizer access.
 *
 */
template <class ArgType, class RowIndexType, class ColIndexType>
class indexing_functor
{

    const ArgType&      m_arg;
    const RowIndexType& m_rowIndices;
    const ColIndexType& m_colIndices;

public:
    typedef Eigen::Matrix<typename ArgType::Scalar,
                          RowIndexType::SizeAtCompileTime,
                          ColIndexType::SizeAtCompileTime,
                          ArgType::Flags & Eigen::RowMajorBit ? Eigen::RowMajor : Eigen::ColMajor,
                          RowIndexType::MaxSizeAtCompileTime,
                          ColIndexType::MaxSizeAtCompileTime>
        MatrixType;
    indexing_functor(const ArgType& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
        : m_arg(arg), m_rowIndices(row_indices), m_colIndices(col_indices)
    {
    }
    //TODO: did't use Scalar&
    const typename ArgType::Scalar operator()(Eigen::Index row, Eigen::Index col) const
    {
        return m_arg(m_rowIndices[row], m_colIndices[col]);
    }
};

template <class ArgType, class RowIndexType, class ColIndexType>
Eigen::CwiseNullaryOp<indexing_functor<ArgType, RowIndexType, ColIndexType>, typename indexing_functor<ArgType, RowIndexType, ColIndexType>::MatrixType>
indexing(const Eigen::MatrixBase<ArgType>& arg, const RowIndexType& row_indices, const ColIndexType& col_indices)
{

    typedef indexing_functor<ArgType, RowIndexType, ColIndexType> Func;
    typedef typename Func::MatrixType                             MatrixType;
    return MatrixType::NullaryExpr(row_indices.size(), col_indices.size(), Func(arg.derived(), row_indices, col_indices));
}

}  // namespace PhysIKA
#endif
