/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: Block Compressed Row Format (BCSR) for sparse matrix.
 * @version    : 1.0
 */
#ifndef PhysIKA_BCSR
#define PhysIKA_BCSR
#include <map>
#include <omp.h>
#include <vector>

#include <Eigen/Dense>
#include <Eigen/Sparse>

#include "error.h"
#include "DEFINE_TYPE.h"

template <typename mat_type>
using VEC_MAT = std::vector<mat_type, Eigen::aligned_allocator<mat_type>>;

namespace PhysIKA {
/**
 * block compressed row format for sparse matrix, for more effective computing.
 *
 */
template <typename T, const size_t block_size>
class BCSR
{
public:
    using ele_type = Eigen::Matrix<T, block_size, block_size>;
    using VEC      = Eigen::Matrix<T, -1, 1>;

public:
    BCSR()
        : rowsb_(0), colsb_(0), rows_(0), cols_(0), size_(0) {}
    BCSR(int rowsb, int colsb)
        : rowsb_(rowsb), colsb_(colsb), rows_(rowsb * block_size), cols_(colsb * block_size), size_(cols_ * rows_) {}
    BCSR(const BCSR<T, block_size>& other)
        : rowsb_(other.rowsb()), colsb_(other.colsb()), rows_(other.rows()), cols_(other.cols()), size_(other.size()), value_(other.get_value()), offset_(other.get_offset()), index_(other.get_index()) {}
    // BCSR(const BCSR<T, block_size>&& other)
    //     :rowsb_(other.rowsb()), colsb_(other.colsb()), rows_(other.rows()),cols_(other.cols()),size_(other.size()), value_(other.get_value()), offset_(other.get_offset()), index_(other.get_index()){}

public:
    void setFromEigenMatrix(const Eigen::SparseMatrix<T, Eigen::RowMajor>& A);

    inline void clear()
    {
        value_.clear();
        offset_.clear();
        index_.clear();
        rowsb_ = colsb_ = rows_ = cols_ = size_ = 0;
    }

    inline size_t rows() const
    {
        return rows_;
    }
    inline size_t cols() const
    {
        return cols_;
    }
    inline size_t size() const
    {
        return size_;
    }
    inline size_t rowsb() const
    {
        return rowsb_;
    }
    inline size_t colsb() const
    {
        return colsb_;
    }

    inline VEC_MAT<ele_type> get_value() const
    {
        return value_;
    }
    inline std::vector<size_t> get_offset() const
    {
        return offset_;
    }
    inline std::vector<size_t> get_index() const
    {
        return index_;
    }
    // operators override.
public:
    VEC                 operator*(const VEC& rhs) const;
    BCSR<T, block_size> operator*(const BCSR<T, block_size>& rhs) const;

    inline std::vector<ele_type, Eigen::aligned_allocator<ele_type>>
    get_diagonal() const;
    //TODO: move this value_ to protected
    VEC_MAT<ele_type> value_;

protected:
    std::vector<size_t> offset_;
    std::vector<size_t> index_;
    // number of rows and cols per block
    size_t rowsb_, colsb_;
    // number of rows and cols for the matrix.
    size_t rows_, cols_, size_;
};

template <typename T>
VEC_MAT<MAT3<T>> get_block_diagonal(const SPM_R<T>& A);

}  // namespace PhysIKA

////////////////////////////////////////////////////////////////////////
//                       template implementation                      //
////////////////////////////////////////////////////////////////////////
namespace PhysIKA {
template <typename T, const size_t block_size>
void BCSR<T, block_size>::setFromEigenMatrix(
    const Eigen::SparseMatrix<T, Eigen::RowMajor>& A)
{
    clear();
    error_msg_ext_cond(
        A.rows() % block_size != 0 || A.cols() % block_size != 0,
        "Convert Eigen Matrix to BCSR failed. Since the Matrix's dim cannot "
        "be divided by block_size. A-->(%lu, %lu) and block_size: %lu",
        A.rows(),
        A.cols(),
        block_size);
    rows_  = A.rows();
    cols_  = A.cols();
    size_  = A.size();
    rowsb_ = rows_ / block_size;
    colsb_ = cols_ / block_size;
    // very simple version.
    std::map<
        size_t,
        ele_type,
        std::less<size_t>,
        Eigen::aligned_allocator<std::pair<const size_t, ele_type>>>
        coos;
    for (size_t i = 0; i < A.outerSize(); ++i)
    {
        for (typename Eigen::SparseMatrix<T, Eigen::RowMajor>::InnerIterator iter(
                 A, i);
             iter;
             ++iter)
        {
            int rid = iter.row() / block_size, cid = iter.col() / block_size;
            int bid = rid * colsb_ + cid;
            if (coos.count(bid) == 0)
            {
                coos.insert({ bid, ele_type::Zero() });
            }
            coos[bid](
                iter.row() - rid * block_size, iter.col() - cid * block_size) =
                iter.value();
        }
    }
    // coo to csr
    size_t num = 0, last_rid = -1;
    for (const auto& item : coos)
    {
        size_t rid = item.first / colsb_;
        size_t cid = item.first % colsb_;
        value_.emplace_back(item.second);
        index_.emplace_back(cid);
        if (rid != last_rid)
        {
            offset_.emplace_back(num);
            last_rid = rid;
        }
        num++;
    }
    offset_.emplace_back(num);
}

// operators override.
// explicit name VEC
// TODO: optimize for eigen
template <typename T, const size_t block_size>
Eigen::Matrix<T, -1, 1> BCSR<T, block_size>::operator*(const VEC& rhs) const
{
    // error_msg_ext_cond(
    //     rhs.rows() != cols_, "BCSR<%lu, %lu>. rhs<%lu>. dim does not match. ",
    //     rows_, cols_, rhs.rows());
    VEC res(rows_);
    res.setZero();

#pragma omp parallel for
    for (size_t i = 0; i < rowsb_; ++i)
    {
        for (size_t j = offset_[i]; j < offset_[i + 1]; ++j)
        {
            size_t k = index_[j];
            res.segment(i * block_size, block_size).noalias() +=
                value_[j] * rhs.segment(k * block_size, block_size);
        }
    }

    return std::move(res);
}

template <typename T, const size_t block_size>
BCSR<T, block_size>
BCSR<T, block_size>::operator*(const BCSR<T, block_size>& rhs) const
{
    // TODO
    return BCSR<T, block_size>();
}

// explicit name ele_type
template <typename T, const size_t block_size>
std::vector<
    Eigen::Matrix<T, block_size, block_size>,
    Eigen::aligned_allocator<Eigen::Matrix<T, block_size, block_size>>>
BCSR<T, block_size>::get_diagonal() const
{
    std::vector<ele_type, Eigen::aligned_allocator<ele_type>> block_diag;
    // very simple version.
    for (size_t i = 0; i < rowsb_; ++i)
    {
        size_t lid = offset_[i], rid = offset_[i + 1];
        while (true)
        {
            size_t mid = (lid + rid) / 2;
            size_t cid = index_[mid];
            if (cid == i)
            {
                block_diag.emplace_back(value_[mid]);
                break;
            }
            else if (cid < i)
            {
                lid = mid + 1;
            }
            else
            {
                rid = mid;
            }

            if (lid >= rid)
            {
                block_diag.emplace_back(ele_type::Zero());
                break;
            }
        }
    }
    return std::move(block_diag);
}

template <typename T>
VEC_MAT<MAT3<T>> get_block_diagonal(const SPM_R<T>& A)
{
    exit_if(A.rows() != A.cols(), "A should be sysmetric.");
    VEC_MAT<MAT3<T>> diag_A(A.rows() / 3, MAT3<T>::Zero());

    auto fill_one_dim = [&](const size_t offset) -> void {
#pragma omp parallel for
        for (size_t i = offset; i < A.outerSize(); i += 3)
        {
            const size_t vert_id     = i / 3;
            const size_t first_index = i - offset;
            MAT3<T>&     A_i         = diag_A[vert_id];
            for (typename SPM_R<T>::InnerIterator it(A, i); it; ++it)
            {
                if (it.index() >= first_index)
                {
                    size_t diff = it.index() - first_index;
                    for (size_t j = 0; j < diff; ++j)
                    {
                        A_i(offset, j) = 0.0;
                    }
                    A_i(offset, diff)       = it.value();
                    const size_t left       = 2 - diff;
                    bool         if_advance = true;
                    for (size_t j = 0; j < left; ++j)
                    {
                        if (if_advance)
                        {
                            ++it;
                            ++diff;
                        }
                        if (it && it.index() == first_index + diff)
                        {
                            A_i(offset, diff) = it.value();
                            if_advance        = true;
                        }
                        else
                        {
                            A_i(offset, diff) = 0.0;
                            if_advance        = false;
                        }
                    }
                    break;
                }
            }
        }
    };
    fill_one_dim(0);
    fill_one_dim(1);
    fill_one_dim(2);
    return diag_A;
}

}  // namespace PhysIKA

#endif
