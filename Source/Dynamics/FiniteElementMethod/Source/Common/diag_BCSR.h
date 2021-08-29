/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: diagnoal BSCR implementation.
 * @version    : 1.0
 */
#ifndef PhysIKA_DIAG_BCSR_H
#define PhysIKA_DIAG_BCSR_H
#include <iostream>
#include "BCSR.h"
namespace PhysIKA {
template <typename T, const size_t block_size>
class diag_BCSR : public BCSR<T, block_size>
{
public:
    using ele_type = typename BCSR<T, block_size>::ele_type;
    using VEC      = typename BCSR<T, block_size>::VEC;
    diag_BCSR()
        : BCSR<T, block_size>() {}
    diag_BCSR(int rowsb, int colsb)
        : BCSR<T, block_size>(rowsb, colsb) {}
    diag_BCSR(const diag_BCSR<T, block_size>& other)
        : BCSR<T, block_size>(static_cast<BCSR<T, block_size>>(other)) {}

    ele_type operator()(const size_t& i) const
    {
        if (i < 0 || i >= this->rowsb_)
        {
            std::cout << "access out of data.\n";
            exit(EXIT_FAILURE);
        }
        return this->value_[i];
    }

    void setFromDiagMat(const std::vector<ele_type, Eigen::aligned_allocator<ele_type>>& diag_eles)
    {
        this->rowsb_ = this->colsb_ = diag_eles.size();
        this->rows_ = this->cols_ = this->rowsb_ * block_size;
        this->size_               = this->rows_ * this->cols_;

        this->value_.resize(this->rowsb_);
        this->index_.resize(this->rowsb_);
        this->offset_.resize(this->rowsb_ + 1);
#pragma omp parallel for
        for (size_t i = 0; i < this->rowsb_; ++i)
        {
            this->index_[i]  = i;
            this->offset_[i] = i;
            this->value_[i]  = diag_eles[i];
        }
        this->offset_[this->rowsb_] = this->rowsb_;
        return;
    }
    BCSR<T, block_size> operator*(const BCSR<T, block_size>& rhs) const
    {
        error_msg_ext_cond(
            rhs.rows() == this->rowsb_ && rhs.colsb() == this->colsb_,
            "The colsb() of lhs is not equal to rowsb() of rhs.");
        BCSR<T, block_size> res = rhs;
        for (size_t i = 0; i < this->rowsb_; ++i)
        {
            res.value_[i] = this->value_[i] * rhs.value_[i];
        }
        return res;
    }

    diag_BCSR<T, block_size> transpose() const
    {
        diag_BCSR<T, block_size> res(*this);
#pragma omp parallel for
        for (size_t i = 0; i < this->rowsb_; ++i)
            res.value_[i].transposeInPlace();
        return res;
    }

    VEC operator*(const VEC& rhs) const
    {
        return BCSR<T, block_size>::operator*(rhs);
    }
};

}  // namespace PhysIKA
#endif
