/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: copy helper for mass spring method.
 * @version    : 1.0
 */
#ifndef COPY_MACRO_JJ_H
#define COPY_MACRO_JJ_H

namespace PhysIKA {
#ifdef COMMA
#error
#endif

#ifdef LBRACKETS
#error
#endif

#ifdef RBRACKETS
#error
#endif

#ifdef THREE_ELEMENT_COPY
#error
#endif

#define COMMA ,
#define LBRACKETS (
#define RBRACKETS )

#define THREE_ELEMENT_COPY(L, LR_, R, RR_) \
    {                                      \
        L 0 LR_ = R 0 RR_;                 \
        L 1 LR_ = R 1 RR_;                 \
        L 2 LR_ = R 2 RR_;                 \
    }

#define FOUR_ELEMENT_COPY(L, LR_, R, RR_) \
    {                                     \
        L 0 LR_ = R 0 RR_;                \
        L 1 LR_ = R 1 RR_;                \
        L 2 LR_ = R 2 RR_;                \
        L 3 LR_ = R 3 RR_;                \
    }

}  // namespace PhysIKA

#endif  // COPY_MACRO_JJ_H
