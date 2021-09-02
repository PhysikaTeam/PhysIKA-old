/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: some helper function for error handling.
 * @version    : 1.0
 */
#pragma once

#include <iostream>
#include <cstdio>

namespace PhysIKA {

//! \brief exit if op return non zero;
#define EXITIF(OP, M)                                                               \
    {                                                                               \
        if (OP)                                                                     \
        {                                                                           \
            printf("[  \033[1;31merror\033[0m  ] error in %s, exit. %s\n", M, #OP); \
            exit(1);                                                                \
        }                                                                           \
    }

//! \brief return/exit if __VA_ARGS__ return non zero;
#define IFERROR(OP, ...)                                                                        \
    {                                                                                           \
        if (__VA_ARGS__)                                                                        \
        {                                                                                       \
            printf("[ \033[1;35mwarning\033[0m ] warning in call function %s\n", #__VA_ARGS__); \
            OP;                                                                                 \
        }                                                                                       \
    }

template <typename T>
void exit_if(const T& v, const char* msg = 0)
{
    if (v)
    {
        printf("%s", msg);
        exit(v);
    }
}

#define IF_ERR(end_op, ...)         \
    {                               \
        const int err(__VA_ARGS__); \
        if (err)                    \
            end_op(err);            \
    }
}  // namespace PhysIKA
