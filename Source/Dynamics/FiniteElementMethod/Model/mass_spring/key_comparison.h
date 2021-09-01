/**
 * @author     : Zhao Chonyyao (cyzhao@zju.edu.cn)
 * @date       : 2021-04-30
 * @description: key comparison for mass spring method.
 * @version    : 1.0
 */
#ifndef KEY_COMPARISON_JJ_H
#define KEY_COMPARISON_JJ_H

//! \file comparison_func.h

#include <algorithm>

namespace PhysIKA {
//! \brief vertices comparison function used for map with tolerance which is controlled by coeff.
//! \param coeff is a coefficient to scale and cast vertices to grid vertices

//! \note coeff = long long int(1/e). for example, coeff(1e6) means 1e-6 tolerance and coeff(10) means 0.1 tolerance. The larger coeff is, the more precise comparison is.

template <typename V, long long int coeff>
struct KeyCompare
{
    bool operator()(const V& a, const V& b) const
    {
        std::array<long long int, 3> va = { static_cast<long long int>(coeff * a[0] + 0.5),
                                            static_cast<long long int>(coeff * a[1] + 0.5),
                                            static_cast<long long int>(coeff * a[2] + 0.5) };
        std::array<long long int, 3> vb = { static_cast<long long int>(coeff * b[0] + 0.5),
                                            static_cast<long long int>(coeff * b[1] + 0.5),
                                            static_cast<long long int>(coeff * b[2] + 0.5) };
        return va < vb;
    }
};

//! \brief exact vertices comparison function used for map without tolerance
template <typename V>
struct KeyCompare<V, 0>
{
    bool operator()(const V& a, const V& b) const
    {
        return std::tie(a[0], a[1], a[2]) < std::tie(b[0], b[1], b[2]);
    }
};

template <class T>
inline void hash_combine(std::size_t& seed, const T& v)
{
    std::hash<T> hasher;
    seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

//! \brief vertices equal comparison with tolerance
template <typename V, long long int coeff>
struct KeyEqual
{
    bool operator()(const V& a, const V& b) const
    {
        std::array<long long int, 3> va = { static_cast<long long int>(coeff * a[0] + 0.5),
                                            static_cast<long long int>(coeff * a[1] + 0.5),
                                            static_cast<long long int>(coeff * a[2] + 0.5) };
        std::array<long long int, 3> vb = { static_cast<long long int>(coeff * b[0] + 0.5),
                                            static_cast<long long int>(coeff * b[1] + 0.5),
                                            static_cast<long long int>(coeff * b[2] + 0.5) };
        return (va == vb);
    }
};

//! \brief vertices hash function with tolerance
template <typename V, long long int coeff>
struct KeyHash
{
    bool operator()(const V& key) const
    {
        std::array<long long int, 3> v = { static_cast<long long int>(coeff * key[0] + 0.5),
                                           static_cast<long long int>(coeff * key[1] + 0.5),
                                           static_cast<long long int>(coeff * key[2] + 0.5) };

        size_t value = 0;
        hash_combine(value, v[0]);
        hash_combine(value, v[1]);
        hash_combine(value, v[2]);
        return value;
    }
};

//! \brief exact vertices equal comparison without tolerance
template <typename V>
struct KeyEqual<V, 0>
{
    bool operator()(const V& a, const V& b) const
    {
        return (std::tie(a[0], a[1], a[2]) == std::tie(b[0], b[1], b[2]));
    }
};

//! \brief vertices hash function
template <typename V>
struct KeyHash<V, 0>
{
    bool operator()(const V& key) const
    {
        size_t value = 0;
        hash_combine(value, key[0]);
        hash_combine(value, key[1]);
        hash_combine(value, key[2]);
        return value;
    }
};

//! \note UnorderedKey method is designed for stl container. sort and operator[] must be supported
template <typename T>
struct UnorderedKeyHash
{
    size_t operator()(const T& key) const
    {
        T k = key;
        std::sort(k.begin(), k.end());

        size_t value = 0;
        size_t k_num = k.size();
        for (size_t i = 0; i < k_num; ++i)
            hash_combine(value, k[i]);
        return value;
    }
};

template <typename T>
struct UnorderedKeyEqual
{
    bool operator()(const T& lhs, const T& rhs) const
    {
        T lower = lhs;
        T upper = rhs;
        std::sort(lower.begin(), lower.end());
        std::sort(upper.begin(), upper.end());

        return (lower == upper);
    }
};

template <typename T>
struct UnorderedKeyCompare
{
    bool operator()(const T& lhs, const T& rhs) const
    {
        T lower = lhs;
        T upper = rhs;
        std::sort(lower.begin(), lower.end());
        std::sort(upper.begin(), upper.end());

        return (lhs < rhs);
    }
};

}  // namespace PhysIKA

#endif  // KEY_COMPARISON_JJ_H
