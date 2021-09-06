#pragma once

#include <vector>
#include <cassert>
#include <numeric>

/**
 * @brief FEM Geometry UnionFindSet
 * 
 */
class UnionFindSet
{
public:
    /**
     * @brief Construct a new Union Find Set object
     * 
     */
    UnionFindSet()
    {
        n_ = 0;
    }

    /**
     * @brief Construct a new Union Find Set object
     * 
     * @param n 
     */
    UnionFindSet(size_t n)
        : n_(n)
    {
        parent_.resize(n_, n_ + 5);
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    /**
     * @brief Destroy the Union Find Set object
     * 
     */
    ~UnionFindSet() {}

    /**
     * @brief Find and return the value
     * 
     * @param a 
     * @return size_t 
     */
    size_t find(size_t a) const
    {
        assert(a < n_);
        if (parent_[a] == a)
            return a;
        return find(parent_[a]);
        // return (parent_[a] == a) ? a : (parent_[a] = find(parent_[a]));
    }

    /**
     * @brief Set the union object
     * 
     * @param a 
     * @param b 
     */
    void set_union(size_t a, size_t b)
    {
        assert(a < n_ && b < n_);
        a = find(a);
        b = find(b);
        if (a != b)
            parent_[a] = b;
        //parent_[find(a)] = find(b);
    }

    // b is the parent of a
    /**
     * @brief Set the union by order object
     * 
     * @param a 
     * @param b 
     */
    void set_union_by_order(size_t a, size_t b)
    {
        assert(a < n_ && b < n_);
        a = find(a);
        b = find(b);
        if (a != b)
            parent_[a] = b;
    }

    /**
     * @brief Determine whether a and b connected
     * 
     * @param a 
     * @param b 
     * @return true 
     * @return false 
     */
    bool is_connected(size_t a, size_t b) const
    {
        assert(a < n_ && b < n_);
        return (find(a) == find(b));
    }

    /**
     * @brief Reset the object
     * 
     * @param n 
     */
    void reset(size_t n)
    {
        n_ = n;
        parent_.clear();
        parent_.resize(n_, n_ + 5);
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    /**
     * @brief Add the element
     * 
     */
    void add_element()
    {
        n_ += 1;
        parent_.push_back(parent_.size());
    }

    /**
     * @brief Get the number of the elements
     * 
     * @return size_t 
     */
    size_t num() const
    {
        return n_;
    }

    /**
     * @brief Get the group object
     * 
     * @param a 
     * @return std::vector<size_t> 
     */
    std::vector<size_t> get_group(size_t a) const
    {
        const size_t        pa = find(a);
        std::vector<size_t> group;
        for (size_t i = 0; i < n_; ++i)
        {
            if (find(i) == pa)
                group.push_back(i);
        }
        return group;
    }

    /**
     * @brief Get the group object
     * 
     * @return std::unordered_map<size_t, std::vector<size_t>> 
     */
    std::unordered_map<size_t, std::vector<size_t>> get_group() const
    {
        std::unordered_map<size_t, std::vector<size_t>> group;
        for (size_t i = 0; i < n_; ++i)
            group[find(i)].push_back(i);
        return group;
    }

private:
    size_t              n_;
    std::vector<size_t> parent_;
};
