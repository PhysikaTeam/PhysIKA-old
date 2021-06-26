#ifndef UNION_FIND_SET_H
#define UNION_FIND_SET_H

#include <vector>
#include <cassert>
#include <numeric>


class UnionFindSet
{
public:
  UnionFindSet() { n_=0;}
  UnionFindSet(size_t n) : n_(n)
    {
      parent_.resize(n_, n_+5);
      std::iota(parent_.begin(), parent_.end(), 0);
    }
  ~UnionFindSet() { }

  size_t find(size_t a) const
    {
      assert(a < n_);
      if (parent_[a] == a)
        return a;
      return find(parent_[a]);
      // return (parent_[a] == a) ? a : (parent_[a] = find(parent_[a]));
    }

  void set_union(size_t a, size_t b)
    {
      assert(a < n_ && b < n_);
      a = find(a); b = find(b);
      if (a != b) parent_[a] = b;
      //parent_[find(a)] = find(b);
    }

  // b is the parent of a
  void set_union_by_order(size_t a, size_t b)
    {
      assert(a<n_ && b<n_);
      a = find(a); b = find(b);
      if (a != b) parent_[a] = b;
    }

  bool is_connected(size_t a, size_t b) const
    {
      assert(a<n_ && b<n_);
      return (find(a)==find(b));
    }

  void reset(size_t n)
    {
      n_ = n; parent_.clear();
      parent_.resize(n_, n_+5);
      std::iota(parent_.begin(), parent_.end(), 0);
    }
  void add_element()
    {
      n_ += 1;
      parent_.push_back(parent_.size());
    }
  size_t num() const { return n_; }

  std::vector<size_t> get_group(size_t a) const
    {
      const size_t pa = find(a);
      std::vector<size_t> group;
      for (size_t i = 0; i < n_; ++i)
      {
        if (find(i) == pa)
          group.push_back(i);
      } 
      return group;
    }
  std::unordered_map<size_t, std::vector<size_t>> get_group() const
    {
      std::unordered_map<size_t, std::vector<size_t>> group;
      for (size_t i = 0; i < n_; ++i)
        group[find(i)].push_back(i);
      return group;
    }
private:
  size_t n_;
  std::vector<size_t> parent_;
};


#endif
