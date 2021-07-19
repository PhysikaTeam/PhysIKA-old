#pragma once

#include <map>
#include <memory>
#include <vector>

namespace pbal {
class Field
{
public:
    virtual ~Field() {}

protected:
    virtual size_t size()           = 0;
    virtual void   resize(size_t)   = 0;
    virtual void   permute(size_t*) = 0;
    friend class DataFrame;
};

template <typename T>
class DenseField : public Field
    , public std::vector<T>
{
public:
    using BaseType = std::vector<T>;

private:
    size_t size()
    {
        return BaseType::size();
    }
    void resize(size_t s)
    {
        BaseType::resize(s);
    }
    void permute(size_t* perm)
    {
        std::vector<T> copy(size());
        for (size_t i = 0; i < size(); i++)
            copy[i] = (*this)[perm[i]];
        (*( BaseType* )this) = std::move(copy);
    }
};

class DataFrame : std::map<std::string, std::unique_ptr<Field>>
{
    using Base = std::map<std::string, std::unique_ptr<Field>>;
    size_t _size;

    int idCount = 0;

public:
    DataFrame()
        : _size(0)
    {
        addField<DenseField<int>>("entity-id");
        resize(0);
    }
    size_t size()
    {
        return _size;
    }
    void resize(size_t s)
    {
        _size                = s;
        std::vector<int>& id = getField<DenseField<int>>("entity-id");
        while (_size > id.size())
            id.push_back(++idCount);
        for (auto it = begin(); it != end(); it++)
        {
            it->second->resize(_size);
        }
    }
    void permute(size_t* perm)
    {
        for (auto it = begin(); it != end(); it++)
        {
            it->second->permute(perm);
        }
    }
    template <typename T>
    void addDenseField(const std::string name)
    {
        addField<DenseField<T>>(name);
    }
    template <typename T>
    std::vector<T>& getDenseField(const std::string name)
    {
        return getField<DenseField<T>>(name);
    }
    using Base::begin;
    using Base::end;

private:
    template <typename T>
    void addField(const std::string& name)
    {
        if (find(name) == end())
        {
            std::unique_ptr<Field> ptr = std::make_unique<T>();
            ptr->resize(_size);
            (*this)[name] = std::move(ptr);
        }
    }
    template <typename T>
    typename T::BaseType& getField(const std::string& name)
    {
        if (find(name) == end())
            throw std::runtime_error("DataFrame: field \"" + name + "\" does not exist!");
        T* ptr = dynamic_cast<T*>((*this)[name].get());
        if (!ptr)
            throw std::runtime_error("DataFrame: field \"" + name + "\" type not match!");
        return *ptr;
    }
};
}  // namespace pbal