#pragma once
//----------------------------------------------------------------------------
// Check for unsupported old compilers.
#if defined(_MSC_VER) && _MSC_VER < 1800
# error PhysIKA requires MSVC++ 12.0 aka Visual Studio 2013 or newer
#endif

#if !defined(__clang__) && defined(__GNUC__) && (__GNUC__ < 4 || (__GNUC__ == 4 && __GNUC_MINOR__ < 8))
# error PhysIKA requires GCC 4.8 or newer
#endif

namespace PhysIKA
{

#define DEF_VAR(name, T, value, desc) \
private:									\
	VarField<T> var_##name = VarField<T>(T(value), std::string(#name), desc, FieldType::Param, this);			\
public:										\
	inline VarField<T>* var##name() {return &var_##name;}

#define DEF_IN_VAR(name, T, value, desc) \
private:									\
	VarField<T> in_##name = VarField<T>(T(value), std::string(#name), desc, FieldType::In, this);			\
public:										\
	inline VarField<T>* in##name() {return &in_##name;}

#define DEF_OUT_VAR(name, T, value, desc) \
private:									\
	VarField<T> out_##name = VarField<T>(T(value), std::string(#name), desc, FieldType::Out, this);			\
public:										\
	inline VarField<T>* out##name() {return &out_##name;}

#define DEF_EMPTY_VAR(name, T, desc) \
private:									\
	VarField<T> var_##name = VarField<T>(std::string(#name), desc, FieldType::Param, this);			\
public:										\
	inline VarField<T>* var##name() {return &var_##name;}

#define DEF_EMPTY_IN_VAR(name, T, desc) \
private:									\
	VarField<T> in_##name = VarField<T>(std::string(#name), desc, FieldType::In, this);			\
public:										\
	inline VarField<T>* in##name() {return &in_##name;}

#define DEF_EMPTY_OUT_VAR(name, T, desc) \
private:									\
	VarField<T> out_##name = VarField<T>(std::string(#name), desc, FieldType::Out, this);			\
public:									\
	inline VarField<T>* out##name() {return &out_##name;}


#define DEF_IN_ARRAY(name, T, size, device, desc) \
private:									\
	ArrayField<T, device> in_##name = ArrayField<T, device>(size, std::string(#name), desc, FieldType::In, this);	\
public:									\
	inline ArrayField<T, device>* in##name() {return &in_##name;}

#define DEF_OUT_ARRAY(name, T, size, device, desc) \
private:									\
	ArrayField<T, device> out_##name = ArrayField<T, device>(size, std::string(#name), desc, FieldType::Out, this);	\
public:									\
	inline ArrayField<T, device>* out##name() {return &out_##name;}


#define DEF_EMPTY_IN_ARRAY(name, T, device, desc) \
private:									\
	ArrayField<T, device> in_##name = ArrayField<T, device>(0, std::string(#name), desc, FieldType::In, this);	\
public:									\
	inline ArrayField<T, device>* in##name() {return &in_##name;}

#define DEF_EMPTY_OUT_ARRAY(name, T, device, desc) \
private:									\
	ArrayField<T, device> out_##name = ArrayField<T, device>(0, std::string(#name), desc, FieldType::Out, this);	\
public:									\
	inline ArrayField<T, device>* out##name() {return &out_##name;}




#define DEF_EMPTY_IN_NEIGHBOR_LIST(name, T, desc)		\
private:									\
	NeighborField<T> in_##name = NeighborField<T>(std::string(#name), desc, FieldType::In, this, 0, 0);	\
public:									\
	inline NeighborField<T>* in##name() {return &in_##name;}

#define DEF_EMPTY_OUT_NEIGHBOR_LIST(name, T, desc)		\
private:									\
	NeighborField<T> out_##name = NeighborField<T>(std::string(#name), desc, FieldType::Out, this, 0, 0);	\
public:									\
	inline NeighborField<T>* out##name() {return &out_##name;}

#define DEF_IN_NEIGHBOR_LIST(name, T, size, nbr, desc)		\
private:									\
	NeighborField<T> in_##name = NeighborField<T>(std::string(#name), desc, FieldType::In, this, size, nbr);	\
public:									\
	inline NeighborField<T>* in##name() {return &in_##name;}

#define DEF_OUT_NEIGHBOR_LIST(name, T, size, nbr, desc)		\
private:									\
	NeighborField<T> out_##name = NeighborField<T>(std::string(#name), desc, FieldType::Out, this, size, nbr);	\
public:									\
	inline NeighborField<T>* out##name() {return &out_##name;}
}