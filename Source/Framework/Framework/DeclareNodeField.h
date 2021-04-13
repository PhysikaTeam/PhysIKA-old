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

#define DEF_CURRENT_VAR(name, T, value, desc) \
private:									\
	VarField<T> current_##name = VarField<T>(T(value), std::string(#name), desc, FieldType::Current, this);			\
public:										\
	inline VarField<T>* current##name() {return &current_##name;}

#define DEF_NEXT_VAR(name, T, value, desc) \
private:									\
	VarField<T> next_##name = VarField<T>(T(value), std::string(#name), desc, FieldType::Next, this);			\
public:										\
	inline VarField<T>* next##name() {return &next_##name;}

#define DEF_EMPTY_CURRENT_VAR(name, T, desc) \
private:									\
	VarField<T> current_##name = VarField<T>(std::string(#name), desc, FieldType::Current, this);			\
public:										\
	inline VarField<T>* current##name() {return &current_##name;}

#define DEF_EMPTY_NEXT_VAR(name, T, desc) \
private:									\
	VarField<T> next_##name = VarField<T>(std::string(#name), desc, FieldType::Next, this);			\
public:									\
	inline VarField<T>* next##name() {return &next_##name;}



#define DEF_CURRENT_ARRAY(name, T, size, device, desc) \
private:									\
	ArrayField<T, device> current_##name = ArrayField<T, device>(size, std::string(#name), desc, FieldType::Current, this);	\
public:									\
	inline ArrayField<T, device>* current##name() {return &current_##name;}

#define DEF_NEXT_ARRAY(name, T, size, device, desc) \
private:									\
	ArrayField<T, device> next_##name = ArrayField<T, device>(size, std::string(#name), desc, FieldType::Next, this);	\
public:									\
	inline ArrayField<T, device>* next##name() {return &next_##name;}


#define DEF_EMPTY_CURRENT_ARRAY(name, T, device, desc) \
private:									\
	ArrayField<T, device> current_##name = ArrayField<T, device>(0, std::string(#name), desc, FieldType::Current, this);	\
public:									\
	inline ArrayField<T, device>* current##name() {return &current_##name;}

#define DEF_EMPTY_NEXT_ARRAY(name, T, device, desc) \
private:									\
	ArrayField<T, device> next_##name = ArrayField<T, device>(0, std::string(#name), desc, FieldType::Next, this);	\
public:									\
	inline ArrayField<T, device>* next##name() {return &next_##name;}




#define DEF_EMPTY_CURRENT_NEIGHBOR_LIST(name, T, desc)		\
private:									\
	NeighborField<T> current_##name = NeighborField<T>(std::string(#name), desc, FieldType::Current, this, 0, 0);	\
public:									\
	inline NeighborField<T>* current##name() {return &current_##name;}

#define DEF_EMPTY_NEXT_NEIGHBOR_LIST(name, T, desc)		\
private:									\
	NeighborField<T> next_##name = NeighborField<T>(std::string(#name), desc, FieldType::Next, this, 0, 0);	\
public:									\
	inline NeighborField<T>* next##name() {return &next_##name;}

#define DEF_CURRENT_NEIGHBOR_LIST(name, T, size, nbr, desc)		\
private:									\
	NeighborField<T> current_##name = NeighborField<T>(std::string(#name), desc, FieldType::Current, this, size, nbr);	\
public:									\
	inline NeighborField<T>* in##name() {return &current_##name;}

#define DEF_NEXT_NEIGHBOR_LIST(name, T, size, nbr, desc)		\
private:									\
	NeighborField<T> next_##name = NeighborField<T>(std::string(#name), desc, FieldType::Next, this, size, nbr);	\
public:									\
	inline NeighborField<T>* next##name() {return &next_##name;}


#define DEF_NODE_PORT(name, T, desc)				\
private:									\
	SingleNodePort<T> single_##name = SingleNodePort<T>(std::string(#name), desc, this);					\
public:									\
	inline SingleNodePort<T>* get##name() {	return &single_##name; }

#define DEF_NODE_PORTS(name, T, desc)				\
private:									\
	MultipleNodePort<T> multiple_##name = MultipleNodePort<T>(std::string(#name)+std::string("(s)"), desc, this);					\
public:									\
	inline MultipleNodePort<T>* inport##name##s() { return &multiple_##name; }			\
	inline std::vector<std::shared_ptr<T>>& get##name##s(){return multiple_##name.getDerivedNodes();}				\
												\
	bool add##name(std::shared_ptr<T> c){		\
		multiple_##name.addDerivedNode(c);		\
		c->setParent(this);						\
		return true;							\
	}											\
												\
	bool remove##name(std::shared_ptr<T> c) {	\
		multiple_##name.removeDerivedNode(c);		\
		return true;							\
	}
}
