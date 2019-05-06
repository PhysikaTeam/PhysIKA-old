#ifndef PHYSIKA_SURFACE_FUILD_SURFACE_UTILITIES_BOUNDRECORDER_H_
#define PHYSIKA_SURFACE_FUILD_SURFACE_UTILITIES_BOUNDRECORDER_H_

namespace Physika{
template<class T>
class BoundRecorder {
public:
	BoundRecorder(): _n(0) { }
	virtual ~BoundRecorder() { }
	void insert(T const &t) {
		if (_n == 0) {
			_min = _max = t;
		} else {
			if (t < _min)
				_min = t;
			if (_max < t)
				_max = t;
		}
		_n++;
	}
	T const &get_min() const { return _min; }
	T const &get_max() const { return _max; }
	int get_n() const { return _n; }
private:
	int _n;
	T _min;
	T _max;
};
}
#endif
