#if defined(_MSC_VER) && (_MSC_VER >= 1800)
// eliminating duplicated round() declaration
#define HAVE_ROUND 1
#endif

#include <Python.h>

#if !PYTHON_USE_NUMPY
#error "The module can only be built if NumPy is available"
#endif

#define MODULESTR "cv2"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/ndarrayobject.h"

#include "opencv2/core/core.hpp"
#include "opencv2/contrib/contrib.hpp"
#include "opencv2/flann/miniflann.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/ml/ml.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/video/background_segm.hpp"
#include "opencv2/photo/photo.hpp"
#include "opencv2/highgui/highgui.hpp"

#include "opencv2/opencv_modules.hpp"

#ifdef HAVE_OPENCV_NONFREE
#  include "opencv2/nonfree/nonfree.hpp"
#endif

using cv::flann::IndexParams;
using cv::flann::SearchParams;

static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

struct ArgInfo
{
    const char * name;
    bool outputarg;
    // more fields may be added if necessary

    ArgInfo(const char * name_, bool outputarg_)
        : name(name_)
        , outputarg(outputarg_) {}

    // to match with older pyopencv_to function signature
    operator const char *() const { return name; }
};

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

using namespace cv;

typedef vector<uchar> vector_uchar;
typedef vector<int> vector_int;
typedef vector<float> vector_float;
typedef vector<double> vector_double;
typedef vector<Point> vector_Point;
typedef vector<Point2f> vector_Point2f;
typedef vector<Vec2f> vector_Vec2f;
typedef vector<Vec3f> vector_Vec3f;
typedef vector<Vec4f> vector_Vec4f;
typedef vector<Vec6f> vector_Vec6f;
typedef vector<Vec4i> vector_Vec4i;
typedef vector<Rect> vector_Rect;
typedef vector<KeyPoint> vector_KeyPoint;
typedef vector<Mat> vector_Mat;
typedef vector<DMatch> vector_DMatch;
typedef vector<string> vector_string;
typedef vector<vector<Point> > vector_vector_Point;
typedef vector<vector<Point2f> > vector_vector_Point2f;
typedef vector<vector<Point3f> > vector_vector_Point3f;
typedef vector<vector<DMatch> > vector_vector_DMatch;

typedef Ptr<Algorithm> Ptr_Algorithm;
typedef Ptr<FeatureDetector> Ptr_FeatureDetector;
typedef Ptr<DescriptorExtractor> Ptr_DescriptorExtractor;
typedef Ptr<Feature2D> Ptr_Feature2D;
typedef Ptr<DescriptorMatcher> Ptr_DescriptorMatcher;
typedef Ptr<CLAHE> Ptr_CLAHE;

typedef SimpleBlobDetector::Params SimpleBlobDetector_Params;

typedef cvflann::flann_distance_t cvflann_flann_distance_t;
typedef cvflann::flann_algorithm_t cvflann_flann_algorithm_t;
typedef Ptr<flann::IndexParams> Ptr_flann_IndexParams;
typedef Ptr<flann::SearchParams> Ptr_flann_SearchParams;

typedef Ptr<FaceRecognizer> Ptr_FaceRecognizer;
typedef vector<Scalar> vector_Scalar;

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

class NumpyAllocator : public MatAllocator
{
public:
    NumpyAllocator() {}
    ~NumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
            _sizes[i] = sizes[i];
        if( cn > 1 )
        {
            /*if( _sizes[dims-1] == 1 )
                _sizes[dims-1] = cn;
            else*/
                _sizes[dims++] = cn;
        }
        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);
        if(!o)
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        refcount = refcountFromPyObject(o);
        npy_intp* _strides = PyArray_STRIDES((PyArrayObject*) o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA((PyArrayObject*) o);
    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
};

NumpyAllocator g_numpyAllocator;

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

// special case, when the convertor needs full ArgInfo structure
static int pyopencv_to(const PyObject* o, Mat& m, const ArgInfo info, bool allowND=true)
{
    if(!o || o == Py_None)
    {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
        return true;
    }

    if( PyInt_Check(o) )
    {
        double v[] = {
            static_cast<double>(PyInt_AsLong((PyObject*)o)),
            0.,
            0.,
            0.,
        };
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyFloat_Check(o) )
    {
        double v[] = {PyFloat_AsDouble((PyObject*)o), 0., 0., 0.};
        m = Mat(4, 1, CV_64F, v).clone();
        return true;
    }
    if( PyTuple_Check(o) )
    {
        int i, sz = (int)PyTuple_Size((PyObject*)o);
        m = Mat(sz, 1, CV_64F);
        for( i = 0; i < sz; i++ )
        {
            PyObject* oi = PyTuple_GET_ITEM(o, i);
            if( PyInt_Check(oi) )
                m.at<double>(i) = (double)PyInt_AsLong(oi);
            else if( PyFloat_Check(oi) )
                m.at<double>(i) = (double)PyFloat_AsDouble(oi);
            else
            {
                failmsg("%s is not a numerical tuple", info.name);
                m.release();
                return false;
            }
        }
        return true;
    }

    if( !PyArray_Check(o) )
    {
        failmsg("%s is not a numpy array, neither a scalar", info.name);
        return false;
    }

    PyArrayObject* oarr = (PyArrayObject*) o;

    bool needcopy = false, needcast = false;
    int typenum = PyArray_TYPE(oarr), new_typenum = typenum;
    int type = typenum == NPY_UBYTE ? CV_8U :
               typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U :
               typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT ? CV_32S :
               typenum == NPY_INT32 ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 )
    {
        if( typenum == NPY_INT64 || typenum == NPY_UINT64 || type == NPY_LONG )
        {
            needcopy = needcast = true;
            new_typenum = NPY_INT;
            type = CV_32S;
        }
        else
        {
            failmsg("%s data type = %d is not supported", info.name, typenum);
            return false;
        }
    }

    int ndims = PyArray_NDIM(oarr);
    if(ndims >= CV_MAX_DIM)
    {
        failmsg("%s dimensionality (=%d) is too high", info.name, ndims);
        return false;
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(oarr);
    const npy_intp* _strides = PyArray_STRIDES(oarr);
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;

    for( int i = ndims-1; i >= 0 && !needcopy; i-- )
    {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        if( (i == ndims-1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }

    if( ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2] )
        needcopy = true;

    if (needcopy)
    {
        if (info.outputarg)
        {
            failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
            return false;
        }

        if( needcast ) {
            o = PyArray_Cast(oarr, new_typenum);
            oarr = (PyArrayObject*) o;
        }
        else {
            oarr = PyArray_GETCONTIGUOUS(oarr);
            o = (PyObject*) oarr;
        }

        _strides = PyArray_STRIDES(oarr);
    }

    for(int i = 0; i < ndims; i++)
    {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }

    // handle degenerate case
    if( ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }

    if( ismultichannel )
    {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }

    if( ndims > 2 && !allowND )
    {
        failmsg("%s has more than 2 dimensions", info.name);
        return false;
    }

    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);

    if( m.data )
    {
        m.refcount = refcountFromPyObject(o);
        if (!needcopy)
        {
            m.addref(); // protect the original numpy array from deallocation
                        // (since Mat destructor will decrement the reference counter)
        }
    };
    m.allocator = &g_numpyAllocator;

    return true;
}

static PyObject* pyopencv_from(const Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        ERRWRAP2(m.copyTo(temp));
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
}

static bool pyopencv_to(PyObject *o, Scalar& s, const char *name = "<unknown>")
{
    if(!o || o == Py_None)
        return true;
    if (PySequence_Check(o)) {
        PyObject *fi = PySequence_Fast(o, name);
        if (fi == NULL)
            return false;
        if (4 < PySequence_Fast_GET_SIZE(fi))
        {
            failmsg("Scalar value for argument '%s' is longer than 4", name);
            return false;
        }
        for (Py_ssize_t i = 0; i < PySequence_Fast_GET_SIZE(fi); i++) {
            PyObject *item = PySequence_Fast_GET_ITEM(fi, i);
            if (PyFloat_Check(item) || PyInt_Check(item)) {
                s[(int)i] = PyFloat_AsDouble(item);
            } else {
                failmsg("Scalar value for argument '%s' is not numeric", name);
                return false;
            }
        }
        Py_DECREF(fi);
    } else {
        if (PyFloat_Check(o) || PyInt_Check(o)) {
            s[0] = PyFloat_AsDouble(o);
        } else {
            failmsg("Scalar value for argument '%s' is not numeric", name);
            return false;
        }
    }
    return true;
}

static inline PyObject* pyopencv_from(const Scalar& src)
{
    return Py_BuildValue("(dddd)", src[0], src[1], src[2], src[3]);
}

static PyObject* pyopencv_from(bool value)
{
    return PyBool_FromLong(value);
}

static bool pyopencv_to(PyObject* obj, bool& value, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    int _val = PyObject_IsTrue(obj);
    if(_val < 0)
        return false;
    value = _val > 0;
    return true;
}

static PyObject* pyopencv_from(size_t value)
{
    return PyLong_FromUnsignedLong((unsigned long)value);
}

static bool pyopencv_to(PyObject* obj, size_t& value, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    value = (int)PyLong_AsUnsignedLong(obj);
    return value != (size_t)-1 || !PyErr_Occurred();
}

static PyObject* pyopencv_from(int value)
{
    return PyInt_FromLong(value);
}

static PyObject* pyopencv_from(cvflann_flann_algorithm_t value)
{
    return PyInt_FromLong(int(value));
}

static PyObject* pyopencv_from(cvflann_flann_distance_t value)
{
    return PyInt_FromLong(int(value));
}

static bool pyopencv_to(PyObject* obj, int& value, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(PyInt_Check(obj))
        value = (int)PyInt_AsLong(obj);
    else if(PyLong_Check(obj))
        value = (int)PyLong_AsLong(obj);
    else
        return false;
    return value != -1 || !PyErr_Occurred();
}

static PyObject* pyopencv_from(uchar value)
{
    return PyInt_FromLong(value);
}

static bool pyopencv_to(PyObject* obj, uchar& value, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    int ivalue = (int)PyInt_AsLong(obj);
    value = cv::saturate_cast<uchar>(ivalue);
    return ivalue != -1 || !PyErr_Occurred();
}

static PyObject* pyopencv_from(double value)
{
    return PyFloat_FromDouble(value);
}

static bool pyopencv_to(PyObject* obj, double& value, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyInt_CheckExact(obj))
        value = (double)PyInt_AS_LONG(obj);
    else
        value = PyFloat_AsDouble(obj);
    return !PyErr_Occurred();
}

static PyObject* pyopencv_from(float value)
{
    return PyFloat_FromDouble(value);
}

static bool pyopencv_to(PyObject* obj, float& value, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyInt_CheckExact(obj))
        value = (float)PyInt_AS_LONG(obj);
    else
        value = (float)PyFloat_AsDouble(obj);
    return !PyErr_Occurred();
}

static PyObject* pyopencv_from(int64 value)
{
    return PyFloat_FromDouble((double)value);
}

static PyObject* pyopencv_from(const string& value)
{
    return PyString_FromString(value.empty() ? "" : value.c_str());
}

static bool pyopencv_to(PyObject* obj, string& value, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    char* str = PyString_AsString(obj);
    if(!str)
        return false;
    value = string(str);
    return true;
}

static inline bool pyopencv_to(PyObject* obj, Size& sz, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "ii", &sz.width, &sz.height) > 0;
}

static inline PyObject* pyopencv_from(const Size& sz)
{
    return Py_BuildValue("(ii)", sz.width, sz.height);
}

static inline bool pyopencv_to(PyObject* obj, Rect& r, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    return PyArg_ParseTuple(obj, "iiii", &r.x, &r.y, &r.width, &r.height) > 0;
}

static inline PyObject* pyopencv_from(const Rect& r)
{
    return Py_BuildValue("(iiii)", r.x, r.y, r.width, r.height);
}

static inline bool pyopencv_to(PyObject* obj, Range& r, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(PyObject_Size(obj) == 0)
    {
        r = Range::all();
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &r.start, &r.end) > 0;
}

static inline PyObject* pyopencv_from(const Range& r)
{
    return Py_BuildValue("(ii)", r.start, r.end);
}

static inline bool pyopencv_to(PyObject* obj, CvSlice& r, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(PyObject_Size(obj) == 0)
    {
        r = CV_WHOLE_SEQ;
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &r.start_index, &r.end_index) > 0;
}

static inline PyObject* pyopencv_from(const CvSlice& r)
{
    return Py_BuildValue("(ii)", r.start_index, r.end_index);
}

static inline bool pyopencv_to(PyObject* obj, Point& p, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyComplex_CheckExact(obj))
    {
        Py_complex c = PyComplex_AsCComplex(obj);
        p.x = saturate_cast<int>(c.real);
        p.y = saturate_cast<int>(c.imag);
        return true;
    }
    return PyArg_ParseTuple(obj, "ii", &p.x, &p.y) > 0;
}

static inline bool pyopencv_to(PyObject* obj, Point2f& p, const char* name = "<unknown>")
{
    (void)name;
    if(!obj || obj == Py_None)
        return true;
    if(!!PyComplex_CheckExact(obj))
    {
        Py_complex c = PyComplex_AsCComplex(obj);
        p.x = saturate_cast<float>(c.real);
        p.y = saturate_cast<float>(c.imag);
        return true;
    }
    return PyArg_ParseTuple(obj, "ff", &p.x, &p.y) > 0;
}

static inline PyObject* pyopencv_from(const Point& p)
{
    return Py_BuildValue("(ii)", p.x, p.y);
}

static inline PyObject* pyopencv_from(const Point2f& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

static inline bool pyopencv_to(PyObject* obj, Vec3d& v, const char* name = "<unknown>")
{
    (void)name;
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "ddd", &v[0], &v[1], &v[2]) > 0;
}

static inline PyObject* pyopencv_from(const Vec3d& v)
{
    return Py_BuildValue("(ddd)", v[0], v[1], v[2]);
}

static inline PyObject* pyopencv_from(const Point2d& p)
{
    return Py_BuildValue("(dd)", p.x, p.y);
}

template<typename _Tp> struct pyopencvVecConverter
{
    static bool to(PyObject* obj, vector<_Tp>& value, const ArgInfo info)
    {
        typedef typename DataType<_Tp>::channel_type _Cp;
        if(!obj || obj == Py_None)
            return true;
        if (PyArray_Check(obj))
        {
            Mat m;
            pyopencv_to(obj, m, info);
            m.copyTo(value);
        }
        if (!PySequence_Check(obj))
            return false;
        PyObject *seq = PySequence_Fast(obj, info.name);
        if (seq == NULL)
            return false;
        int i, j, n = (int)PySequence_Fast_GET_SIZE(seq);
        value.resize(n);

        int type = DataType<_Tp>::type;
        int depth = CV_MAT_DEPTH(type), channels = CV_MAT_CN(type);
        PyObject** items = PySequence_Fast_ITEMS(seq);

        for( i = 0; i < n; i++ )
        {
            PyObject* item = items[i];
            PyObject* seq_i = 0;
            PyObject** items_i = &item;
            _Cp* data = (_Cp*)&value[i];

            if( channels == 2 && PyComplex_CheckExact(item) )
            {
                Py_complex c = PyComplex_AsCComplex(obj);
                data[0] = saturate_cast<_Cp>(c.real);
                data[1] = saturate_cast<_Cp>(c.imag);
                continue;
            }
            if( channels > 1 )
            {
                if( PyArray_Check(item))
                {
                    Mat src;
                    pyopencv_to(item, src, info);
                    if( src.dims != 2 || src.channels() != 1 ||
                       ((src.cols != 1 || src.rows != channels) &&
                        (src.cols != channels || src.rows != 1)))
                        break;
                    Mat dst(src.rows, src.cols, depth, data);
                    src.convertTo(dst, type);
                    if( dst.data != (uchar*)data )
                        break;
                    continue;
                }

                seq_i = PySequence_Fast(item, info.name);
                if( !seq_i || (int)PySequence_Fast_GET_SIZE(seq_i) != channels )
                {
                    Py_XDECREF(seq_i);
                    break;
                }
                items_i = PySequence_Fast_ITEMS(seq_i);
            }

            for( j = 0; j < channels; j++ )
            {
                PyObject* item_ij = items_i[j];
                if( PyInt_Check(item_ij))
                {
                    int v = (int)PyInt_AsLong(item_ij);
                    if( v == -1 && PyErr_Occurred() )
                        break;
                    data[j] = saturate_cast<_Cp>(v);
                }
                else if( PyLong_Check(item_ij))
                {
                    int v = (int)PyLong_AsLong(item_ij);
                    if( v == -1 && PyErr_Occurred() )
                        break;
                    data[j] = saturate_cast<_Cp>(v);
                }
                else if( PyFloat_Check(item_ij))
                {
                    double v = PyFloat_AsDouble(item_ij);
                    if( PyErr_Occurred() )
                        break;
                    data[j] = saturate_cast<_Cp>(v);
                }
                else
                    break;
            }
            Py_XDECREF(seq_i);
            if( j < channels )
                break;
        }
        Py_DECREF(seq);
        return i == n;
    }

    static PyObject* from(const vector<_Tp>& value)
    {
        if(value.empty())
            return PyTuple_New(0);
        Mat src((int)value.size(), DataType<_Tp>::channels, DataType<_Tp>::depth, (uchar*)&value[0]);
        return pyopencv_from(src);
    }
};


template<typename _Tp> static inline bool pyopencv_to(PyObject* obj, vector<_Tp>& value, const ArgInfo info)
{
    return pyopencvVecConverter<_Tp>::to(obj, value, info);
}

template<typename _Tp> static inline PyObject* pyopencv_from(const vector<_Tp>& value)
{
    return pyopencvVecConverter<_Tp>::from(value);
}

static PyObject* pyopencv_from(const KeyPoint&);
static PyObject* pyopencv_from(const DMatch&);

template<typename _Tp> static inline bool pyopencv_to_generic_vec(PyObject* obj, vector<_Tp>& value, const ArgInfo info)
{
    if(!obj || obj == Py_None)
       return true;
    if (!PySequence_Check(obj))
        return false;
    PyObject *seq = PySequence_Fast(obj, info.name);
    if (seq == NULL)
        return false;
    int i, n = (int)PySequence_Fast_GET_SIZE(seq);
    value.resize(n);

    PyObject** items = PySequence_Fast_ITEMS(seq);

    for( i = 0; i < n; i++ )
    {
        PyObject* item = items[i];
        if(!pyopencv_to(item, value[i], info))
            break;
    }
    Py_DECREF(seq);
    return i == n;
}

template<typename _Tp> static inline PyObject* pyopencv_from_generic_vec(const vector<_Tp>& value)
{
    int i, n = (int)value.size();
    PyObject* seq = PyList_New(n);
    for( i = 0; i < n; i++ )
    {
        PyObject* item = pyopencv_from(value[i]);
        if(!item)
            break;
        PyList_SET_ITEM(seq, i, item);
    }
    if( i < n )
    {
        Py_DECREF(seq);
        return 0;
    }
    return seq;
}


template<typename _Tp> struct pyopencvVecConverter<vector<_Tp> >
{
    static bool to(PyObject* obj, vector<vector<_Tp> >& value, const char* name="<unknown>")
    {
        return pyopencv_to_generic_vec(obj, value, name);
    }

    static PyObject* from(const vector<vector<_Tp> >& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<Mat>
{
    static bool to(PyObject* obj, vector<Mat>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const vector<Mat>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<KeyPoint>
{
    static bool to(PyObject* obj, vector<KeyPoint>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const vector<KeyPoint>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<DMatch>
{
    static bool to(PyObject* obj, vector<DMatch>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const vector<DMatch>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};

template<> struct pyopencvVecConverter<string>
{
    static bool to(PyObject* obj, vector<string>& value, const ArgInfo info)
    {
        return pyopencv_to_generic_vec(obj, value, info);
    }

    static PyObject* from(const vector<string>& value)
    {
        return pyopencv_from_generic_vec(value);
    }
};


static inline bool pyopencv_to(PyObject *obj, CvTermCriteria& dst, const char *name="<unknown>")
{
    (void)name;
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "iid", &dst.type, &dst.max_iter, &dst.epsilon) > 0;
}

static inline PyObject* pyopencv_from(const CvTermCriteria& src)
{
    return Py_BuildValue("(iid)", src.type, src.max_iter, src.epsilon);
}

static inline bool pyopencv_to(PyObject *obj, TermCriteria& dst, const char *name="<unknown>")
{
    (void)name;
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "iid", &dst.type, &dst.maxCount, &dst.epsilon) > 0;
}

static inline PyObject* pyopencv_from(const TermCriteria& src)
{
    return Py_BuildValue("(iid)", src.type, src.maxCount, src.epsilon);
}

static inline bool pyopencv_to(PyObject *obj, RotatedRect& dst, const char *name="<unknown>")
{
    (void)name;
    if(!obj)
        return true;
    return PyArg_ParseTuple(obj, "(ff)(ff)f", &dst.center.x, &dst.center.y, &dst.size.width, &dst.size.height, &dst.angle) > 0;
}

static inline PyObject* pyopencv_from(const RotatedRect& src)
{
    return Py_BuildValue("((ff)(ff)f)", src.center.x, src.center.y, src.size.width, src.size.height, src.angle);
}

static inline PyObject* pyopencv_from(const Moments& m)
{
    return Py_BuildValue("{s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d,s:d}",
                         "m00", m.m00, "m10", m.m10, "m01", m.m01,
                         "m20", m.m20, "m11", m.m11, "m02", m.m02,
                         "m30", m.m30, "m21", m.m21, "m12", m.m12, "m03", m.m03,
                         "mu20", m.mu20, "mu11", m.mu11, "mu02", m.mu02,
                         "mu30", m.mu30, "mu21", m.mu21, "mu12", m.mu12, "mu03", m.mu03,
                         "nu20", m.nu20, "nu11", m.nu11, "nu02", m.nu02,
                         "nu30", m.nu30, "nu21", m.nu21, "nu12", m.nu12, "nu03", m.nu03);
}

static inline PyObject* pyopencv_from(const CvDTreeNode* node)
{
    double value = node->value;
    int ivalue = cvRound(value);
    return value == ivalue ? PyInt_FromLong(ivalue) : PyFloat_FromDouble(value);
}

static bool pyopencv_to(PyObject *o, cv::flann::IndexParams& p, const char *name="<unknown>")
{
    (void)name;
    bool ok = false;
    PyObject* keys = PyObject_CallMethod(o,(char*)"keys",0);
    PyObject* values = PyObject_CallMethod(o,(char*)"values",0);

    if( keys && values )
    {
        int i, n = (int)PyList_GET_SIZE(keys);
        for( i = 0; i < n; i++ )
        {
            PyObject* key = PyList_GET_ITEM(keys, i);
            PyObject* item = PyList_GET_ITEM(values, i);
            if( !PyString_Check(key) )
                break;
            std::string k = PyString_AsString(key);
            if( PyString_Check(item) )
            {
                const char* value = PyString_AsString(item);
                p.setString(k, value);
            }
            else if( !!PyBool_Check(item) )
                p.setBool(k, item == Py_True);
            else if( PyInt_Check(item) )
            {
                int value = (int)PyInt_AsLong(item);
                if( strcmp(k.c_str(), "algorithm") == 0 )
                    p.setAlgorithm(value);
                else
                    p.setInt(k, value);
            }
            else if( PyFloat_Check(item) )
            {
                double value = PyFloat_AsDouble(item);
                p.setDouble(k, value);
            }
            else
                break;
        }
        ok = i == n && !PyErr_Occurred();
    }

    Py_XDECREF(keys);
    Py_XDECREF(values);
    return ok;
}

template <class T>
static bool pyopencv_to(PyObject *o, Ptr<T>& p, const char *name="<unknown>")
{
    p = new T();
    return pyopencv_to(o, *p, name);
}


static bool pyopencv_to(PyObject *o, cvflann::flann_distance_t& dist, const char *name="<unknown>")
{
    int d = (int)dist;
    bool ok = pyopencv_to(o, d, name);
    dist = (cvflann::flann_distance_t)d;
    return ok;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static void OnMouse(int event, int x, int y, int flags, void* param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("iiiiO", event, x, y, flags, PyTuple_GetItem(o, 1));

    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    else
        Py_DECREF(r);
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

static PyObject *pycvSetMouseCallback(PyObject*, PyObject *args, PyObject *kw)
{
    const char *keywords[] = { "window_name", "on_mouse", "param", NULL };
    char* name;
    PyObject *on_mouse;
    PyObject *param = NULL;

    if (!PyArg_ParseTupleAndKeywords(args, kw, "sO|O", (char**)keywords, &name, &on_mouse, &param))
        return NULL;
    if (!PyCallable_Check(on_mouse)) {
        PyErr_SetString(PyExc_TypeError, "on_mouse must be callable");
        return NULL;
    }
    if (param == NULL) {
        param = Py_None;
    }
    ERRWRAP2(cvSetMouseCallback(name, OnMouse, Py_BuildValue("OO", on_mouse, param)));
    Py_RETURN_NONE;
}

static void OnChange(int pos, void *param)
{
    PyGILState_STATE gstate;
    gstate = PyGILState_Ensure();

    PyObject *o = (PyObject*)param;
    PyObject *args = Py_BuildValue("(i)", pos);
    PyObject *r = PyObject_Call(PyTuple_GetItem(o, 0), args, NULL);
    if (r == NULL)
        PyErr_Print();
    Py_DECREF(args);
    PyGILState_Release(gstate);
}

static PyObject *pycvCreateTrackbar(PyObject*, PyObject *args)
{
    PyObject *on_change;
    char* trackbar_name;
    char* window_name;
    int *value = new int;
    int count;

    if (!PyArg_ParseTuple(args, "ssiiO", &trackbar_name, &window_name, value, &count, &on_change))
        return NULL;
    if (!PyCallable_Check(on_change)) {
        PyErr_SetString(PyExc_TypeError, "on_change must be callable");
        return NULL;
    }
    ERRWRAP2(cvCreateTrackbar2(trackbar_name, window_name, value, count, OnChange, Py_BuildValue("OO", on_change, Py_None)));
    Py_RETURN_NONE;
}

///////////////////////////////////////////////////////////////////////////////////////

#define MKTYPE2(NAME) pyopencv_##NAME##_specials(); if (!to_ok(&pyopencv_##NAME##_Type)) return

#ifdef __GNUC__
#  pragma GCC diagnostic ignored "-Wunused-parameter"
#  pragma GCC diagnostic ignored "-Wmissing-field-initializers"
#endif

#include "pyopencv_generated_types.h"
#include "pyopencv_generated_funcs.h"

static PyMethodDef methods[] = {

#include "pyopencv_generated_func_tab.h"
  {"createTrackbar", pycvCreateTrackbar, METH_VARARGS, "createTrackbar(trackbarName, windowName, value, count, onChange) -> None"},
  {"setMouseCallback", (PyCFunction)pycvSetMouseCallback, METH_KEYWORDS, "setMouseCallback(windowName, onMouse [, param]) -> None"},
  {NULL, NULL},
};

/************************************************************************/
/* Module init */

static int to_ok(PyTypeObject *to)
{
  to->tp_alloc = PyType_GenericAlloc;
  to->tp_new = PyType_GenericNew;
  to->tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
  return (PyType_Ready(to) == 0);
}

#include "cv2.cv.hpp"

extern "C"
#if defined WIN32 || defined _WIN32
__declspec(dllexport)
#endif
void initcv2();

void initcv2()
{
#if PYTHON_USE_NUMPY
    import_array();
#endif

#if PYTHON_USE_NUMPY
#include "pyopencv_generated_type_reg.h"
#endif

  PyObject* m = Py_InitModule(MODULESTR, methods);
  PyObject* d = PyModule_GetDict(m);

  PyDict_SetItemString(d, "__version__", PyString_FromString(CV_VERSION));

  opencv_error = PyErr_NewException((char*)MODULESTR".error", NULL, NULL);
  PyDict_SetItemString(d, "error", opencv_error);

  PyObject* cv_m = init_cv();

  PyDict_SetItemString(d, "cv", cv_m);

#define PUBLISH(I) PyDict_SetItemString(d, #I, PyInt_FromLong(I))
#define PUBLISHU(I) PyDict_SetItemString(d, #I, PyLong_FromUnsignedLong(I))
#define PUBLISH2(I, value) PyDict_SetItemString(d, #I, PyLong_FromLong(value))

  PUBLISHU(IPL_DEPTH_8U);
  PUBLISHU(IPL_DEPTH_8S);
  PUBLISHU(IPL_DEPTH_16U);
  PUBLISHU(IPL_DEPTH_16S);
  PUBLISHU(IPL_DEPTH_32S);
  PUBLISHU(IPL_DEPTH_32F);
  PUBLISHU(IPL_DEPTH_64F);

  PUBLISH(CV_LOAD_IMAGE_COLOR);
  PUBLISH(CV_LOAD_IMAGE_GRAYSCALE);
  PUBLISH(CV_LOAD_IMAGE_UNCHANGED);
  PUBLISH(CV_HIST_ARRAY);
  PUBLISH(CV_HIST_SPARSE);
  PUBLISH(CV_8U);
  PUBLISH(CV_8UC1);
  PUBLISH(CV_8UC2);
  PUBLISH(CV_8UC3);
  PUBLISH(CV_8UC4);
  PUBLISH(CV_8S);
  PUBLISH(CV_8SC1);
  PUBLISH(CV_8SC2);
  PUBLISH(CV_8SC3);
  PUBLISH(CV_8SC4);
  PUBLISH(CV_16U);
  PUBLISH(CV_16UC1);
  PUBLISH(CV_16UC2);
  PUBLISH(CV_16UC3);
  PUBLISH(CV_16UC4);
  PUBLISH(CV_16S);
  PUBLISH(CV_16SC1);
  PUBLISH(CV_16SC2);
  PUBLISH(CV_16SC3);
  PUBLISH(CV_16SC4);
  PUBLISH(CV_32S);
  PUBLISH(CV_32SC1);
  PUBLISH(CV_32SC2);
  PUBLISH(CV_32SC3);
  PUBLISH(CV_32SC4);
  PUBLISH(CV_32F);
  PUBLISH(CV_32FC1);
  PUBLISH(CV_32FC2);
  PUBLISH(CV_32FC3);
  PUBLISH(CV_32FC4);
  PUBLISH(CV_64F);
  PUBLISH(CV_64FC1);
  PUBLISH(CV_64FC2);
  PUBLISH(CV_64FC3);
  PUBLISH(CV_64FC4);
  PUBLISH(CV_NEXT_AROUND_ORG);
  PUBLISH(CV_NEXT_AROUND_DST);
  PUBLISH(CV_PREV_AROUND_ORG);
  PUBLISH(CV_PREV_AROUND_DST);
  PUBLISH(CV_NEXT_AROUND_LEFT);
  PUBLISH(CV_NEXT_AROUND_RIGHT);
  PUBLISH(CV_PREV_AROUND_LEFT);
  PUBLISH(CV_PREV_AROUND_RIGHT);

  PUBLISH(CV_WINDOW_AUTOSIZE);

  PUBLISH(CV_PTLOC_INSIDE);
  PUBLISH(CV_PTLOC_ON_EDGE);
  PUBLISH(CV_PTLOC_VERTEX);
  PUBLISH(CV_PTLOC_OUTSIDE_RECT);

  PUBLISH(GC_BGD);
  PUBLISH(GC_FGD);
  PUBLISH(GC_PR_BGD);
  PUBLISH(GC_PR_FGD);
  PUBLISH(GC_INIT_WITH_RECT);
  PUBLISH(GC_INIT_WITH_MASK);
  PUBLISH(GC_EVAL);

  PUBLISH(CV_ROW_SAMPLE);
  PUBLISH(CV_VAR_NUMERICAL);
  PUBLISH(CV_VAR_ORDERED);
  PUBLISH(CV_VAR_CATEGORICAL);

  PUBLISH(CV_AA);

#include "pyopencv_generated_const_reg.h"
}
