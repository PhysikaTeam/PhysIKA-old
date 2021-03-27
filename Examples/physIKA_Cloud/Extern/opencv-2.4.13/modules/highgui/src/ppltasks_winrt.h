/***
* ==++==
*
* Copyright (c) Microsoft Corporation. All rights reserved.
*
* Modified for native C++ WRL support by Gregory Morse
*
* ==--==
* =+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+=+
*
* ppltasks_winrt.h
*
* Parallel Patterns Library - PPL Tasks
*
* =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
****/

#pragma once

#ifndef _PPLTASKS_WINRT_H
#define _PPLTASKS_WINRT_H

#include <concrt.h>
#include <ppltasks.h>
#if _MSC_VER >= 1800
#include <pplconcrt.h>

// Cannot build using a compiler that is older than dev10 SP1
#ifdef _MSC_VER
#if _MSC_FULL_VER < 160040219 /*IFSTRIP=IGN*/
#error ERROR: Visual Studio 2010 SP1 or later is required to build ppltasks
#endif /*IFSTRIP=IGN*/
#endif
#else
#include <ppl.h>
#endif
#include <functional>
#include <vector>
#include <utility>
#include <exception>
#if _MSC_VER >= 1800
#include <algorithm>
#endif

#ifndef __cplusplus_winrt

#include <wrl\implements.h>
#include <wrl\async.h>
#if _MSC_VER >= 1800
#include "agile_wrl.h"
#endif
#include <windows.foundation.h>
#include <ctxtcall.h>

#ifndef _UITHREADCTXT_SUPPORT

#ifdef WINAPI_FAMILY /*IFSTRIP=IGN*/

// It is safe to include winapifamily as WINAPI_FAMILY was defined by the user
#include <winapifamily.h>

#if WINAPI_FAMILY == WINAPI_FAMILY_APP /*IFSTRIP=IGN*/
    // UI thread context support is not required for desktop and Windows Store apps
    #define _UITHREADCTXT_SUPPORT 0
#elif WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP /*IFSTRIP=IGN*/
    // UI thread context support is not required for desktop and Windows Store apps
    #define _UITHREADCTXT_SUPPORT 0
#else  /* WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP */
    #define _UITHREADCTXT_SUPPORT 1
#endif  /* WINAPI_FAMILY == WINAPI_FAMILY_DESKTOP_APP */

#else   /* WINAPI_FAMILY */
    // Not supported without a WINAPI_FAMILY setting.
    #define _UITHREADCTXT_SUPPORT 0
#endif  /* WINAPI_FAMILY */

#endif  /* _UITHREADCTXT_SUPPORT */

#if _UITHREADCTXT_SUPPORT
#include <uithreadctxt.h>
#endif  /* _UITHREADCTXT_SUPPORT */

#pragma detect_mismatch("_PPLTASKS_WITH_WINRT", "0")

#ifdef _DEBUG
#define _DBG_ONLY(X) X
#else
#define _DBG_ONLY(X)
#endif // #ifdef _DEBUG

// std::copy_exception changed to std::make_exception_ptr from VS 2010 to VS 11.
#ifdef _MSC_VER
#if _MSC_VER < 1700 /*IFSTRIP=IGN*/
namespace std
{
    template<class _E> exception_ptr make_exception_ptr(_E _Except)
    {
        return copy_exception(_Except);
    }
}
#endif
#ifndef _PPLTASK_ASYNC_LOGGING
#if _MSC_VER >= 1800 && defined(__cplusplus_winrt)
#define _PPLTASK_ASYNC_LOGGING 1  // Only enable async logging under dev12 winrt
#else
#define _PPLTASK_ASYNC_LOGGING 0
#endif
#endif
#endif

#pragma pack(push,_CRT_PACKING)

#pragma warning(push)
#pragma warning(disable: 28197)
#pragma warning(disable: 4100) // Unreferenced formal parameter - needed for document generation
#if _MSC_VER >= 1800
#pragma warning(disable: 4127) // constant express in if condition - we use it for meta programming
#else
#pragma warning(disable: 4702) // Unreachable code - it is caused by user lambda throw exceptions
#endif

// All CRT public header files are required to be protected from the macro new
#pragma push_macro("new")
#undef new

// stuff ported from Dev11 CRT
// NOTE: this doesn't actually match std::declval. it behaves differently for void!
// so don't blindly change it to std::declval.
namespace stdx
{
    template<class _T>
    _T&& declval();
}

/// <summary>
///     The <c>Concurrency_winrt</c> namespace provides classes and functions that give you access to the Concurrency Runtime,
///     a concurrent programming framework for C++. For more information, see <see cref="Concurrency Runtime"/>.
/// </summary>
/**/
namespace Concurrency_winrt
{
    // In debug builds, default to 10 frames, unless this is overridden prior to #includ'ing ppltasks.h.  In retail builds, default to only one frame.
#ifndef PPL_TASK_SAVE_FRAME_COUNT
#ifdef _DEBUG
#define PPL_TASK_SAVE_FRAME_COUNT 10
#else
#define PPL_TASK_SAVE_FRAME_COUNT 1
#endif
#endif

    /// <summary>
    /// Helper macro to determine how many stack frames need to be saved. When any number less or equal to 1 is specified,
    /// only one frame is captured and no stackwalk will be involved. Otherwise, the number of callstack frames will be captured.
    /// </summary>
    /// <ramarks>
    /// This needs to be defined as a macro rather than a function so that if we're only gathering one frame, _ReturnAddress()
    /// will evaluate to client code, rather than a helper function inside of _TaskCreationCallstack, itself.
    /// </remarks>
#ifdef _CAPTURE_CALLSTACK
#undef _CAPTURE_CALLSTACK
#endif
#if PPL_TASK_SAVE_FRAME_COUNT > 1
#if !defined(_DEBUG)
#pragma message ("WARNING: Redefinning PPL_TASK_SAVE_FRAME_COUNT under Release build for non-desktop applications is not supported; only one frame will be captured!")
#define _CAPTURE_CALLSTACK() ::Concurrency_winrt::details::_TaskCreationCallstack::_CaptureSingleFrameCallstack(_ReturnAddress())
#else
#define _CAPTURE_CALLSTACK() ::Concurrency_winrt::details::_TaskCreationCallstack::_CaptureMultiFramesCallstack(PPL_TASK_SAVE_FRAME_COUNT)
#endif
#else
#define _CAPTURE_CALLSTACK() ::Concurrency_winrt::details::_TaskCreationCallstack::_CaptureSingleFrameCallstack(_ReturnAddress())
#endif
/// <summary>

///     A type that represents the terminal state of a task. Valid values are <c>completed</c> and <c>canceled</c>.
/// </summary>
/// <seealso cref="task Class"/>
/**/
typedef Concurrency::task_group_status task_status;

template <typename _Type> class task;
template <> class task<void>;

/// <summary>
///     Returns an indication of whether the task that is currently executing has received a request to cancel its
///     execution. Cancellation is requested on a task if the task was created with a cancellation token, and
///     the token source associated with that token is canceled.
/// </summary>
/// <returns>
///     <c>true</c> if the currently executing task has received a request for cancellation, <c>false</c> otherwise.
/// </returns>
/// <remarks>
///     If you call this method in the body of a task and it returns <c>true</c>, you must respond with a call to
///     <see cref="cancel_current_task Function">cancel_current_task</see> to acknowledge the cancellation request,
///     after performing any cleanup you need. This will abort the execution of the task and cause it to enter into
///     the <c>canceled</c> state. If you do not respond and continue execution, or return instead of calling
///     <c>cancel_current_task</c>, the task will enter the <c>completed</c> state when it is done.
///     state.
///     <para>A task is not cancellable if it was created without a cancellation token.</para>
/// </remarks>
/// <seealso cref="task Class"/>
/// <seealso cref="cancellation_token_source Class"/>
/// <seealso cref="cancellation_token Class"/>
/// <seealso cref="cancel_current_task Function"/>
/**/
#if _MSC_VER >= 1800
inline bool __cdecl is_task_cancellation_requested()
{
    return ::Concurrency::details::_TaskCollection_t::_Is_cancellation_requested();
}
#else
inline bool __cdecl is_task_cancellation_requested()
{
    // ConcRT scheduler under the hood is using TaskCollection, which is same as task_group
    return ::Concurrency::is_current_task_group_canceling();
}
#endif

/// <summary>
///     Cancels the currently executing task. This function can be called from within the body of a task to abort the
///     task's execution and cause it to enter the <c>canceled</c> state. While it may be used in response to
///     the <see cref="is_task_cancellation_requested Function">is_task_cancellation_requested</see> function, you may
///     also use it by itself, to initiate cancellation of the task that is currently executing.
///     <para>It is not a supported scenario to call this function if you are not within the body of a <c>task</c>.
///     Doing so will result in undefined behavior such as a crash or a hang in your application.</para>
/// </summary>
/// <seealso cref="task Class"/>
/// <seealso cref="is_task_cancellation_requested"/>
/**/
//#if _MSC_VER >= 1800
inline __declspec(noreturn) void __cdecl cancel_current_task()
{
    throw Concurrency::task_canceled();
}
//#else
//_CRTIMP2 __declspec(noreturn) void __cdecl cancel_current_task();
//#endif

namespace details
{
#if _MSC_VER >= 1800
    /// <summary>
    ///     Callstack container, which is used to capture and preserve callstacks in ppltasks.
    ///     Members of this class is examined by vc debugger, thus there will be no public access methods.
    ///     Please note that names of this class should be kept stable for debugger examining.
    /// </summary>
    class _TaskCreationCallstack
    {
    private:
        // If _M_SingleFrame != nullptr, there will be only one frame of callstacks, which is stored in _M_SingleFrame;
        // otherwise, _M_Frame will store all the callstack frames.
        void* _M_SingleFrame;
        std::vector<void *> _M_frames;
    public:
        _TaskCreationCallstack()
        {
            _M_SingleFrame = nullptr;
        }

        // Store one frame of callstack. This function works for both Debug / Release CRT.
        static _TaskCreationCallstack _CaptureSingleFrameCallstack(void *_SingleFrame)
        {
            _TaskCreationCallstack _csc;
            _csc._M_SingleFrame = _SingleFrame;
            return _csc;
        }

        // Capture _CaptureFrames number of callstack frames. This function only work properly for Desktop or Debug CRT.
        __declspec(noinline)
            static _TaskCreationCallstack _CaptureMultiFramesCallstack(size_t _CaptureFrames)
        {
                _TaskCreationCallstack _csc;
                _csc._M_frames.resize(_CaptureFrames);
                // skip 2 frames to make sure callstack starts from user code
                _csc._M_frames.resize(::Concurrency::details::platform::CaptureCallstack(&_csc._M_frames[0], 2, _CaptureFrames));
                return _csc;
        }
    };
#endif
    typedef UINT32 _Unit_type;

    struct _TypeSelectorNoAsync {};
    struct _TypeSelectorAsyncOperationOrTask {};
    struct _TypeSelectorAsyncOperation : public _TypeSelectorAsyncOperationOrTask { };
    struct _TypeSelectorAsyncTask : public _TypeSelectorAsyncOperationOrTask { };
    struct _TypeSelectorAsyncAction {};
    struct _TypeSelectorAsyncActionWithProgress {};
    struct _TypeSelectorAsyncOperationWithProgress {};

    template<typename _Ty>
    struct _NormalizeVoidToUnitType
    {
        typedef _Ty _Type;
    };

    template<>
    struct _NormalizeVoidToUnitType<void>
    {
        typedef _Unit_type _Type;
    };

    template<typename _T>
    struct _IsUnwrappedAsyncSelector
    {
        static const bool _Value = true;
    };

    template<>
    struct _IsUnwrappedAsyncSelector<_TypeSelectorNoAsync>
    {
        static const bool _Value = false;
    };

    template <typename _Ty>
    struct _UnwrapTaskType
    {
        typedef _Ty _Type;
    };

    template <typename _Ty>
    struct _UnwrapTaskType<task<_Ty>>
    {
        typedef _Ty _Type;
    };

    template <typename _T>
    _TypeSelectorAsyncTask _AsyncOperationKindSelector(task<_T>);

    _TypeSelectorNoAsync _AsyncOperationKindSelector(...);

    template <typename _Type>
    struct _Unhat
    {
        typedef _Type _Value;
    };

    template <typename _Type>
    struct _Unhat<_Type*>
    {
        typedef _Type _Value;
    };

    //struct _NonUserType { public: int _Dummy; };

    template <typename _Type, bool _IsValueTypeOrRefType = __is_valid_winrt_type(_Type)>
    struct _ValueTypeOrRefType
    {
        typedef _Unit_type _Value;
    };

    template <typename _Type>
    struct _ValueTypeOrRefType<_Type, true>
    {
        typedef _Type _Value;
    };

    template <typename _Ty>
    _Ty _UnwrapAsyncActionWithProgressSelector(ABI::Windows::Foundation::IAsyncActionWithProgress_impl<_Ty>*);

    template <typename _Ty>
    _Ty _UnwrapAsyncActionWithProgressSelector(...);

    template <typename _Ty, typename _Progress>
    _Progress _UnwrapAsyncOperationWithProgressProgressSelector(ABI::Windows::Foundation::IAsyncOperationWithProgress_impl<_Ty, _Progress>*);

    template <typename _Ty, typename _Progress>
    _Progress _UnwrapAsyncOperationWithProgressProgressSelector(...);

    template <typename _T1, typename _T2>
    _T2 _ProgressTypeSelector(ABI::Windows::Foundation::IAsyncOperationWithProgress<_T1, _T2>*);

    template <typename _T1>
    _T1 _ProgressTypeSelector(ABI::Windows::Foundation::IAsyncActionWithProgress<_T1>*);

    template <typename _Type>
    struct _GetProgressType
    {
        typedef decltype(_ProgressTypeSelector(stdx::declval<_Type>())) _Value;
    };

    template <typename _T>
    _TypeSelectorAsyncOperation _AsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncOperation<_T>*);

    _TypeSelectorAsyncAction _AsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncAction*);

    template <typename _T1, typename _T2>
    _TypeSelectorAsyncOperationWithProgress _AsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncOperationWithProgress<_T1, _T2>*);

    template <typename _T>
    _TypeSelectorAsyncActionWithProgress _AsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncActionWithProgress<_T>*);

    template <typename _Type>
    struct _IsIAsyncInfo
    {
        static const bool _Value = std::is_base_of<ABI::Windows::Foundation::IAsyncInfo, typename _Unhat<_Type>::_Value>::value ||
            std::is_same<_TypeSelectorAsyncAction, decltype(details::_AsyncOperationKindSelector(stdx::declval<_Type>()))>::value ||
            std::is_same<_TypeSelectorAsyncOperation, decltype(details::_AsyncOperationKindSelector(stdx::declval<_Type>()))>::value ||
            std::is_same<_TypeSelectorAsyncOperationWithProgress, decltype(details::_AsyncOperationKindSelector(stdx::declval<_Type>()))>::value ||
            std::is_same<_TypeSelectorAsyncActionWithProgress, decltype(details::_AsyncOperationKindSelector(stdx::declval<_Type>()))>::value;
    };

    template <>
    struct _IsIAsyncInfo<void>
    {
        static const bool _Value = false;
    };

    template <typename _Ty>
    _Ty _UnwrapAsyncOperationSelector(ABI::Windows::Foundation::IAsyncOperation_impl<_Ty>*);

    template <typename _Ty>
    _Ty _UnwrapAsyncOperationSelector(...);

    template <typename _Ty, typename _Progress>
    _Ty _UnwrapAsyncOperationWithProgressSelector(ABI::Windows::Foundation::IAsyncOperationWithProgress_impl<_Ty, _Progress>*);

    template <typename _Ty, typename _Progress>
    _Ty _UnwrapAsyncOperationWithProgressSelector(...);

    // Unwrap functions for asyncOperations
    template<typename _Ty>
    auto _GetUnwrappedType(ABI::Windows::Foundation::IAsyncOperation<_Ty>*) -> typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperation<_Ty>*>()))>::type;

    void _GetUnwrappedType(ABI::Windows::Foundation::IAsyncAction*);

    template<typename _Ty, typename _Progress>
    auto _GetUnwrappedType(ABI::Windows::Foundation::IAsyncOperationWithProgress<_Ty, _Progress>*) -> typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationWithProgressSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperationWithProgress<_Ty, _Progress>*>()))>::type;

    template<typename _Progress>
    void _GetUnwrappedType(ABI::Windows::Foundation::IAsyncActionWithProgress<_Progress>*);

    template <typename _T>
    _T _ReturnAsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncOperation<_T>*);

    void _ReturnAsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncAction*);

    template <typename _T1, typename _T2>
    _T1 _ReturnAsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncOperationWithProgress<_T1, _T2>*);

    template <typename _T>
    void _ReturnAsyncOperationKindSelector(ABI::Windows::Foundation::IAsyncActionWithProgress<_T>*);

    class _ProgressReporterCtorArgType{};

    template <typename _Type, bool _IsAsync = _IsIAsyncInfo<_Type>::_Value>
    struct _TaskTypeTraits
    {
        typedef typename details::_UnwrapTaskType<_Type>::_Type _TaskRetType;
        typedef _TaskRetType _TaskRetType_abi;
        typedef decltype(_AsyncOperationKindSelector(stdx::declval<_Type>())) _AsyncKind;
        typedef typename details::_NormalizeVoidToUnitType<_TaskRetType>::_Type _NormalizedTaskRetType;

        static const bool _IsAsyncTask = _IsAsync;
        static const bool _IsUnwrappedTaskOrAsync = details::_IsUnwrappedAsyncSelector<_AsyncKind>::_Value;
    };

    template<typename _Type>
    struct _TaskTypeTraits<_Type, true>
    {
        typedef decltype(_ReturnAsyncOperationKindSelector(stdx::declval<_Type>())) _TaskRetType;
        typedef decltype(_GetUnwrappedType(stdx::declval<_Type>())) _TaskRetType_abi;
        typedef _TaskRetType _NormalizedTaskRetType;
        typedef decltype(_AsyncOperationKindSelector(stdx::declval<_Type>())) _AsyncKind;

        static const bool _IsAsyncTask = true;
        static const bool _IsUnwrappedTaskOrAsync = details::_IsUnwrappedAsyncSelector<_AsyncKind>::_Value;
    };

    template <typename _ReturnType, typename _Function> auto _IsCallable(_Function _Func, int, int, int) -> decltype(_Func(stdx::declval<task<_ReturnType>*>()), std::true_type()) { (void)_Func; return std::true_type(); }
    template <typename _ReturnType, typename _Function> auto _IsCallable(_Function _Func, int, int, ...) -> decltype(_Func(stdx::declval<_ReturnType*>()), std::true_type()) { (void)_Func; return std::true_type(); }
    template <typename _ReturnType, typename _Function> auto _IsCallable(_Function _Func, int, ...) -> decltype(_Func(), std::true_type()) { (void)_Func; return std::true_type(); }
    template <typename _ReturnType, typename _Function> std::false_type _IsCallable(_Function, ...) { return std::false_type(); }

    template <>
    struct _TaskTypeTraits<void>
    {
        typedef void _TaskRetType;
        typedef void _TaskRetType_abi;
        typedef _TypeSelectorNoAsync _AsyncKind;
        typedef _Unit_type _NormalizedTaskRetType;

        static const bool _IsAsyncTask = false;
        static const bool _IsUnwrappedTaskOrAsync = false;
    };

    // ***************************************************************************
    // Template type traits and helpers for async production APIs:
    //

    struct _ZeroArgumentFunctor { };
    struct _OneArgumentFunctor { };
    struct _TwoArgumentFunctor { };
    struct _ThreeArgumentFunctor { };

    // ****************************************
    // CLASS TYPES:

    // mutable functions
    // ********************
    // THREE ARGUMENTS:

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg1 _Arg1ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3));

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg2 _Arg2ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3));

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg3 _Arg3ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3));

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3));

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ThreeArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3));

    // ********************
    // TWO ARGUMENTS:

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg1 _Arg1ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2));

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg2 _Arg2ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2));

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    void _Arg3ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2));

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2));

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _TwoArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)(_Arg1, _Arg2));

    // ********************
    // ONE ARGUMENT:

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1>
    _Arg1 _Arg1ClassHelperThunk(_ReturnType(_Class::*)(_Arg1));

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1>
    void _Arg2ClassHelperThunk(_ReturnType(_Class::*)(_Arg1));

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1>
    void _Arg3ClassHelperThunk(_ReturnType(_Class::*)(_Arg1));

    template<typename _Class, typename _ReturnType, typename _Arg1>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)(_Arg1));

    template<typename _Class, typename _ReturnType, typename _Arg1>
    _OneArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)(_Arg1));

    // ********************
    // ZERO ARGUMENT:

    // void arg:
    template<typename _Class, typename _ReturnType>
    void _Arg1ClassHelperThunk(_ReturnType(_Class::*)());

    // void arg:
    template<typename _Class, typename _ReturnType>
    void _Arg2ClassHelperThunk(_ReturnType(_Class::*)());

    // void arg:
    template<typename _Class, typename _ReturnType>
    void _Arg3ClassHelperThunk(_ReturnType(_Class::*)());

    // void arg:
    template<typename _Class, typename _ReturnType>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)());

    template<typename _Class, typename _ReturnType>
    _ZeroArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)());

    // ********************
    // THREE ARGUMENTS:

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg1 _Arg1ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3) const);

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg2 _Arg2ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3) const);

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg3 _Arg3ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3) const);

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3) const);

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ThreeArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)(_Arg1, _Arg2, _Arg3) const);

    // ********************
    // TWO ARGUMENTS:

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg1 _Arg1ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2) const);

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg2 _Arg2ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2) const);

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    void _Arg3ClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2) const);

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)(_Arg1, _Arg2) const);

    template<typename _Class, typename _ReturnType, typename _Arg1, typename _Arg2>
    _TwoArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)(_Arg1, _Arg2) const);

    // ********************
    // ONE ARGUMENT:

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1>
    _Arg1 _Arg1ClassHelperThunk(_ReturnType(_Class::*)(_Arg1) const);

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1>
    void _Arg2ClassHelperThunk(_ReturnType(_Class::*)(_Arg1) const);

    // non-void arg:
    template<typename _Class, typename _ReturnType, typename _Arg1>
    void _Arg3ClassHelperThunk(_ReturnType(_Class::*)(_Arg1) const);

    template<typename _Class, typename _ReturnType, typename _Arg1>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)(_Arg1) const);

    template<typename _Class, typename _ReturnType, typename _Arg1>
    _OneArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)(_Arg1) const);

    // ********************
    // ZERO ARGUMENT:

    // void arg:
    template<typename _Class, typename _ReturnType>
    void _Arg1ClassHelperThunk(_ReturnType(_Class::*)() const);

    // void arg:
    template<typename _Class, typename _ReturnType>
    void _Arg2ClassHelperThunk(_ReturnType(_Class::*)() const);

    // void arg:
    template<typename _Class, typename _ReturnType>
    void _Arg3ClassHelperThunk(_ReturnType(_Class::*)() const);

    // void arg:
    template<typename _Class, typename _ReturnType>
    _ReturnType _ReturnTypeClassHelperThunk(_ReturnType(_Class::*)() const);

    template<typename _Class, typename _ReturnType>
    _ZeroArgumentFunctor _ArgumentCountHelper(_ReturnType(_Class::*)() const);

    // ****************************************
    // POINTER TYPES:

    // ********************
    // THREE ARGUMENTS:

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg2 _Arg2PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg3 _Arg3PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ThreeArgumentFunctor _ArgumentCountHelper(_ReturnType(__cdecl *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg2 _Arg2PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg3 _Arg3PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ThreeArgumentFunctor _ArgumentCountHelper(_ReturnType(__stdcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg2 _Arg2PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _Arg3 _Arg3PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2, _Arg3));

    template<typename _ReturnType, typename _Arg1, typename _Arg2, typename _Arg3>
    _ThreeArgumentFunctor _ArgumentCountHelper(_ReturnType(__fastcall *)(_Arg1, _Arg2, _Arg3));

    // ********************
    // TWO ARGUMENTS:

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg2 _Arg2PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    void _Arg3PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__cdecl *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _TwoArgumentFunctor _ArgumentCountHelper(_ReturnType(__cdecl *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg2 _Arg2PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    void _Arg3PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__stdcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _TwoArgumentFunctor _ArgumentCountHelper(_ReturnType(__stdcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _Arg2 _Arg2PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    void _Arg3PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__fastcall *)(_Arg1, _Arg2));

    template<typename _ReturnType, typename _Arg1, typename _Arg2>
    _TwoArgumentFunctor _ArgumentCountHelper(_ReturnType(__fastcall *)(_Arg1, _Arg2));

    // ********************
    // ONE ARGUMENT:

    template<typename _ReturnType, typename _Arg1>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    void _Arg2PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    void _Arg3PFNHelperThunk(_ReturnType(__cdecl *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__cdecl *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _OneArgumentFunctor _ArgumentCountHelper(_ReturnType(__cdecl *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    void _Arg2PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    void _Arg3PFNHelperThunk(_ReturnType(__stdcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__stdcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _OneArgumentFunctor _ArgumentCountHelper(_ReturnType(__stdcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _Arg1 _Arg1PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    void _Arg2PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    void _Arg3PFNHelperThunk(_ReturnType(__fastcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__fastcall *)(_Arg1));

    template<typename _ReturnType, typename _Arg1>
    _OneArgumentFunctor _ArgumentCountHelper(_ReturnType(__fastcall *)(_Arg1));

    // ********************
    // ZERO ARGUMENT:

    template<typename _ReturnType>
    void _Arg1PFNHelperThunk(_ReturnType(__cdecl *)());

    template<typename _ReturnType>
    void _Arg2PFNHelperThunk(_ReturnType(__cdecl *)());

    template<typename _ReturnType>
    void _Arg3PFNHelperThunk(_ReturnType(__cdecl *)());

    template<typename _ReturnType>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__cdecl *)());

    template<typename _ReturnType>
    _ZeroArgumentFunctor _ArgumentCountHelper(_ReturnType(__cdecl *)());

    template<typename _ReturnType>
    void _Arg1PFNHelperThunk(_ReturnType(__stdcall *)());

    template<typename _ReturnType>
    void _Arg2PFNHelperThunk(_ReturnType(__stdcall *)());

    template<typename _ReturnType>
    void _Arg3PFNHelperThunk(_ReturnType(__stdcall *)());

    template<typename _ReturnType>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__stdcall *)());

    template<typename _ReturnType>
    _ZeroArgumentFunctor _ArgumentCountHelper(_ReturnType(__stdcall *)());

    template<typename _ReturnType>
    void _Arg1PFNHelperThunk(_ReturnType(__fastcall *)());

    template<typename _ReturnType>
    void _Arg2PFNHelperThunk(_ReturnType(__fastcall *)());

    template<typename _ReturnType>
    void _Arg3PFNHelperThunk(_ReturnType(__fastcall *)());

    template<typename _ReturnType>
    _ReturnType _ReturnTypePFNHelperThunk(_ReturnType(__fastcall *)());

    template<typename _ReturnType>
    _ZeroArgumentFunctor _ArgumentCountHelper(_ReturnType(__fastcall *)());

    template<typename _T>
    struct _FunctorArguments
    {
        static const size_t _Count = 0;
    };

    template<>
    struct _FunctorArguments<_OneArgumentFunctor>
    {
        static const size_t _Count = 1;
    };

    template<>
    struct _FunctorArguments<_TwoArgumentFunctor>
    {
        static const size_t _Count = 2;
    };

    template<>
    struct _FunctorArguments<_ThreeArgumentFunctor>
    {
        static const size_t _Count = 3;
    };

    template<typename _T>
    struct _FunctorTypeTraits
    {
        typedef decltype(_ArgumentCountHelper(&(_T::operator()))) _ArgumentCountType;
        static const size_t _ArgumentCount = _FunctorArguments<_ArgumentCountType>::_Count;

        typedef decltype(_ReturnTypeClassHelperThunk(&(_T::operator()))) _ReturnType;
        typedef decltype(_Arg1ClassHelperThunk(&(_T::operator()))) _Argument1Type;
        typedef decltype(_Arg2ClassHelperThunk(&(_T::operator()))) _Argument2Type;
        typedef decltype(_Arg3ClassHelperThunk(&(_T::operator()))) _Argument3Type;
    };

    template<typename _T>
    struct _FunctorTypeTraits<_T *>
    {
        typedef decltype(_ArgumentCountHelper(stdx::declval<_T*>())) _ArgumentCountType;
        static const size_t _ArgumentCount = _FunctorArguments<_ArgumentCountType>::_Count;

        typedef decltype(_ReturnTypePFNHelperThunk(stdx::declval<_T*>())) _ReturnType;
        typedef decltype(_Arg1PFNHelperThunk(stdx::declval<_T*>())) _Argument1Type;
        typedef decltype(_Arg2PFNHelperThunk(stdx::declval<_T*>())) _Argument2Type;
        typedef decltype(_Arg3PFNHelperThunk(stdx::declval<_T*>())) _Argument3Type;
    };

    task<void> _To_task();

    template <typename _Function> auto _IsVoidConversionHelper(_Function _Func, int) -> typename decltype(_Func(_To_task()), std::true_type());
    template <typename _Function> std::false_type _IsVoidConversionHelper(_Function _Func, ...);

    template <typename T> std::true_type _VoidIsTaskHelper(task<T> _Arg, int);
    template <typename T> std::false_type _VoidIsTaskHelper(T _Arg, ...);

    template<typename _Function, typename _ExpectedParameterType, const bool _IsVoidConversion = std::is_same<decltype(_IsVoidConversionHelper(stdx::declval<_Function>(), 0)), std::true_type>::value, const size_t _Count = _FunctorTypeTraits<_Function>::_ArgumentCount>
    struct _FunctionTypeTraits
    {
        typedef typename _Unhat<typename _FunctorTypeTraits<_Function>::_Argument2Type>::_Value _FuncRetType;
        static_assert(std::is_same<typename _FunctorTypeTraits<_Function>::_Argument1Type, _ExpectedParameterType>::value ||
                    std::is_same<typename _FunctorTypeTraits<_Function>::_Argument1Type, task<_ExpectedParameterType>>::value, "incorrect parameter type for the callable object in 'then'; consider _ExpectedParameterType or task<_ExpectedParameterType> (see below)");

        typedef decltype(_VoidIsTaskHelper(stdx::declval<_FunctorTypeTraits<_Function>::_Argument1Type>(), 0)) _Takes_task;
    };

    //if there is a continuation parameter, then must use void/no return value
    template<typename _Function, typename _ExpectedParameterType, const bool _IsVoidConversion>
    struct _FunctionTypeTraits<_Function, _ExpectedParameterType, _IsVoidConversion, 1>
    {
        typedef void _FuncRetType;
        static_assert(std::is_same<typename _FunctorTypeTraits<_Function>::_Argument1Type, _ExpectedParameterType>::value ||
                    std::is_same<typename _FunctorTypeTraits<_Function>::_Argument1Type, task<_ExpectedParameterType>>::value, "incorrect parameter type for the callable object in 'then'; consider _ExpectedParameterType or task<_ExpectedParameterType> (see below)");

        typedef decltype(_VoidIsTaskHelper(stdx::declval<_FunctorTypeTraits<_Function>::_Argument1Type>(), 0)) _Takes_task;
    };

    template<typename _Function>
    struct _FunctionTypeTraits<_Function, void, true, 1>
    {
        typedef void _FuncRetType;
        static_assert(std::is_same<typename _FunctorTypeTraits<_Function>::_Argument1Type, decltype(_To_task())>::value, "incorrect parameter type for the callable object in 'then'; consider _ExpectedParameterType or task<_ExpectedParameterType> (see below)");

        typedef decltype(_VoidIsTaskHelper(stdx::declval<_FunctorTypeTraits<_Function>::_Argument1Type>(), 0)) _Takes_task;
    };

    template<typename _Function>
    struct _FunctionTypeTraits<_Function, void, false, 1>
    {
        typedef typename _Unhat<typename _FunctorTypeTraits<_Function>::_Argument1Type>::_Value _FuncRetType;

        typedef std::false_type _Takes_task;
    };

    template<typename _Function, typename _ExpectedParameterType, const bool _IsVoidConversion>
    struct _FunctionTypeTraits<_Function, _ExpectedParameterType, _IsVoidConversion, 0>
    {
        typedef void _FuncRetType;

        typedef std::false_type _Takes_task;
    };

    template<typename _Function, typename _ReturnType>
    struct _ContinuationTypeTraits
    {
        typedef typename task<typename _TaskTypeTraits<typename _FunctionTypeTraits<_Function, _ReturnType>::_FuncRetType>::_TaskRetType_abi> _TaskOfType;
    };

    // _InitFunctorTypeTraits is used to decide whether a task constructed with a lambda should be unwrapped. Depending on how the variable is
    // declared, the constructor may or may not perform unwrapping. For eg.
    //
    //  This declaration SHOULD NOT cause unwrapping
    //    task<task<void>> t1([]() -> task<void> {
    //        task<void> t2([]() {});
    //        return t2;
    //    });
    //
    // This declaration SHOULD cause unwrapping
    //    task<void>> t1([]() -> task<void> {
    //        task<void> t2([]() {});
    //        return t2;
    //    });
    // If the type of the task is the same as the return type of the function, no unwrapping should take place. Else normal rules apply.
    template <typename _TaskType, typename _FuncRetType>
    struct _InitFunctorTypeTraits
    {
        typedef typename _TaskTypeTraits<_FuncRetType>::_AsyncKind _AsyncKind;
        static const bool _IsAsyncTask = _TaskTypeTraits<_FuncRetType>::_IsAsyncTask;
        static const bool _IsUnwrappedTaskOrAsync = _TaskTypeTraits<_FuncRetType>::_IsUnwrappedTaskOrAsync;
    };

    template<typename T>
    struct _InitFunctorTypeTraits<T, T>
    {
        typedef _TypeSelectorNoAsync _AsyncKind;
        static const bool _IsAsyncTask = false;
        static const bool _IsUnwrappedTaskOrAsync = false;
    };
    /// <summary>
    ///     Helper object used for LWT invocation.
    /// </summary>
    struct _TaskProcThunk
    {
        _TaskProcThunk(const std::function<HRESULT(void)> & _Callback) :
        _M_func(_Callback)
        {
        }

        static void __cdecl _Bridge(void *_PData)
        {
            _TaskProcThunk *_PThunk = reinterpret_cast<_TaskProcThunk *>(_PData);
#if _MSC_VER >= 1800
            _Holder _ThunkHolder(_PThunk);
#endif
            _PThunk->_M_func();
#if _MSC_VER < 1800
            delete _PThunk;
#endif
        }
    private:
#if _MSC_VER >= 1800
        // RAII holder
        struct _Holder
        {
            _Holder(_TaskProcThunk * _PThunk) : _M_pThunk(_PThunk)
            {
            }

            ~_Holder()
            {
                delete _M_pThunk;
            }

            _TaskProcThunk * _M_pThunk;

        private:
            _Holder& operator=(const _Holder&);
        };
#endif
        std::function<HRESULT(void)> _M_func;
        _TaskProcThunk& operator=(const _TaskProcThunk&);
    };

    /// <summary>
    ///     Schedule a functor with automatic inlining. Note that this is "fire and forget" scheduling, which cannot be
    ///     waited on or canceled after scheduling.
    ///     This schedule method will perform automatic inlining base on <paramref value="_InliningMode"/>.
    /// </summary>
    /// <param name="_Func">
    ///     The user functor need to be scheduled.
    /// </param>
    /// <param name="_InliningMode">
    ///     The inlining scheduling policy for current functor.
    /// </param>
#if _MSC_VER >= 1800
    typedef Concurrency::details::_TaskInliningMode_t _TaskInliningMode;
#else
    typedef Concurrency::details::_TaskInliningMode _TaskInliningMode;
#endif
    static void _ScheduleFuncWithAutoInline(const std::function<HRESULT(void)> & _Func, _TaskInliningMode _InliningMode)
    {
#if _MSC_VER >= 1800
        Concurrency::details::_TaskCollection_t::_RunTask(&_TaskProcThunk::_Bridge, new _TaskProcThunk(_Func), _InliningMode);
#else
        Concurrency::details::_StackGuard _Guard;
        if (_Guard._ShouldInline(_InliningMode))
        {
            _Func();
        }
        else
        {
            Concurrency::details::_CurrentScheduler::_ScheduleTask(reinterpret_cast<Concurrency::TaskProc>(&_TaskProcThunk::_Bridge), new _TaskProcThunk(_Func));
        }
#endif
    }
    class _ContextCallback
    {
        typedef std::function<HRESULT(void)> _CallbackFunction;

    public:

        static _ContextCallback _CaptureCurrent()
        {
            _ContextCallback _Context;
            _Context._Capture();
            return _Context;
        }

        ~_ContextCallback()
        {
            _Reset();
        }

        _ContextCallback(bool _DeferCapture = false)
        {
            if (_DeferCapture)
            {
                _M_context._M_captureMethod = _S_captureDeferred;
            }
            else
            {
                _M_context._M_pContextCallback = nullptr;
            }
        }

        // Resolves a context that was created as _S_captureDeferred based on the environment (ancestor, current context).
        void _Resolve(bool _CaptureCurrent)
        {
            if (_M_context._M_captureMethod == _S_captureDeferred)
            {
                _M_context._M_pContextCallback = nullptr;

                if (_CaptureCurrent)
                {
                    if (_IsCurrentOriginSTA())
                    {
                        _Capture();
                    }
#if _UITHREADCTXT_SUPPORT
                    else
                    {
                        // This method will fail if not called from the UI thread.
                        HRESULT _Hr = CaptureUiThreadContext(&_M_context._M_pContextCallback);
                        if (FAILED(_Hr))
                        {
                            _M_context._M_pContextCallback = nullptr;
                        }
                    }
#endif // _UITHREADCTXT_SUPPORT
                }
            }
        }

        void _Capture()
        {
            HRESULT _Hr = CoGetObjectContext(IID_IContextCallback, reinterpret_cast<void **>(&_M_context._M_pContextCallback));
            if (FAILED(_Hr))
            {
                _M_context._M_pContextCallback = nullptr;
            }
        }

        _ContextCallback(const _ContextCallback& _Src)
        {
            _Assign(_Src._M_context._M_pContextCallback);
        }

        _ContextCallback(_ContextCallback&& _Src)
        {
            _M_context._M_pContextCallback = _Src._M_context._M_pContextCallback;
            _Src._M_context._M_pContextCallback = nullptr;
        }

        _ContextCallback& operator=(const _ContextCallback& _Src)
        {
            if (this != &_Src)
            {
                _Reset();
                _Assign(_Src._M_context._M_pContextCallback);
            }
            return *this;
        }

        _ContextCallback& operator=(_ContextCallback&& _Src)
        {
            if (this != &_Src)
            {
                _M_context._M_pContextCallback = _Src._M_context._M_pContextCallback;
                _Src._M_context._M_pContextCallback = nullptr;
            }
            return *this;
        }

        bool _HasCapturedContext() const
        {
            _CONCRT_ASSERT(_M_context._M_captureMethod != _S_captureDeferred);
            return (_M_context._M_pContextCallback != nullptr);
        }

        HRESULT _CallInContext(_CallbackFunction _Func) const
        {
            if (!_HasCapturedContext())
            {
                _Func();
            }
            else
            {
                ComCallData callData;
                ZeroMemory(&callData, sizeof(callData));
                callData.pUserDefined = reinterpret_cast<void *>(&_Func);

                HRESULT _Hr = _M_context._M_pContextCallback->ContextCallback(&_Bridge, &callData, IID_ICallbackWithNoReentrancyToApplicationSTA, 5, nullptr);
                if (FAILED(_Hr))
                {
                    return _Hr;
                }
            }
            return S_OK;
        }

        bool operator==(const _ContextCallback& _Rhs) const
        {
            return (_M_context._M_pContextCallback == _Rhs._M_context._M_pContextCallback);
        }

        bool operator!=(const _ContextCallback& _Rhs) const
        {
            return !(operator==(_Rhs));
        }

    private:

        void _Reset()
        {
            if (_M_context._M_captureMethod != _S_captureDeferred && _M_context._M_pContextCallback != nullptr)
            {
                _M_context._M_pContextCallback->Release();
            }
        }

        void _Assign(IContextCallback *_PContextCallback)
        {
            _M_context._M_pContextCallback = _PContextCallback;
            if (_M_context._M_captureMethod != _S_captureDeferred && _M_context._M_pContextCallback != nullptr)
            {
                _M_context._M_pContextCallback->AddRef();
            }
        }

        static HRESULT __stdcall _Bridge(ComCallData *_PParam)
        {
            _CallbackFunction *pFunc = reinterpret_cast<_CallbackFunction *>(_PParam->pUserDefined);
            return (*pFunc)();
        }

        // Returns the origin information for the caller (runtime / Windows Runtime apartment as far as task continuations need know)
        static bool _IsCurrentOriginSTA()
        {
            APTTYPE _AptType;
            APTTYPEQUALIFIER _AptTypeQualifier;

            HRESULT hr = CoGetApartmentType(&_AptType, &_AptTypeQualifier);
            if (SUCCEEDED(hr))
            {
                // We determine the origin of a task continuation by looking at where .then is called, so we can tell whether
                // to need to marshal the continuation back to the originating apartment. If an STA thread is in executing in
                // a neutral aparment when it schedules a continuation, we will not marshal continuations back to the STA,
                // since variables used within a neutral apartment are expected to be apartment neutral.
                switch (_AptType)
                {
                case APTTYPE_MAINSTA:
                case APTTYPE_STA:
                    return true;
                default:
                    break;
                }
            }
            return false;
        }

        union
        {
            IContextCallback *_M_pContextCallback;
            size_t _M_captureMethod;
        } _M_context;

        static const size_t _S_captureDeferred = 1;
    };

#if _MSC_VER >= 1800
    template<typename _Type>
    struct _ResultHolder
    {
        void Set(const _Type& _type)
        {
            _Result = _type;
        }

        _Type Get()
        {
            return _Result;
        }

        _Type _Result;
    };

    template<typename _Type>
    struct _ResultHolder<_Type*>
    {
        void Set(_Type* const & _type)
        {
            _M_Result = _type;
        }

        _Type* Get()
        {
            return _M_Result.Get();
        }
    private:
        // ::Platform::Agile handle specialization of all hats
        // including ::Platform::String and ::Platform::Array
        Agile<_Type*> _M_Result;
    };

    //
    // The below are for composability with tasks auto-created from when_any / when_all / && / || constructs.
    //
    template<typename _Type>
    struct _ResultHolder<std::vector<_Type*>>
    {
        void Set(const std::vector<_Type*>& _type)
        {
            _Result.reserve(_type.size());

            for (auto _PTask = _type.begin(); _PTask != _type.end(); ++_PTask)
            {
                _Result.emplace_back(*_PTask);
            }
        }

        std::vector<_Type*> Get()
        {
            // Return vectory<T^> with the objects that are marshaled in the proper appartment
            std::vector<_Type*> _Return;
            _Return.reserve(_Result.size());

            for (auto _PTask = _Result.begin(); _PTask != _Result.end(); ++_PTask)
            {
                _Return.push_back(_PTask->Get()); // Agile will marshal the object to appropriate appartment if neccessary
            }

            return _Return;
        }

        std::vector< Agile<_Type*> > _Result;
    };

    template<typename _Type>
    struct _ResultHolder<std::pair<_Type*, void*> >
    {
        void Set(const std::pair<_Type*, size_t>& _type)
        {
            _M_Result = _type;
        }

        std::pair<_Type*, size_t> Get()
        {
            return std::make_pair(_M_Result.first, _M_Result.second);
        }
    private:
        std::pair<Agile<_Type*>, size_t> _M_Result;
    };
#else
    template<typename _Type>
    struct _ResultContext
    {
        static _ContextCallback _GetContext(bool /* _RuntimeAggregate */)
        {
            return _ContextCallback();
        }

        static _Type _GetValue(_Type _ObjInCtx, const _ContextCallback & /* _Ctx */, bool /* _RuntimeAggregate */)
        {
            return _ObjInCtx;
        }
    };

    template<typename _Type, size_t N = 0, bool bIsArray = std::is_array<_Type>::value>
    struct _MarshalHelper
    {
    };
    template<typename _Type, size_t N>
    struct _MarshalHelper<_Type, N, true>
    {
        static _Type* _Perform(_Type(&_ObjInCtx)[N], const _ContextCallback& _Ctx)
        {
            static_assert(__is_valid_winrt_type(_Type*), "must be a WinRT array compatible type");
            if (_ObjInCtx == nullptr)
            {
                return nullptr;
            }

            HRESULT _Hr;
            IStream * _PStream;
            _Ctx._CallInContext([&]() -> HRESULT {
                // It isn't safe to simply reinterpret_cast a hat type to IUnknown* because some types do not have a real vtable ptr.
                // Instead, we could to create a property value to make it "grow" the vtable ptr but instead primitives are not marshalled.

                IUnknown * _PUnk = winrt_array_type::create(_ObjInCtx, N);
                _Hr = CoMarshalInterThreadInterfaceInStream(winrt_type<_Type>::getuuid(), _PUnk, &_PStream);
                return S_OK;
            });

            // With an APPX manifest, this call should never fail.
            _CONCRT_ASSERT(SUCCEEDED(_Hr));

            _Type* _Proxy;
            //
            // Cannot use IID_PPV_ARGS with ^ types.
            //
            _Hr = CoGetInterfaceAndReleaseStream(_PStream, winrt_type<_Type>::getuuid(), reinterpret_cast<void**>(&_Proxy));
            if (FAILED(_Hr))
            {
                throw std::make_exception_ptr(_Hr);
            }
            return _Proxy;
        }
    };
    template<typename _Type>
    struct _MarshalHelper<_Type, 0, false>
    {
        static _Type* _Perform(_Type* _ObjInCtx, const _ContextCallback& _Ctx)
        {
            static_assert(std::is_base_of<IUnknown, _Type>::value || __is_valid_winrt_type(_Type), "must be a COM or WinRT type");
            if (_ObjInCtx == nullptr)
            {
                return nullptr;
            }

            HRESULT _Hr;
            IStream * _PStream;
            _Ctx._CallInContext([&]() -> HRESULT {
                // It isn't safe to simply reinterpret_cast a hat type to IUnknown* because some types do not have a real vtable ptr.
                // Instead, we could to create a property value to make it "grow" the vtable ptr but instead primitives are not marshalled.

                IUnknown * _PUnk = winrt_type<_Type>::create(_ObjInCtx);
                _Hr = CoMarshalInterThreadInterfaceInStream(winrt_type<_Type>::getuuid(), _PUnk, &_PStream);
                return S_OK;
            });

            // With an APPX manifest, this call should never fail.
            _CONCRT_ASSERT(SUCCEEDED(_Hr));

            _Type* _Proxy;
            //
            // Cannot use IID_PPV_ARGS with ^ types.
            //
            _Hr = CoGetInterfaceAndReleaseStream(_PStream, winrt_type<_Type>::getuuid(), reinterpret_cast<void**>(&_Proxy));
            if (FAILED(_Hr))
            {
                throw std::make_exception_ptr(_Hr);
            }
            return _Proxy;
        }
    };

    // Arrays must be converted to IPropertyValue objects.

    template<>
    struct _MarshalHelper<HSTRING__>
    {
        static HSTRING _Perform(HSTRING _ObjInCtx, const _ContextCallback& _Ctx)
        {
            return _ObjInCtx;
        }
    };

    template<typename _Type>
    _Type* _Marshal(_Type* _ObjInCtx, const _ContextCallback& _Ctx)
    {
        return _MarshalHelper<_Type>::_Perform(_ObjInCtx, _Ctx);
    }

    template<typename _Type>
    struct _InContext
    {
        static _Type _Get(_Type _ObjInCtx, const _ContextCallback& _Ctx)
        {
            return _ObjInCtx;
        }
    };

    template<typename _Type>
    struct _InContext<_Type*>
    {
        static _Type* _Get(_Type* _ObjInCtx, const _ContextCallback& _Ctx)
        {
            _ContextCallback _CurrentContext = _ContextCallback::_CaptureCurrent();
            if (!_Ctx._HasCapturedContext() || _Ctx == _CurrentContext)
            {
                return _ObjInCtx;
            }

            //
            // The object is from another apartment. If it's marshalable, do so.
            //
            return _Marshal<_Type>(_ObjInCtx, _Ctx);
        }
    };

    template<typename _Type>
    struct _ResultContext<_Type*>
    {
        static _Type* _GetValue(_Type* _ObjInCtx, const _ContextCallback& _Ctx, bool /* _RuntimeAggregate */)
        {
            return _InContext<_Type*>::_Get(_ObjInCtx, _Ctx);
        }

        static _ContextCallback _GetContext(bool /* _RuntimeAggregate */)
        {
            return _ContextCallback::_CaptureCurrent();
        }
    };

    //
    // The below are for composability with tasks auto-created from when_any / when_all / && / || constructs.
    //
    template<typename _Type>
    struct _ResultContext<std::vector<_Type*>>
    {
        static std::vector<_Type*> _GetValue(std::vector<_Type*> _ObjInCtx, const _ContextCallback& _Ctx, bool _RuntimeAggregate)
        {
            if (!_RuntimeAggregate)
            {
                return _ObjInCtx;
            }

            _ContextCallback _CurrentContext = _ContextCallback::_CaptureCurrent();
            if (!_Ctx._HasCapturedContext() || _Ctx == _CurrentContext)
            {
                return _ObjInCtx;
            }

            for (auto _It = _ObjInCtx.begin(); _It != _ObjInCtx.end(); ++_It)
            {
                *_It = _Marshal<_Type>(*_It, _Ctx);
            }

            return _ObjInCtx;
        }

        static _ContextCallback _GetContext(bool _RuntimeAggregate)
        {
            if (!_RuntimeAggregate)
            {
                return _ContextCallback();
            }
            else
            {
                return _ContextCallback::_CaptureCurrent();
            }
        }
    };

    template<typename _Type>
    struct _ResultContext<std::pair<_Type*, size_t>>
    {
        static std::pair<_Type*, size_t> _GetValue(std::pair<_Type*, size_t> _ObjInCtx, const _ContextCallback& _Ctx, bool _RuntimeAggregate)
        {
            if (!_RuntimeAggregate)
            {
                return _ObjInCtx;
            }

            _ContextCallback _CurrentContext = _ContextCallback::_CaptureCurrent();
            if (!_Ctx._HasCapturedContext() || _Ctx == _CurrentContext)
            {
                return _ObjInCtx;
            }

            return std::pair<_Type*, size_t>(_Marshal<_Type>(_ObjInCtx.first, _Ctx), _ObjInCtx.second);
        }

        static _ContextCallback _GetContext(bool _RuntimeAggregate)
        {
            if (!_RuntimeAggregate)
            {
                return _ContextCallback();
            }
            else
            {
                return _ContextCallback::_CaptureCurrent();
            }
        }
    };
#endif
    // An exception thrown by the task body is captured in an exception holder and it is shared with all value based continuations rooted at the task.
    // The exception is 'observed' if the user invokes get()/wait() on any of the tasks that are sharing this exception holder. If the exception
    // is not observed by the time the internal object owned by the shared pointer destructs, the process will fail fast.
    struct _ExceptionHolder
    {
#if _MSC_VER >= 1800
    private:
        void ReportUnhandledError()
        {
            if (_M_winRTException != nullptr)
            {
                throw _M_winRTException.Get();
            }
        }
    public:
        explicit _ExceptionHolder(const std::exception_ptr& _E, const _TaskCreationCallstack &_stackTrace) :
            _M_exceptionObserved(0), _M_stdException(_E), _M_stackTrace(_stackTrace)
        {
        }

        explicit _ExceptionHolder(IRestrictedErrorInfo*& _E, const _TaskCreationCallstack &_stackTrace) :
            _M_exceptionObserved(0), _M_winRTException(_E), _M_stackTrace(_stackTrace)
        {
        }
#else
        explicit _ExceptionHolder(const std::exception_ptr& _E, void* _SourceAddressHint) :
        _M_exceptionObserved(0), _M_stdException(_E), _M_disassembleMe(_SourceAddressHint)
        {
        }

        explicit _ExceptionHolder(IRestrictedErrorInfo*& _E, void* _SourceAddressHint) :
            _M_exceptionObserved(0), _M_disassembleMe(_SourceAddressHint), _M_winRTException(_E)
        {
        }
#endif
        __declspec(noinline)
            ~_ExceptionHolder()
        {
            if (_M_exceptionObserved == 0)
            {
#if _MSC_VER >= 1800
                // If you are trapped here, it means an exception thrown in task chain didn't get handled.
                // Please add task-based continuation to handle all exceptions coming from tasks.
                // this->_M_stackTrace keeps the creation callstack of the task generates this exception.
                _REPORT_PPLTASK_UNOBSERVED_EXCEPTION();
#else
                // Disassemble at this->_M_disassembleMe to get to the source location right after either the creation of the task (constructor
                // or then method) that encountered this exception, or the set_exception call for a task_completion_event.
                Concurrency::details::_ReportUnobservedException();
#endif
            }
        }

        void _RethrowUserException()
        {
            if (_M_exceptionObserved == 0)
            {
#if _MSC_VER >= 1800
                Concurrency::details::atomic_exchange(_M_exceptionObserved, 1l);
#else
                _InterlockedExchange(&_M_exceptionObserved, 1);
#endif
            }

            if (_M_winRTException != nullptr)
            {
                throw _M_winRTException.Get();
            }
            std::rethrow_exception(_M_stdException);
        }

        // A variable that remembers if this exception was every rethrown into user code (and hence handled by the user). Exceptions that
        // are unobserved when the exception holder is destructed will terminate the process.
#if _MSC_VER >= 1800
        Concurrency::details::atomic_long _M_exceptionObserved;
#else
        long volatile _M_exceptionObserved;
#endif

        // Either _M_stdException or _M_winRTException is populated based on the type of exception encountered.
        std::exception_ptr _M_stdException;
        Microsoft::WRL::ComPtr<IRestrictedErrorInfo> _M_winRTException;

        // Disassembling this value will point to a source instruction right after a call instruction. If the call is to create_task,
        // a task constructor or the then method, the task created by that method is the one that encountered this exception. If the call
        // is to task_completion_event::set_exception, the set_exception method was the source of the exception.
        // DO NOT REMOVE THIS VARIABLE. It is extremely helpful for debugging.
#if _MSC_VER >= 1800
        _TaskCreationCallstack _M_stackTrace;
#else
        void* _M_disassembleMe;
#endif
    };

#ifndef RUNTIMECLASS_Concurrency_winrt_details__AsyncInfoImpl_DEFINED
#define RUNTIMECLASS_Concurrency_winrt_details__AsyncInfoImpl_DEFINED
    extern const __declspec(selectany) WCHAR RuntimeClass_Concurrency_winrt_details__AsyncInfoImpl[] = L"Concurrency_winrt.details._AsyncInfoImpl";
#endif

    /// <summary>
    ///     Base converter class for converting asynchronous interfaces to IAsyncOperation
    /// </summary>
    template<typename _AsyncOperationType, typename _CompletionHandlerType, typename _Result_abi>
    struct _AsyncInfoImpl abstract : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::WinRt>,
        Microsoft::WRL::Implements<Microsoft::WRL::AsyncBase<_CompletionHandlerType>>>
    {
        InspectableClass(RuntimeClass_Concurrency_winrt_details__AsyncInfoImpl, BaseTrust)
    public:
        // The async action, action with progress or operation with progress that this stub forwards to.
#if _MSC_VER >= 1800
        Agile<_AsyncOperationType> _M_asyncInfo;
#else
        Microsoft::WRL::ComPtr<_AsyncOperationType> _M_asyncInfo;
        // The context in which this async info is valid - may be different from the context where the completion handler runs,
        // and may require marshalling before it is used.
        _ContextCallback _M_asyncInfoContext;
#endif

        Microsoft::WRL::ComPtr<_CompletionHandlerType> _M_CompletedHandler;

        _AsyncInfoImpl(_AsyncOperationType* _AsyncInfo) : _M_asyncInfo(_AsyncInfo)
#if _MSC_VER < 1800
            , _M_asyncInfoContext(_ContextCallback::_CaptureCurrent())
#endif
        {}

    public:
        virtual HRESULT OnStart() { return S_OK; }
        virtual void OnCancel() {
            Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncInfo> pAsyncInfo;
            HRESULT hr;
#if _MSC_VER >= 1800
            if (SUCCEEDED(hr = _M_asyncInfo.Get()->QueryInterface<ABI::Windows::Foundation::IAsyncInfo>(pAsyncInfo.GetAddressOf())))
#else
            if (SUCCEEDED(hr = _M_asyncInfo.As(&pAsyncInfo)))
#endif
                pAsyncInfo->Cancel();
            else
                throw std::make_exception_ptr(hr);
        }
        virtual void OnClose() {
            Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncInfo> pAsyncInfo;
            HRESULT hr;
#if _MSC_VER >= 1800
            if (SUCCEEDED(hr = _M_asyncInfo.Get()->QueryInterface<ABI::Windows::Foundation::IAsyncInfo>(pAsyncInfo.GetAddressOf())))
#else
            if (SUCCEEDED(hr = _M_asyncInfo.As(&pAsyncInfo)))
#endif
                pAsyncInfo->Close();
            else
                throw std::make_exception_ptr(hr);
        }

        virtual STDMETHODIMP get_ErrorCode(HRESULT* errorCode)
        {
            Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncInfo> pAsyncInfo;
            HRESULT hr;
#if _MSC_VER >= 1800
            if (SUCCEEDED(hr = _M_asyncInfo.Get()->QueryInterface<ABI::Windows::Foundation::IAsyncInfo>(pAsyncInfo.GetAddressOf())))
#else
            if (SUCCEEDED(hr = _M_asyncInfo.As(&pAsyncInfo)))
#endif
                return pAsyncInfo->get_ErrorCode(errorCode);
            return hr;
        }

        virtual STDMETHODIMP get_Id(UINT* id)
        {
            Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncInfo> pAsyncInfo;
            HRESULT hr;
#if _MSC_VER >= 1800
            if (SUCCEEDED(hr = _M_asyncInfo.Get()->QueryInterface<ABI::Windows::Foundation::IAsyncInfo>(pAsyncInfo.GetAddressOf())))
#else
            if (SUCCEEDED(hr = _M_asyncInfo.As(&pAsyncInfo)))
#endif
                return pAsyncInfo->get_Id(id);
            return hr;
        }

        virtual STDMETHODIMP get_Status(ABI::Windows::Foundation::AsyncStatus *status)
        {
            Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncInfo> pAsyncInfo;
            HRESULT hr;
#if _MSC_VER >= 1800
            if (SUCCEEDED(hr = _M_asyncInfo.Get()->QueryInterface<ABI::Windows::Foundation::IAsyncInfo>(pAsyncInfo.GetAddressOf())))
#else
            if (SUCCEEDED(hr = _M_asyncInfo.As(&pAsyncInfo)))
#endif
                return pAsyncInfo->get_Status(status);
            return hr;
        }

        virtual STDMETHODIMP GetResults(_Result_abi*) { throw std::runtime_error("derived class must implement"); }

        virtual STDMETHODIMP get_Completed(_CompletionHandlerType** handler)
        {
            if (!handler) return E_POINTER;
            _M_CompletedHandler.CopyTo(handler);
            return S_OK;
        }

        virtual    STDMETHODIMP put_Completed(_CompletionHandlerType* value)
        {
            _M_CompletedHandler = value;
            Microsoft::WRL::ComPtr<_CompletionHandlerType> handler = Microsoft::WRL::Callback<_CompletionHandlerType>([&](_AsyncOperationType*, ABI::Windows::Foundation::AsyncStatus status) -> HRESULT {
#if _MSC_VER < 1800
                // Update the saved _M_asyncInfo with a proxy valid in the current context if required. Some Windows APIs return an IAsyncInfo
                // that is only valid for the thread that called the API to retrieve. Since this completion handler can run on any thread, we
                // need to ensure that the async info is valid in the current apartment. _M_asyncInfo will be accessed via calls to 'this' inside
                // _AsyncInit.
                _M_asyncInfo = _ResultContext<_AsyncOperationType*>::_GetValue(_M_asyncInfo.Get(), _M_asyncInfoContext, false);
#endif
                return _M_CompletedHandler->Invoke(_M_asyncInfo.Get(), status);
            });
#if _MSC_VER >= 1800
            return _M_asyncInfo.Get()->put_Completed(handler.Get());
#else
            return _M_asyncInfo->put_Completed(handler.Get());
#endif
        }
    };

    extern const __declspec(selectany) WCHAR RuntimeClass_IAsyncOperationToAsyncOperationConverter[] = L"_IAsyncOperationToAsyncOperationConverter";

    /// <summary>
    ///     Class _IAsyncOperationToAsyncOperationConverter is used to convert an instance of IAsyncOperationWithProgress<T> into IAsyncOperation<T>
    /// </summary>
    template<typename _Result>
    struct _IAsyncOperationToAsyncOperationConverter :
        _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncOperation<_Result>,
        ABI::Windows::Foundation::IAsyncOperationCompletedHandler<_Result>,
        typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperation<_Result>*>()))>::type>
    {
        typedef typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperation<_Result>*>()))>::type _Result_abi;

        InspectableClass(RuntimeClass_IAsyncOperationToAsyncOperationConverter, BaseTrust)
    public:
        _IAsyncOperationToAsyncOperationConverter(ABI::Windows::Foundation::IAsyncOperation<_Result>* _Operation) :
            _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncOperation<_Result>,
            ABI::Windows::Foundation::IAsyncOperationCompletedHandler<_Result>,
            _Result_abi>(_Operation) {}
    public:
        virtual STDMETHODIMP GetResults(_Result_abi* results) override {
            if (!results) return E_POINTER;
#if _MSC_VER >= 1800
            return _M_asyncInfo.Get()->GetResults(results);
#else
            return _M_asyncInfo->GetResults(results);
#endif
        }
    };

    extern const __declspec(selectany) WCHAR RuntimeClass_IAsyncOperationWithProgressToAsyncOperationConverter[] = L"_IAsyncOperationWithProgressToAsyncOperationConverter";

    /// <summary>
    ///     Class _IAsyncOperationWithProgressToAsyncOperationConverter is used to convert an instance of IAsyncOperationWithProgress<T> into IAsyncOperation<T>
    /// </summary>
    template<typename _Result, typename _Progress>
    struct _IAsyncOperationWithProgressToAsyncOperationConverter :
    _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncOperationWithProgress<_Result, _Progress>,
        ABI::Windows::Foundation::IAsyncOperationWithProgressCompletedHandler<_Result, _Progress>,
        typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationWithProgressSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperationWithProgress<_Result, _Progress>*>()))>::type>
    {
        typedef typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationWithProgressSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperationWithProgress<_Result, _Progress>*>()))>::type _Result_abi;

        InspectableClass(RuntimeClass_IAsyncOperationWithProgressToAsyncOperationConverter, BaseTrust)
    public:
        _IAsyncOperationWithProgressToAsyncOperationConverter(ABI::Windows::Foundation::IAsyncOperationWithProgress<_Result, _Progress>* _Operation) :
            _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncOperationWithProgress<_Result, _Progress>,
            ABI::Windows::Foundation::IAsyncOperationWithProgressCompletedHandler<_Result, _Progress>,
            _Result_abi>(_Operation) {}
    public:
        virtual STDMETHODIMP GetResults(_Result_abi* results) override {
            if (!results) return E_POINTER;
#if _MSC_VER >= 1800
            return _M_asyncInfo.Get()->GetResults(results);
#else
            return _M_asyncInfo->GetResults(results);
#endif
        }
    };

    extern const __declspec(selectany) WCHAR RuntimeClass_IAsyncActionToAsyncOperationConverter[] = L"_IAsyncActionToAsyncOperationConverter";

    /// <summary>
    ///     Class _IAsyncActionToAsyncOperationConverter is used to convert an instance of IAsyncAction into IAsyncOperation<_Unit_type>
    /// </summary>
    struct _IAsyncActionToAsyncOperationConverter :
    _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncAction,
        ABI::Windows::Foundation::IAsyncActionCompletedHandler,
        _Unit_type>
    {
        InspectableClass(RuntimeClass_IAsyncActionToAsyncOperationConverter, BaseTrust)
    public:
        _IAsyncActionToAsyncOperationConverter(ABI::Windows::Foundation::IAsyncAction* _Operation) :
            _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncAction,
            ABI::Windows::Foundation::IAsyncActionCompletedHandler,
            _Unit_type>(_Operation) {}

    public:
        virtual STDMETHODIMP GetResults(details::_Unit_type* results)
        {
            if (!results) return E_POINTER;
            // Invoke GetResults on the IAsyncAction to allow exceptions to be thrown to higher layers before returning a dummy value.
#if _MSC_VER >= 1800
            HRESULT hr = _M_asyncInfo.Get()->GetResults();
#else
            HRESULT hr = _M_asyncInfo->GetResults();
#endif
            if (SUCCEEDED(hr)) *results = _Unit_type();
            return hr;
        }
    };

    extern const __declspec(selectany) WCHAR RuntimeClass_IAsyncActionWithProgressToAsyncOperationConverter[] = L"_IAsyncActionWithProgressToAsyncOperationConverter";

    /// <summary>
    ///     Class _IAsyncActionWithProgressToAsyncOperationConverter is used to convert an instance of IAsyncActionWithProgress into IAsyncOperation<_Unit_type>
    /// </summary>
    template<typename _Progress>
    struct _IAsyncActionWithProgressToAsyncOperationConverter :
    _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncActionWithProgress<_Progress>,
        ABI::Windows::Foundation::IAsyncActionWithProgressCompletedHandler<_Progress>,
        _Unit_type>
    {
        InspectableClass(RuntimeClass_IAsyncActionWithProgressToAsyncOperationConverter, BaseTrust)
    public:
        _IAsyncActionWithProgressToAsyncOperationConverter(ABI::Windows::Foundation::IAsyncActionWithProgress<_Progress>* _Action) :
            _AsyncInfoImpl<ABI::Windows::Foundation::IAsyncActionWithProgress<_Progress>,
            ABI::Windows::Foundation::IAsyncActionWithProgressCompletedHandler<_Progress>,
            _Unit_type>(_Action) {}
    public:
        virtual STDMETHODIMP GetResults(_Unit_type* results) override
        {
            if (!results) return E_POINTER;
            // Invoke GetResults on the IAsyncActionWithProgress to allow exceptions to be thrown before returning a dummy value.
#if _MSC_VER >= 1800
            HRESULT hr = _M_asyncInfo.Get()->GetResults();
#else
            HRESULT hr = _M_asyncInfo->GetResults();
#endif
            if (SUCCEEDED(hr)) *results = _Unit_type();
            return hr;
        }
    };
}

/// <summary>
///     The <c>task_continuation_context</c> class allows you to specify where you would like a continuation to be executed.
///     It is only useful to use this class from a Windows Store app. For non-Windows Store apps, the task continuation's
///     execution context is determined by the runtime, and not configurable.
/// </summary>
/// <seealso cref="task Class"/>
/**/
class task_continuation_context : public details::_ContextCallback
{
public:

    /// <summary>
    ///     Creates the default task continuation context.
    /// </summary>
    /// <returns>
    ///     The default continuation context.
    /// </returns>
    /// <remarks>
    ///     The default context is used if you don't specifiy a continuation context when you call the <c>then</c> method. In Windows
    ///     applications for Windows 7 and below, as well as desktop applications on Windows 8 and higher, the runtime determines where
    ///     task continuations will execute. However, in a Windows Store app, the default continuation context for a continuation on an
    ///     apartment aware task is the apartment where <c>then</c> is invoked.
    ///     <para>An apartment aware task is a task that unwraps a Windows Runtime <c>IAsyncInfo</c> interface, or a task that is descended from such
    ///     a task. Therefore, if you schedule a continuation on an apartment aware task in a Windows Runtime STA, the continuation will execute in
    ///     that STA.</para>
    ///     <para>A continuation on a non-apartment aware task will execute in a context the Runtime chooses.</para>
    /// </remarks>
    /**/
    static task_continuation_context use_default()
    {
        // The callback context is created with the context set to CaptureDeferred and resolved when it is used in .then()
        return task_continuation_context(true); // sets it to deferred, is resolved in the constructor of _ContinuationTaskHandle
    }

    /// <summary>
    ///     Creates a task continuation context which allows the Runtime to choose the execution context for a continuation.
    /// </summary>
    /// <returns>
    ///     A task continuation context that represents an arbitrary location.
    /// </returns>
    /// <remarks>
    ///     When this continuation context is used the continuation will execute in a context the runtime chooses even if the antecedent task
    ///     is apartment aware.
    ///     <para><c>use_arbitrary</c> can be used to turn off the default behavior for a continuation on an apartment
    ///     aware task created in an STA. </para>
    ///     <para>This method is only available to Windows Store apps.</para>
    /// </remarks>
    /**/
    static task_continuation_context use_arbitrary()
    {
        task_continuation_context _Arbitrary(true);
        _Arbitrary._Resolve(false);
        return _Arbitrary;
    }

    /// <summary>
    ///     Returns a task continuation context object that represents the current execution context.
    /// </summary>
    /// <returns>
    ///     The current execution context.
    /// </returns>
    /// <remarks>
    ///     This method captures the caller's Windows Runtime context so that continuations can be executed in the right apartment.
    ///     <para>The value returned by <c>use_current</c> can be used to indicate to the Runtime that the continuation should execute in
    ///     the captured context (STA vs MTA) regardless of whether or not the antecedent task is apartment aware. An apartment aware task is
    ///     a task that unwraps a Windows Runtime <c>IAsyncInfo</c> interface, or a task that is descended from such a task. </para>
    ///     <para>This method is only available to Windows Store apps.</para>
    /// </remarks>
    /**/
    static task_continuation_context use_current()
    {
        task_continuation_context _Current(true);
        _Current._Resolve(true);
        return _Current;
    }

private:

    task_continuation_context(bool _DeferCapture = false) : details::_ContextCallback(_DeferCapture)
    {
    }
};

#if _MSC_VER >= 1800
class task_options;
namespace details
{
    struct _Internal_task_options
    {
        bool _M_hasPresetCreationCallstack;
        _TaskCreationCallstack _M_presetCreationCallstack;

        void _set_creation_callstack(const _TaskCreationCallstack &_callstack)
        {
            _M_hasPresetCreationCallstack = true;
            _M_presetCreationCallstack = _callstack;
        }
        _Internal_task_options()
        {
            _M_hasPresetCreationCallstack = false;
        }
    };

    inline _Internal_task_options &_get_internal_task_options(task_options &options);
    inline const _Internal_task_options &_get_internal_task_options(const task_options &options);
}
/// <summary>
///     Represents the allowed options for creating a task
/// </summary>
class task_options
{
public:


    /// <summary>
    ///     Default list of task creation options
    /// </summary>
    task_options()
        : _M_Scheduler(Concurrency::get_ambient_scheduler()),
        _M_CancellationToken(Concurrency::cancellation_token::none()),
        _M_ContinuationContext(task_continuation_context::use_default()),
        _M_HasCancellationToken(false),
        _M_HasScheduler(false)
    {
    }

    /// <summary>
    ///     Task option that specify a cancellation token
    /// </summary>
    task_options(Concurrency::cancellation_token _Token)
        : _M_Scheduler(Concurrency::get_ambient_scheduler()),
        _M_CancellationToken(_Token),
        _M_ContinuationContext(task_continuation_context::use_default()),
        _M_HasCancellationToken(true),
        _M_HasScheduler(false)
    {
    }

    /// <summary>
    ///     Task option that specify a continuation context. This is valid only for continuations (then)
    /// </summary>
    task_options(task_continuation_context _ContinuationContext)
        : _M_Scheduler(Concurrency::get_ambient_scheduler()),
        _M_CancellationToken(Concurrency::cancellation_token::none()),
        _M_ContinuationContext(_ContinuationContext),
        _M_HasCancellationToken(false),
        _M_HasScheduler(false)
    {
    }

    /// <summary>
    ///     Task option that specify a cancellation token and a continuation context. This is valid only for continuations (then)
    /// </summary>
    task_options(Concurrency::cancellation_token _Token, task_continuation_context _ContinuationContext)
        : _M_Scheduler(Concurrency::get_ambient_scheduler()),
        _M_CancellationToken(_Token),
        _M_ContinuationContext(_ContinuationContext),
        _M_HasCancellationToken(false),
        _M_HasScheduler(false)
    {
    }

    /// <summary>
    ///     Task option that specify a scheduler with shared lifetime
    /// </summary>
    template<typename _SchedType>
    task_options(std::shared_ptr<_SchedType> _Scheduler)
        : _M_Scheduler(std::move(_Scheduler)),
        _M_CancellationToken(cancellation_token::none()),
        _M_ContinuationContext(task_continuation_context::use_default()),
        _M_HasCancellationToken(false),
        _M_HasScheduler(true)
    {
    }

    /// <summary>
    ///     Task option that specify a scheduler reference
    /// </summary>
    task_options(Concurrency::scheduler_interface& _Scheduler)
        : _M_Scheduler(&_Scheduler),
        _M_CancellationToken(Concurrency::cancellation_token::none()),
        _M_ContinuationContext(task_continuation_context::use_default()),
        _M_HasCancellationToken(false),
        _M_HasScheduler(true)
    {
    }

    /// <summary>
    ///     Task option that specify a scheduler
    /// </summary>
    task_options(Concurrency::scheduler_ptr _Scheduler)
        : _M_Scheduler(std::move(_Scheduler)),
        _M_CancellationToken(Concurrency::cancellation_token::none()),
        _M_ContinuationContext(task_continuation_context::use_default()),
        _M_HasCancellationToken(false),
        _M_HasScheduler(true)
    {
    }

    /// <summary>
    ///     Task option copy constructor
    /// </summary>
    task_options(const task_options& _TaskOptions)
        : _M_Scheduler(_TaskOptions.get_scheduler()),
        _M_CancellationToken(_TaskOptions.get_cancellation_token()),
        _M_ContinuationContext(_TaskOptions.get_continuation_context()),
        _M_HasCancellationToken(_TaskOptions.has_cancellation_token()),
        _M_HasScheduler(_TaskOptions.has_scheduler())
    {
    }

    /// <summary>
    ///     Sets the given token in the options
    /// </summary>
    void set_cancellation_token(Concurrency::cancellation_token _Token)
    {
        _M_CancellationToken = _Token;
        _M_HasCancellationToken = true;
    }

    /// <summary>
    ///     Sets the given continuation context in the options
    /// </summary>
    void set_continuation_context(task_continuation_context _ContinuationContext)
    {
        _M_ContinuationContext = _ContinuationContext;
    }

    /// <summary>
    ///     Indicates whether a cancellation token was specified by the user
    /// </summary>
    bool has_cancellation_token() const
    {
        return _M_HasCancellationToken;
    }

    /// <summary>
    ///     Returns the cancellation token
    /// </summary>
    Concurrency::cancellation_token get_cancellation_token() const
    {
        return _M_CancellationToken;
    }

    /// <summary>
    ///     Returns the continuation context
    /// </summary>
    task_continuation_context get_continuation_context() const
    {
        return _M_ContinuationContext;
    }

    /// <summary>
    ///     Indicates whether a scheduler n was specified by the user
    /// </summary>
    bool has_scheduler() const
    {
        return _M_HasScheduler;
    }

    /// <summary>
    ///     Returns the scheduler
    /// </summary>
    Concurrency::scheduler_ptr get_scheduler() const
    {
        return _M_Scheduler;
    }

private:

    task_options const& operator=(task_options const& _Right);
    friend details::_Internal_task_options &details::_get_internal_task_options(task_options &);
    friend const details::_Internal_task_options &details::_get_internal_task_options(const task_options &);

    Concurrency::scheduler_ptr _M_Scheduler;
    Concurrency::cancellation_token _M_CancellationToken;
    task_continuation_context _M_ContinuationContext;
    details::_Internal_task_options _M_InternalTaskOptions;
    bool _M_HasCancellationToken;
    bool _M_HasScheduler;
};
#endif

namespace details
{
#if _MSC_VER >= 1800
    inline _Internal_task_options & _get_internal_task_options(task_options &options)
    {
        return options._M_InternalTaskOptions;
    }
    inline const _Internal_task_options & _get_internal_task_options(const task_options &options)
    {
        return options._M_InternalTaskOptions;
    }
#endif
    struct _Task_impl_base;
    template<typename _ReturnType> struct _Task_impl;

    template<typename _ReturnType>
    struct _Task_ptr
    {
        typedef std::shared_ptr<_Task_impl<_ReturnType>> _Type;
#if _MSC_VER >= 1800
        static _Type _Make(Concurrency::details::_CancellationTokenState * _Ct, Concurrency::scheduler_ptr _Scheduler_arg) { return std::make_shared<_Task_impl<_ReturnType>>(_Ct, _Scheduler_arg); }
#else
        static _Type _Make(Concurrency::details::_CancellationTokenState * _Ct) { return std::make_shared<_Task_impl<_ReturnType>>(_Ct); }
#endif
    };
#if _MSC_VER >= 1800
    typedef Concurrency::details::_TaskCollection_t::_TaskProcHandle_t _UnrealizedChore_t;
    typedef _UnrealizedChore_t _UnrealizedChore;
    typedef Concurrency::extensibility::scoped_critical_section_t scoped_lock;
    typedef Concurrency::extensibility::critical_section_t critical_section;
    typedef Concurrency::details::atomic_size_t atomic_size_t;
#else
    typedef Concurrency::details::_UnrealizedChore _UnrealizedChore;
    typedef Concurrency::critical_section::scoped_lock scoped_lock;
    typedef Concurrency::critical_section critical_section;
    typedef volatile size_t atomic_size_t;
#endif
    typedef std::shared_ptr<_Task_impl_base> _Task_ptr_base;
    // The weak-typed base task handler for continuation tasks.
    struct _ContinuationTaskHandleBase : _UnrealizedChore
    {
        _ContinuationTaskHandleBase * _M_next;
        task_continuation_context _M_continuationContext;
        bool _M_isTaskBasedContinuation;

        // This field gives inlining scheduling policy for current chore.
        _TaskInliningMode _M_inliningMode;

        virtual _Task_ptr_base _GetTaskImplBase() const = 0;

        _ContinuationTaskHandleBase() :
            _M_next(nullptr), _M_isTaskBasedContinuation(false), _M_continuationContext(task_continuation_context::use_default()), _M_inliningMode(Concurrency::details::_NoInline)
        {
        }
        virtual ~_ContinuationTaskHandleBase() {}
    };
#if _MSC_VER >= 1800
#if _PPLTASK_ASYNC_LOGGING
    // GUID used for identifying causality logs from PPLTask
    const ::Platform::Guid _PPLTaskCausalityPlatformID(0x7A76B220, 0xA758, 0x4E6E, 0xB0, 0xE0, 0xD7, 0xC6, 0xD7, 0x4A, 0x88, 0xFE);

    __declspec(selectany) volatile long _isCausalitySupported = 0;

    inline bool _IsCausalitySupported()
    {
#if WINAPI_FAMILY_PARTITION(WINAPI_PARTITION_DESKTOP)
        if (_isCausalitySupported == 0)
        {
            long _causality = 1;
            OSVERSIONINFOEX _osvi = {};
            _osvi.dwOSVersionInfoSize = sizeof(OSVERSIONINFOEX);

            // The Causality is supported on Windows version higher than Windows 8
            _osvi.dwMajorVersion = 6;
            _osvi.dwMinorVersion = 3;

            DWORDLONG _conditionMask = 0;
            VER_SET_CONDITION(_conditionMask, VER_MAJORVERSION, VER_GREATER_EQUAL);
            VER_SET_CONDITION(_conditionMask, VER_MINORVERSION, VER_GREATER_EQUAL);

            if (::VerifyVersionInfo(&_osvi, VER_MAJORVERSION | VER_MINORVERSION, _conditionMask))
            {
                _causality = 2;
            }

            _isCausalitySupported = _causality;
            return _causality == 2;
        }

        return _isCausalitySupported == 2 ? true : false;
#else
        return true;
#endif
    }

    // Stateful logger rests inside task_impl_base.
    struct _TaskEventLogger
    {
        _Task_impl_base *_M_task;
        bool _M_scheduled;
        bool _M_taskPostEventStarted;

        // Log before scheduling task
        void _LogScheduleTask(bool _isContinuation)
        {
            if (details::_IsCausalitySupported())
            {
                ::Windows::Foundation::Diagnostics::AsyncCausalityTracer::TraceOperationCreation(::Windows::Foundation::Diagnostics::CausalityTraceLevel::Required, ::Windows::Foundation::Diagnostics::CausalitySource::Library,
                    _PPLTaskCausalityPlatformID, reinterpret_cast<unsigned long long>(_M_task),
                    _isContinuation ? "Concurrency::PPLTask::ScheduleContinuationTask" : "Concurrency::PPLTask::ScheduleTask", 0);
                _M_scheduled = true;
            }
        }

        // It will log the cancel event but not canceled state. _LogTaskCompleted will log the terminal state, which includes cancel state.
        void _LogCancelTask()
        {
            if (details::_IsCausalitySupported())
            {
                ::Windows::Foundation::Diagnostics::AsyncCausalityTracer::TraceOperationRelation(::Windows::Foundation::Diagnostics::CausalityTraceLevel::Important, ::Windows::Foundation::Diagnostics::CausalitySource::Library,
                    _PPLTaskCausalityPlatformID, reinterpret_cast<unsigned long long>(_M_task), ::Windows::Foundation::Diagnostics::CausalityRelation::Cancel);

            }
        }

        // Log when task reaches terminal state. Note: the task can reach a terminal state (by cancellation or exception) without having run
        void _LogTaskCompleted();

        // Log when task body (which includes user lambda and other scheduling code) begin to run
        void _LogTaskExecutionStarted() { }

        // Log when task body finish executing
        void _LogTaskExecutionCompleted()
        {
            if (_M_taskPostEventStarted && details::_IsCausalitySupported())
            {
                ::Windows::Foundation::Diagnostics::AsyncCausalityTracer::TraceSynchronousWorkCompletion(::Windows::Foundation::Diagnostics::CausalityTraceLevel::Required, ::Windows::Foundation::Diagnostics::CausalitySource::Library,
                    ::Windows::Foundation::Diagnostics::CausalitySynchronousWork::CompletionNotification);
            }
        }

        // Log right before user lambda being invoked
        void _LogWorkItemStarted()
        {
            if (details::_IsCausalitySupported())
            {
                ::Windows::Foundation::Diagnostics::AsyncCausalityTracer::TraceSynchronousWorkStart(::Windows::Foundation::Diagnostics::CausalityTraceLevel::Required, ::Windows::Foundation::Diagnostics::CausalitySource::Library,
                    _PPLTaskCausalityPlatformID, reinterpret_cast<unsigned long long>(_M_task), ::Windows::Foundation::Diagnostics::CausalitySynchronousWork::Execution);
            }
        }

        // Log right after user lambda being invoked
        void _LogWorkItemCompleted()
        {
            if (details::_IsCausalitySupported())
            {
                ::Windows::Foundation::Diagnostics::AsyncCausalityTracer::TraceSynchronousWorkCompletion(::Windows::Foundation::Diagnostics::CausalityTraceLevel::Required, ::Windows::Foundation::Diagnostics::CausalitySource::Library,
                    ::Windows::Foundation::Diagnostics::CausalitySynchronousWork::Execution);

                ::Windows::Foundation::Diagnostics::AsyncCausalityTracer::TraceSynchronousWorkStart(::Windows::Foundation::Diagnostics::CausalityTraceLevel::Required, ::Windows::Foundation::Diagnostics::CausalitySource::Library,
                    _PPLTaskCausalityPlatformID, reinterpret_cast<unsigned long long>(_M_task), ::Windows::Foundation::Diagnostics::CausalitySynchronousWork::CompletionNotification);
                _M_taskPostEventStarted = true;
            }
        }

        _TaskEventLogger(_Task_impl_base *_task) : _M_task(_task)
        {
            _M_scheduled = false;
            _M_taskPostEventStarted = false;
        }
    };

    // Exception safe logger for user lambda
    struct _TaskWorkItemRAIILogger
    {
        _TaskEventLogger &_M_logger;
        _TaskWorkItemRAIILogger(_TaskEventLogger &_taskHandleLogger) : _M_logger(_taskHandleLogger)
        {
            _M_logger._LogWorkItemStarted();
        }

        ~_TaskWorkItemRAIILogger()
        {
            _M_logger._LogWorkItemCompleted();
        }
        _TaskWorkItemRAIILogger &operator =(const _TaskWorkItemRAIILogger &); // cannot be assigned
    };

#else
    inline void _LogCancelTask(_Task_impl_base *) {}
    struct _TaskEventLogger
    {
        void _LogScheduleTask(bool) {}
        void _LogCancelTask() {}
        void _LogWorkItemStarted() {}
        void _LogWorkItemCompleted() {}
        void _LogTaskExecutionStarted() {}
        void _LogTaskExecutionCompleted() {}
        void _LogTaskCompleted() {}
        _TaskEventLogger(_Task_impl_base *) {}
    };
    struct _TaskWorkItemRAIILogger
    {
        _TaskWorkItemRAIILogger(_TaskEventLogger &) {}
    };
#endif
#endif
    /// <summary>
    ///     The _PPLTaskHandle is the strong-typed task handle base. All user task functions need to be wrapped in this task handler
    ///     to be executable by PPL. By deriving from a different _BaseTaskHandle, it can be used for both initial tasks and continuation tasks.
    ///     For initial tasks, _PPLTaskHandle will be derived from _UnrealizedChore, and for continuation tasks, it will be derived from
    ///     _ContinuationTaskHandleBase. The life time of the _PPLTaskHandle object is be managed by runtime if task handle is scheduled.
    /// </summary>
    /// <typeparam name="_ReturnType">
    ///     The result type of the _Task_impl.
    /// </typeparam>
    /// <typeparam name="_DerivedTaskHandle">
    ///     The derived task handle class. The <c>operator ()</c> needs to be implemented.
    /// </typeparam>
    /// <typeparam name="_BaseTaskHandle">
    ///     The base class from which _PPLTaskHandle should be derived. This is either _UnrealizedChore or _ContinuationTaskHandleBase.
    /// </typeparam>
    template<typename _ReturnType, typename _DerivedTaskHandle, typename _BaseTaskHandle>
    struct _PPLTaskHandle : _BaseTaskHandle
    {
        _PPLTaskHandle(const typename _Task_ptr<_ReturnType>::_Type & _PTask) : _M_pTask(_PTask)
        {
#if _MSC_VER < 1800
            m_pFunction = reinterpret_cast <Concurrency::TaskProc> (&_UnrealizedChore::_InvokeBridge<_PPLTaskHandle>);
            _SetRuntimeOwnsLifetime(true);
#endif
        }
        virtual ~_PPLTaskHandle() {
#if _MSC_VER >= 1800
            // Here is the sink of all task completion code paths
            _M_pTask->_M_taskEventLogger._LogTaskCompleted();
#endif
        }
#if _MSC_VER >= 1800
        virtual void invoke() const
#else
        void operator()() const
#endif
        {
            // All exceptions should be rethrown to finish cleanup of the task collection. They will be caught and handled
            // by the runtime.
            _CONCRT_ASSERT(_M_pTask != nullptr);
            if (!_M_pTask->_TransitionedToStarted()) {
#if _MSC_VER >= 1800
                static_cast<const _DerivedTaskHandle *>(this)->_SyncCancelAndPropagateException();
#endif
                return;
            }
#if _MSC_VER >= 1800
            _M_pTask->_M_taskEventLogger._LogTaskExecutionStarted();
#endif
            try
            {
                // All derived task handle must implement this contract function.
                static_cast<const _DerivedTaskHandle *>(this)->_Perform();
            }
            catch (const Concurrency::task_canceled &)
            {
                _M_pTask->_Cancel(true);
#if _MSC_VER < 1800
                throw;
#endif
            }
            catch (const Concurrency::details::_Interruption_exception &)
            {
                _M_pTask->_Cancel(true);
#if _MSC_VER < 1800
                throw;
#endif
            }
            catch (IRestrictedErrorInfo*& _E)
            {
                _M_pTask->_CancelWithException(_E);
#if _MSC_VER < 1800
                throw;
#endif
            }
            catch (...)
            {
                _M_pTask->_CancelWithException(std::current_exception());
#if _MSC_VER < 1800
                throw;
#endif
            }
#if _MSC_VER >= 1800
            _M_pTask->_M_taskEventLogger._LogTaskExecutionCompleted();
#endif
        }

        // Cast _M_pTask pointer to "type-less" _Task_impl_base pointer, which can be used in _ContinuationTaskHandleBase.
        // The return value should be automatically optimized by R-value ref.
        _Task_ptr_base _GetTaskImplBase() const
        {
            return _M_pTask;
        }

        typename _Task_ptr<_ReturnType>::_Type _M_pTask;

    private:
        _PPLTaskHandle const & operator=(_PPLTaskHandle const&);    // no assignment operator
    };

    /// <summary>
    ///     The base implementation of a first-class task. This class contains all the non-type specific
    ///     implementation details of the task.
    /// </summary>
    /**/
    struct _Task_impl_base
    {
        enum _TaskInternalState
        {
            // Tracks the state of the task, rather than the task collection on which the task is scheduled
            _Created,
            _Started,
            _PendingCancel,
            _Completed,
            _Canceled
        };
#if _MSC_VER >= 1800
        _Task_impl_base(Concurrency::details::_CancellationTokenState * _PTokenState, Concurrency::scheduler_ptr _Scheduler_arg)
            : _M_TaskState(_Created),
            _M_fFromAsync(false), _M_fUnwrappedTask(false),
            _M_pRegistration(nullptr), _M_Continuations(nullptr), _M_TaskCollection(_Scheduler_arg),
            _M_taskEventLogger(this)
#else
        _Task_impl_base(Concurrency::details::_CancellationTokenState * _PTokenState) : _M_TaskState(_Created),
            _M_fFromAsync(false), _M_fRuntimeAggregate(false), _M_fUnwrappedTask(false),
            _M_pRegistration(nullptr), _M_Continuations(nullptr), _M_pTaskCollection(nullptr),
            _M_pTaskCreationAddressHint(nullptr)
#endif
        {
            // Set cancelation token
            _M_pTokenState = _PTokenState;
            _CONCRT_ASSERT(_M_pTokenState != nullptr);
            if (_M_pTokenState != Concurrency::details::_CancellationTokenState::_None())
                _M_pTokenState->_Reference();

        }

        virtual ~_Task_impl_base()
        {
            _CONCRT_ASSERT(_M_pTokenState != nullptr);
            if (_M_pTokenState != Concurrency::details::_CancellationTokenState::_None())
            {
                _M_pTokenState->_Release();
            }
#if _MSC_VER < 1800
            if (_M_pTaskCollection != nullptr)
            {
                _M_pTaskCollection->_Release();
                _M_pTaskCollection = nullptr;
            }
#endif
        }

        task_status _Wait()
        {
            bool _DoWait = true;

            if (_IsNonBlockingThread())
            {
                // In order to prevent Windows Runtime STA threads from blocking the UI, calling task.wait() task.get() is illegal
                // if task has not been completed.
                if (!_IsCompleted() && !_IsCanceled())
                {
                    throw Concurrency::invalid_operation("Illegal to wait on a task in a Windows Runtime STA");
                }
                else
                {
                    // Task Continuations are 'scheduled' *inside* the chore that is executing on the ancestors's task group. If a continuation
                    // needs to be marshalled to a different apartment, instead of scheduling, we make a synchronous cross apartment COM
                    // call to execute the continuation. If it then happens to do something which waits on the ancestor (say it calls .get(), which
                    // task based continuations are wont to do), waiting on the task group results in on the chore that is making this
                    // synchronous callback, which causes a deadlock. To avoid this, we test the state ancestor's event , and we will NOT wait on
                    // if it has finished execution (which means now we are on the inline synchronous callback).
                    _DoWait = false;
                }
            }
            if (_DoWait)
            {
#if _MSC_VER < 1800
                // Wait for the task to be actually scheduled, otherwise the underlying task collection
                // might not be created yet. If we don't wait, we will miss the chance to inline this task.
                _M_Scheduled.wait();


                // A PPL task created by a task_completion_event does not have an underlying TaskCollection. For
                // These tasks, a call to wait should wait for the event to be set. The TaskCollection must either
                // be nullptr or allocated (the setting of _M_Scheduled) ensures that.
#endif
                // If this task was created from a Windows Runtime async operation, do not attempt to inline it. The
                // async operation will take place on a thread in the appropriate apartment Simply wait for the completed
                // event to be set.
#if _MSC_VER >= 1800
                if (_M_fFromAsync)
#else
                if ((_M_pTaskCollection == nullptr) || _M_fFromAsync)
#endif
                {
#if _MSC_VER >= 1800
                    _M_TaskCollection._Wait();
#else
                    _M_Completed.wait();
#endif
                }
                else
                {
                    // Wait on the task collection to complete. The task collection is guaranteed to still be
                    // valid since the task must be still within scope so that the _Task_impl_base destructor
                    // has not yet been called. This call to _Wait potentially inlines execution of work.
                    try
                    {
                        // Invoking wait on a task collection resets the state of the task collection. This means that
                        // if the task collection itself were canceled, or had encountered an exception, only the first
                        // call to wait will receive this status. However, both cancellation and exceptions flowing through
                        // tasks set state in the task impl itself.

                        // When it returns cancelled, either work chore or the cancel thread should already have set task's state
                        // properly -- cancelled state or completed state (because there was no interruption point).
                        // For tasks with unwrapped tasks, we should not change the state of current task, since the unwrapped task are still running.
#if _MSC_VER >= 1800
                        _M_TaskCollection._RunAndWait();
#else
                        _M_pTaskCollection->_RunAndWait();
#endif
                    }
                    catch (Concurrency::details::_Interruption_exception&)
                    {
                        // The _TaskCollection will never be an interruption point since it has a none token.
                        _CONCRT_ASSERT(false);
                    }
                    catch (Concurrency::task_canceled&)
                    {
                        // task_canceled is a special exception thrown by cancel_current_task. The spec states that cancel_current_task
                        // must be called from code that is executed within the task (throwing it from parallel work created by and waited
                        // upon by the task is acceptable). We can safely assume that the task wrapper _PPLTaskHandle::operator() has seen
                        // the exception and canceled the task. Swallow the exception here.
                        _CONCRT_ASSERT(_IsCanceled());
                    }
                    catch (IRestrictedErrorInfo*& _E)
                    {
                        // Its possible the task body hasn't seen the exception, if so we need to cancel with exception here.
                        if(!_HasUserException())
                        {
                            _CancelWithException(_E);
                        }
                        // Rethrow will mark the exception as observed.
                        _M_exceptionHolder->_RethrowUserException();
                    }
                    catch (...)
                    {
                        // Its possible the task body hasn't seen the exception, if so we need to cancel with exception here.
                        if (!_HasUserException())
                        {
                            _CancelWithException(std::current_exception());
                        }
                        // Rethrow will mark the exception as observed.
                        _M_exceptionHolder->_RethrowUserException();
                    }

                    // If the lambda body for this task (executed or waited upon in _RunAndWait above) happened to return a task
                    // which is to be unwrapped and plumbed to the output of this task, we must not only wait on the lambda body, we must
                    // wait on the **INNER** body. It is in theory possible that we could inline such if we plumb a series of things through;
                    // however, this takes the tact of simply waiting upon the completion signal.
                    if (_M_fUnwrappedTask)
                    {
#if _MSC_VER >= 1800
                        _M_TaskCollection._Wait();
#else
                        _M_Completed.wait();
#endif
                    }
                }
            }

            if (_HasUserException())
            {
                _M_exceptionHolder->_RethrowUserException();
            }
            else if (_IsCanceled())
            {
                return Concurrency::canceled;
            }
            _CONCRT_ASSERT(_IsCompleted());
            return Concurrency::completed;
        }
        /// <summary>
        ///     Requests cancellation on the task and schedules continuations if the task can be transitioned to a terminal state.
        /// </summary>
        /// <param name="_SynchronousCancel">
        ///     Set to true if the cancel takes place as a result of the task body encountering an exception, or because an ancestor or task_completion_event the task
        ///     was registered with were canceled with an exception. A synchronous cancel is one that assures the task could not be running on a different thread at
        ///     the time the cancellation is in progress. An asynchronous cancel is one where the thread performing the cancel has no control over the thread that could
        ///     be executing the task, that is the task could execute concurrently while the cancellation is in progress.
        /// </param>
        /// <param name="_UserException">
        ///     Whether an exception other than the internal runtime cancellation exceptions caused this cancellation.
        /// </param>
        /// <param name="_PropagatedFromAncestor">
        ///     Whether this exception came from an ancestor task or a task_completion_event as opposed to an exception that was encountered by the task itself. Only valid when
        ///     _UserException is set to true.
        /// </param>
        /// <param name="_ExHolder">
        ///     The exception holder that represents the exception. Only valid when _UserException is set to true.
        /// </param>
        virtual bool _CancelAndRunContinuations(bool _SynchronousCancel, bool _UserException, bool _PropagatedFromAncestor, const std::shared_ptr<_ExceptionHolder>& _ExHolder) = 0;

        bool _Cancel(bool _SynchronousCancel)
        {
            // Send in a dummy value for exception. It is not used when the first parameter is false.
            return _CancelAndRunContinuations(_SynchronousCancel, false, false, _M_exceptionHolder);
        }

        bool _CancelWithExceptionHolder(const std::shared_ptr<_ExceptionHolder>& _ExHolder, bool _PropagatedFromAncestor)
        {
            // This task was canceled because an ancestor task encountered an exception.
            return _CancelAndRunContinuations(true, true, _PropagatedFromAncestor, _ExHolder);
        }

        bool _CancelWithException(IRestrictedErrorInfo*& _Exception)
        {
            // This task was canceled because the task body encountered an exception.
            _CONCRT_ASSERT(!_HasUserException());
#if _MSC_VER >= 1800
            return _CancelAndRunContinuations(true, true, false, std::make_shared<_ExceptionHolder>(_Exception, _GetTaskCreationCallstack()));
#else
            return _CancelAndRunContinuations(true, true, false, std::make_shared<_ExceptionHolder>(_Exception, _GetTaskCreationAddressHint()));
#endif
        }
        bool _CancelWithException(const std::exception_ptr& _Exception)
        {
            // This task was canceled because the task body encountered an exception.
            _CONCRT_ASSERT(!_HasUserException());
#if _MSC_VER >= 1800
            return _CancelAndRunContinuations(true, true, false, std::make_shared<_ExceptionHolder>(_Exception, _GetTaskCreationCallstack()));
#else
            return _CancelAndRunContinuations(true, true, false, std::make_shared<_ExceptionHolder>(_Exception, _GetTaskCreationAddressHint()));
#endif
        }

#if _MSC_VER >= 1800
        void _RegisterCancellation(std::weak_ptr<_Task_impl_base> _WeakPtr)
#else
        void _RegisterCancellation()
#endif
        {
            _CONCRT_ASSERT(Concurrency::details::_CancellationTokenState::_IsValid(_M_pTokenState));
#if _MSC_VER >= 1800
            auto _CancellationCallback = [_WeakPtr](){
                // Taking ownership of the task prevents dead lock during destruction
                // if the destructor waits for the cancellations to be finished
                auto _task = _WeakPtr.lock();
                if (_task != nullptr)
                    _task->_Cancel(false);
            };

            _M_pRegistration = new Concurrency::details::_CancellationTokenCallback<decltype(_CancellationCallback)>(_CancellationCallback);
            _M_pTokenState->_RegisterCallback(_M_pRegistration);
#else
            _M_pRegistration = _M_pTokenState->_RegisterCallback(reinterpret_cast<Concurrency::TaskProc>(&_CancelViaToken), (_Task_impl_base *)this);
#endif
        }

        void _DeregisterCancellation()
        {
            if (_M_pRegistration != nullptr)
            {
                _M_pTokenState->_DeregisterCallback(_M_pRegistration);
                _M_pRegistration->_Release();
                _M_pRegistration = nullptr;
            }
        }
#if _MSC_VER < 1800
        static void _CancelViaToken(_Task_impl_base *_PImpl)
        {
            _PImpl->_Cancel(false);
        }
#endif
        bool _IsCreated()
        {
            return (_M_TaskState == _Created);
        }

        bool _IsStarted()
        {
            return (_M_TaskState == _Started);
        }

        bool _IsPendingCancel()
        {
            return (_M_TaskState == _PendingCancel);
        }

        bool _IsCompleted()
        {
            return (_M_TaskState == _Completed);
        }

        bool _IsCanceled()
        {
            return (_M_TaskState == _Canceled);
        }

        bool _HasUserException()
        {
            return static_cast<bool>(_M_exceptionHolder);
        }
#if _MSC_VER < 1800
        void _SetScheduledEvent()
        {
            _M_Scheduled.set();
        }
#endif
        const std::shared_ptr<_ExceptionHolder>& _GetExceptionHolder()
        {
            _CONCRT_ASSERT(_HasUserException());
            return _M_exceptionHolder;
        }

        bool _IsApartmentAware()
        {
            return _M_fFromAsync;
        }

        void _SetAsync(bool _Async = true)
        {
            _M_fFromAsync = _Async;
        }
#if _MSC_VER >= 1800
        _TaskCreationCallstack _GetTaskCreationCallstack()
        {
            return _M_pTaskCreationCallstack;
        }

        void _SetTaskCreationCallstack(const _TaskCreationCallstack &_Callstack)
        {
            _M_pTaskCreationCallstack = _Callstack;
        }
#else
        void* _GetTaskCreationAddressHint()
        {
            return _M_pTaskCreationAddressHint;
        }

        void _SetTaskCreationAddressHint(void* _AddressHint)
        {
            _M_pTaskCreationAddressHint = _AddressHint;
        }
#endif
        /// <summary>
        ///     Helper function to schedule the task on the Task Collection.
        /// </summary>
        /// <param name="_PTaskHandle">
        ///     The task chore handle that need to be executed.
        /// </param>
        /// <param name="_InliningMode">
        ///     The inlining scheduling policy for current _PTaskHandle.
        /// </param>
        void _ScheduleTask(_UnrealizedChore * _PTaskHandle, _TaskInliningMode _InliningMode)
        {
#if _MSC_VER < 1800
            // Construct the task collection; We use none token to provent it becoming interruption point.
            _M_pTaskCollection = Concurrency::details::_AsyncTaskCollection::_NewCollection(Concurrency::details::_CancellationTokenState::_None());
            // _M_pTaskCollection->_ScheduleWithAutoInline will schedule the chore onto AsyncTaskCollection with automatic inlining, in a way that honors cancellation etc.
#endif
            try
            {
#if _MSC_VER >= 1800
                _M_TaskCollection._ScheduleTask(_PTaskHandle, _InliningMode);
#else
                // Do not need to check its returning state, more details please refer to _Wait method.
                _M_pTaskCollection->_ScheduleWithAutoInline(_PTaskHandle, _InliningMode);
#endif
            }
            catch (const Concurrency::task_canceled &)
            {
                // task_canceled is a special exception thrown by cancel_current_task. The spec states that cancel_current_task
                // must be called from code that is executed within the task (throwing it from parallel work created by and waited
                // upon by the task is acceptable). We can safely assume that the task wrapper _PPLTaskHandle::operator() has seen
                // the exception and canceled the task. Swallow the exception here.
                _CONCRT_ASSERT(_IsCanceled());
            }
            catch (const Concurrency::details::_Interruption_exception &)
            {
                // The _TaskCollection will never be an interruption point since it has a none token.
                _CONCRT_ASSERT(false);
            }
            catch (...)
            {
                // This exception could only have come from within the chore body. It should've been caught
                // and the task should be canceled with exception. Swallow the exception here.
                _CONCRT_ASSERT(_HasUserException());
            }
#if _MSC_VER < 1800
            // Set the event in case anyone is waiting to notify that this task has been scheduled. In the case where we
            // execute the chore inline, the event should be set after the chore has executed, to prevent a different thread
            // performing a wait on the task from waiting on the task collection before the chore is actually added to it,
            // and thereby returning from the wait() before the chore has executed.
            _SetScheduledEvent();
#endif
        }

        /// <summary>
        ///     Function executes a continuation. This function is recorded by a parent task implementation
        ///     when a continuation is created in order to execute later.
        /// </summary>
        /// <param name="_PTaskHandle">
        ///     The continuation task chore handle that need to be executed.
        /// </param>
        /**/
        void _RunContinuation(_ContinuationTaskHandleBase * _PTaskHandle)
        {
            _Task_ptr_base _ImplBase = _PTaskHandle->_GetTaskImplBase();
            if (_IsCanceled() && !_PTaskHandle->_M_isTaskBasedContinuation)
            {
                if (_HasUserException())
                {
                    // If the ancestor encountered an exception, transfer the exception to the continuation
                    // This traverses down the tree to propagate the exception.
                    _ImplBase->_CancelWithExceptionHolder(_GetExceptionHolder(), true);
                }
                else
                {
                    // If the ancestor was canceled, then your own execution should be canceled.
                    // This traverses down the tree to cancel it.
                    _ImplBase->_Cancel(true);
                }
            }
            else
            {
                // This can only run when the ancestor has completed or it's a task based continuation that fires when a task is canceled
                // (with or without a user exception).
                _CONCRT_ASSERT(_IsCompleted() || _PTaskHandle->_M_isTaskBasedContinuation);

#if _MSC_VER >= 1800
                _CONCRT_ASSERT(!_ImplBase->_IsCanceled());
                return _ImplBase->_ScheduleContinuationTask(_PTaskHandle);
#else
                // If it has been canceled here (before starting), do nothing. The guy firing cancel will do the clean up.
                if (!_ImplBase->_IsCanceled())
                {
                    return _ImplBase->_ScheduleContinuationTask(_PTaskHandle);
                }
#endif
            }

            // If the handle is not scheduled, we need to manually delete it.
            delete _PTaskHandle;
        }

        // Schedule a continuation to run
        void _ScheduleContinuationTask(_ContinuationTaskHandleBase * _PTaskHandle)
        {
#if _MSC_VER >= 1800
            _M_taskEventLogger._LogScheduleTask(true);
#endif
            // Ensure that the continuation runs in proper context (this might be on a Concurrency Runtime thread or in a different Windows Runtime apartment)
            if (_PTaskHandle->_M_continuationContext._HasCapturedContext())
            {
                // For those continuations need to be scheduled inside captured context, we will try to apply automatic inlining to their inline modes,
                // if they haven't been specified as _ForceInline yet. This change will encourage those continuations to be executed inline so that reduce
                // the cost of marshaling.
                // For normal continuations we won't do any change here, and their inline policies are completely decided by ._ThenImpl method.
                if (_PTaskHandle->_M_inliningMode != Concurrency::details::_ForceInline)
                {
                    _PTaskHandle->_M_inliningMode = Concurrency::details::_DefaultAutoInline;
                }
                details::_ScheduleFuncWithAutoInline([_PTaskHandle]() -> HRESULT {
                    // Note that we cannot directly capture "this" pointer, instead, we should use _TaskImplPtr, a shared_ptr to the _Task_impl_base.
                    // Because "this" pointer will be invalid as soon as _PTaskHandle get deleted. _PTaskHandle will be deleted after being scheduled.
                    auto _TaskImplPtr = _PTaskHandle->_GetTaskImplBase();
                    if (details::_ContextCallback::_CaptureCurrent() == _PTaskHandle->_M_continuationContext)
                    {
                        _TaskImplPtr->_ScheduleTask(_PTaskHandle, Concurrency::details::_ForceInline);
                    }
                    else
                    {
                        //
                        // It's entirely possible that the attempt to marshal the call into a differing context will fail. In this case, we need to handle
                        // the exception and mark the continuation as canceled with the appropriate exception. There is one slight hitch to this:
                        //
                        // NOTE: COM's legacy behavior is to swallow SEH exceptions and marshal them back as HRESULTS. This will in effect turn an SEH into
                        // a C++ exception that gets tagged on the task. One unfortunate result of this is that various pieces of the task infrastructure will
                        // not be in a valid state after this in /EHsc (due to the lack of destructors running, etc...).
                        //
                        try
                        {
                            // Dev10 compiler needs this!
                            auto _PTaskHandle1 = _PTaskHandle;
                            _PTaskHandle->_M_continuationContext._CallInContext([_PTaskHandle1, _TaskImplPtr]() -> HRESULT {
                                _TaskImplPtr->_ScheduleTask(_PTaskHandle1, Concurrency::details::_ForceInline);
                                return S_OK;
                            });
                        }
                        catch (IRestrictedErrorInfo*& _E)
                        {
                            _TaskImplPtr->_CancelWithException(_E);
                        }
                        catch (...)
                        {
                            _TaskImplPtr->_CancelWithException(std::current_exception());
                        }
                    }
                    return S_OK;
                }, _PTaskHandle->_M_inliningMode);
            }
            else
            {
                _ScheduleTask(_PTaskHandle, _PTaskHandle->_M_inliningMode);
            }
        }

        /// <summary>
        ///     Schedule the actual continuation. This will either schedule the function on the continuation task's implementation
        ///     if the task has completed or append it to a list of functions to execute when the task actually does complete.
        /// </summary>
        /// <typeparam name="_FuncInputType">
        ///     The input type of the task.
        /// </typeparam>
        /// <typeparam name="_FuncOutputType">
        ///     The output type of the task.
        /// </typeparam>
        /**/
        void _ScheduleContinuation(_ContinuationTaskHandleBase * _PTaskHandle)
        {
            enum { _Nothing, _Schedule, _Cancel, _CancelWithException } _Do = _Nothing;

            // If the task has canceled, cancel the continuation. If the task has completed, execute the continuation right away.
            // Otherwise, add it to the list of pending continuations
            {
                scoped_lock _LockHolder(_M_ContinuationsCritSec);
                if (_IsCompleted() || (_IsCanceled() && _PTaskHandle->_M_isTaskBasedContinuation))
                {
                    _Do = _Schedule;
                }
                else if (_IsCanceled())
                {
                    if (_HasUserException())
                    {
                        _Do = _CancelWithException;
                    }
                    else
                    {
                        _Do = _Cancel;
                    }
                }
                else
                {
                    // chain itself on the continuation chain.
                    _PTaskHandle->_M_next = _M_Continuations;
                    _M_Continuations = _PTaskHandle;
                }
            }

            // Cancellation and execution of continuations should be performed after releasing the lock. Continuations off of
            // async tasks may execute inline.
            switch (_Do)
            {
            case _Schedule:
            {
                              _PTaskHandle->_GetTaskImplBase()->_ScheduleContinuationTask(_PTaskHandle);
                              break;
            }
            case _Cancel:
            {
                            // If the ancestor was canceled, then your own execution should be canceled.
                            // This traverses down the tree to cancel it.
                            _PTaskHandle->_GetTaskImplBase()->_Cancel(true);

                            delete _PTaskHandle;
                            break;
            }
            case _CancelWithException:
            {
                                         // If the ancestor encountered an exception, transfer the exception to the continuation
                                         // This traverses down the tree to propagate the exception.
                                         _PTaskHandle->_GetTaskImplBase()->_CancelWithExceptionHolder(_GetExceptionHolder(), true);

                                         delete _PTaskHandle;
                                         break;
            }
            case _Nothing:
            default:
                // In this case, we have inserted continuation to continuation chain,
                // nothing more need to be done, just leave.
                break;
            }
        }

        void _RunTaskContinuations()
        {
            // The link list can no longer be modified at this point,
            // since all following up continuations will be scheduled by themselves.
            _ContinuationList _Cur = _M_Continuations, _Next;
            _M_Continuations = nullptr;
            while (_Cur)
            {
                // Current node might be deleted after running,
                // so we must fetch the next first.
                _Next = _Cur->_M_next;
                _RunContinuation(_Cur);
                _Cur = _Next;
            }
        }
        static bool  _IsNonBlockingThread()
        {
            APTTYPE _AptType;
            APTTYPEQUALIFIER _AptTypeQualifier;

            HRESULT hr = CoGetApartmentType(&_AptType, &_AptTypeQualifier);
            //
            // If it failed, it's not a Windows Runtime/COM initialized thread. This is not a failure.
            //
            if (SUCCEEDED(hr))
            {
                switch (_AptType)
                {
                case APTTYPE_STA:
                case APTTYPE_MAINSTA:
                    return true;
                    break;
                case APTTYPE_NA:
                    switch (_AptTypeQualifier)
                    {
                        // A thread executing in a neutral apartment is either STA or MTA. To find out if this thread is allowed
                        // to wait, we check the app qualifier. If it is an STA thread executing in a neutral apartment, waiting
                        // is illegal, because the thread is responsible for pumping messages and waiting on a task could take the
                        // thread out of circulation for a while.
                    case APTTYPEQUALIFIER_NA_ON_STA:
                    case APTTYPEQUALIFIER_NA_ON_MAINSTA:
                        return true;
                        break;
                    }
                    break;
                }
            }
#if _UITHREADCTXT_SUPPORT
            // This method is used to throw an exepection in _Wait() if called within STA.  We
            // want the same behavior if _Wait is called on the UI thread.
            if (SUCCEEDED(CaptureUiThreadContext(nullptr)))
            {
                return true;
            }
#endif // _UITHREADCTXT_SUPPORT

            return false;
        }

        template<typename _ReturnType, typename _Result, typename _OpType, typename _CompHandlerType, typename _ResultType>
        static void _AsyncInit(const typename _Task_ptr<_ReturnType>::_Type & _OuterTask,
            _AsyncInfoImpl<_OpType, _CompHandlerType, _ResultType>* _AsyncOp)
        {
            typedef typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_GetUnwrappedType(stdx::declval<_OpType*>()))>::type _Result_abi;
            // This method is invoked either when a task is created from an existing async operation or
            // when a lambda that creates an async operation executes.

            // If the outer task is pending cancel, cancel the async operation before setting the completed handler. The COM reference on
            // the IAsyncInfo object will be released when all *references to the operation go out of scope.

            // This assertion uses the existence of taskcollection to determine if the task was created from an event.
            // That is no longer valid as even tasks created from a user lambda could have no underlying taskcollection
            // when a custom scheduler is used.
#if _MSC_VER < 1800
            _CONCRT_ASSERT(((_OuterTask->_M_pTaskCollection == nullptr) || _OuterTask->_M_fUnwrappedTask) && !_OuterTask->_IsCanceled());
#endif

            // Pass the shared_ptr by value into the lambda instead of using 'this'.

            _AsyncOp->put_Completed(Microsoft::WRL::Callback<_CompHandlerType>(
                [_OuterTask, _AsyncOp](_OpType* _Operation, ABI::Windows::Foundation::AsyncStatus _Status) mutable -> HRESULT
            {
                HRESULT hr = S_OK;
                if (_Status == ABI::Windows::Foundation::AsyncStatus::Canceled)
                {
                    _OuterTask->_Cancel(true);
                }
                else if (_Status == ABI::Windows::Foundation::AsyncStatus::Error)
                {
                    HRESULT _hr;
                    Microsoft::WRL::ComPtr<ABI::Windows::Foundation::IAsyncInfo> pAsyncInfo;
                    if (SUCCEEDED(hr = _Operation->QueryInterface<ABI::Windows::Foundation::IAsyncInfo>(pAsyncInfo.GetAddressOf())) && SUCCEEDED(hr = pAsyncInfo->get_ErrorCode(&_hr)))
                        _OuterTask->_CancelWithException(std::make_exception_ptr(_hr));
                }
                else
                {
                    _CONCRT_ASSERT(_Status == ABI::Windows::Foundation::AsyncStatus::Completed);
                    _NormalizeVoidToUnitType<_Result_abi>::_Type results;
                    if (SUCCEEDED(hr = _AsyncOp->GetResults(&results)))
                        _OuterTask->_FinalizeAndRunContinuations(results);
                }
                // Take away this shared pointers reference on the task instead of waiting for the delegate to be released. It could
                // be released on a different thread after a delay, and not releasing the reference here could cause the tasks to hold
                // on to resources longer than they should. As an example, without this reset, writing to a file followed by reading from
                // it using the Windows Runtime Async APIs causes a sharing violation.
                // Using const_cast is the workaround for failed mutable keywords
                const_cast<_Task_ptr<_ReturnType>::_Type &>(_OuterTask).reset();
                return hr;
            }).Get());
            _OuterTask->_SetUnwrappedAsyncOp(_AsyncOp);
        }
        template<typename _ReturnType, typename _InternalReturnType>
        static void _AsyncInit(const typename _Task_ptr<_ReturnType>::_Type& _OuterTask, const task<_InternalReturnType> & _UnwrappedTask)
        {
            _CONCRT_ASSERT(_OuterTask->_M_fUnwrappedTask && !_OuterTask->_IsCanceled());
            //
            // We must ensure that continuations off _OuterTask (especially exception handling ones) continue to function in the
            // presence of an exception flowing out of the inner task _UnwrappedTask. This requires an exception handling continuation
            // off the inner task which does the appropriate funnelling to the outer one. We use _Then instead of then to prevent
            // the exception from being marked as observed by our internal continuation. This continuation must be scheduled regardless
            // of whether or not the _OuterTask task is canceled.
            //
            _UnwrappedTask._Then([_OuterTask](task<_InternalReturnType> _AncestorTask) -> HRESULT {

                if (_AncestorTask._GetImpl()->_IsCompleted())
                {
                    _OuterTask->_FinalizeAndRunContinuations(_AncestorTask._GetImpl()->_GetResult());
                }
                else
                {
                    _CONCRT_ASSERT(_AncestorTask._GetImpl()->_IsCanceled());
                    if (_AncestorTask._GetImpl()->_HasUserException())
                    {
                        // Set _PropagatedFromAncestor to false, since _AncestorTask is not an ancestor of _UnwrappedTask.
                        // Instead, it is the enclosing task.
                        _OuterTask->_CancelWithExceptionHolder(_AncestorTask._GetImpl()->_GetExceptionHolder(), false);
                    }
                    else
                    {
                        _OuterTask->_Cancel(true);
                    }
                }
                return S_OK;
#if _MSC_VER >= 1800
            }, nullptr, Concurrency::details::_DefaultAutoInline);
#else
            }, nullptr, false, Concurrency::details::_DefaultAutoInline);
#endif
        }

#if _MSC_VER >= 1800
        Concurrency::scheduler_ptr _GetScheduler() const
        {
            return _M_TaskCollection._GetScheduler();
        }
#else
        Concurrency::event _M_Completed;
        Concurrency::event _M_Scheduled;
#endif

        // Tracks the internal state of the task
        volatile _TaskInternalState _M_TaskState;
        // Set to true either if the ancestor task had the flag set to true, or if the lambda that does the work of this task returns an
        // async operation or async action that is unwrapped by the runtime.
        bool _M_fFromAsync;
#if _MSC_VER < 1800
        // Set to true if we need to marshal the inner parts of an aggregate type like std::vector<T^> or std::pair<T^, size_t>. We only marshal
        // the contained T^s if we create the vector or pair, such as on a when_any or a when_all operation.
        bool _M_fRuntimeAggregate;
#endif
        // Set to true when a continuation unwraps a task or async operation.
        bool _M_fUnwrappedTask;

        // An exception thrown by the task body is captured in an exception holder and it is shared with all value based continuations rooted at the task.
        // The exception is 'observed' if the user invokes get()/wait() on any of the tasks that are sharing this exception holder. If the exception
        // is not observed by the time the internal object owned by the shared pointer destructs, the process will fail fast.
        std::shared_ptr<_ExceptionHolder> _M_exceptionHolder;

        typedef _ContinuationTaskHandleBase * _ContinuationList;

        critical_section _M_ContinuationsCritSec;
        _ContinuationList _M_Continuations;

        // The cancellation token state.
        Concurrency::details::_CancellationTokenState * _M_pTokenState;

        // The registration on the token.
        Concurrency::details::_CancellationTokenRegistration * _M_pRegistration;

        // The async task collection wrapper
#if _MSC_VER >= 1800
        Concurrency::details::_TaskCollection_t _M_TaskCollection;

        // Callstack for function call (constructor or .then) that created this task impl.
        _TaskCreationCallstack _M_pTaskCreationCallstack;

        _TaskEventLogger _M_taskEventLogger;
#else
        Concurrency::details::_AsyncTaskCollection * _M_pTaskCollection;

        // Points to the source code instruction right after the function call (constructor or .then) that created this task impl.
        void* _M_pTaskCreationAddressHint;
#endif

    private:
        // Must not be copied by value:
        _Task_impl_base(const _Task_impl_base&);
        _Task_impl_base const & operator=(_Task_impl_base const&);
    };

#if _MSC_VER >= 1800
#if _PPLTASK_ASYNC_LOGGING
    inline void _TaskEventLogger::_LogTaskCompleted()
    {
        if (_M_scheduled)
        {
            ::Windows::Foundation::AsyncStatus _State;
            if (_M_task->_IsCompleted())
                _State = ::Windows::Foundation::AsyncStatus::Completed;
            else if (_M_task->_HasUserException())
                _State = ::Windows::Foundation::AsyncStatus::Error;
            else
                _State = ::Windows::Foundation::AsyncStatus::Canceled;

            if (details::_IsCausalitySupported())
            {
                ::Windows::Foundation::Diagnostics::AsyncCausalityTracer::TraceOperationCompletion(::Windows::Foundation::Diagnostics::CausalityTraceLevel::Required, ::Windows::Foundation::Diagnostics::CausalitySource::Library,
                    _PPLTaskCausalityPlatformID, reinterpret_cast<unsigned long long>(_M_task), _State);
            }
        }
    }
#endif
#endif

    template<typename _ReturnType>
    struct _Task_impl : public _Task_impl_base
    {
        typedef ABI::Windows::Foundation::IAsyncInfo _AsyncOperationType;
#if _MSC_VER >= 1800
        _Task_impl(Concurrency::details::_CancellationTokenState * _Ct, Concurrency::scheduler_ptr _Scheduler_arg)
            : _Task_impl_base(_Ct, _Scheduler_arg)
#else
        _Task_impl(Concurrency::details::_CancellationTokenState * _Ct) : _Task_impl_base(_Ct)
#endif
        {
            _M_unwrapped_async_op = nullptr;
        }
        virtual ~_Task_impl()
        {
            // We must invoke _DeregisterCancellation in the derived class destructor. Calling it in the base class destructor could cause
            // a partially initialized _Task_impl to be in the list of registrations for a cancellation token.
            _DeregisterCancellation();
        }
        virtual bool _CancelAndRunContinuations(bool _SynchronousCancel, bool _UserException, bool _PropagatedFromAncestor, const std::shared_ptr<_ExceptionHolder> & _ExceptionHolder)
        {
            enum { _Nothing, _RunContinuations, _Cancel } _Do = _Nothing;
            {
                scoped_lock _LockHolder(_M_ContinuationsCritSec);
                if (_UserException)
                {
                    _CONCRT_ASSERT(_SynchronousCancel && !_IsCompleted());
                    // If the state is _Canceled, the exception has to be coming from an ancestor.
                    _CONCRT_ASSERT(!_IsCanceled() || _PropagatedFromAncestor);
#if _MSC_VER < 1800
                    // If the state is _Started or _PendingCancel, the exception cannot be coming from an ancestor.
                    _CONCRT_ASSERT((!_IsStarted() && !_IsPendingCancel()) || !_PropagatedFromAncestor);
#endif
                    // We should not be canceled with an exception more than once.
                    _CONCRT_ASSERT(!_HasUserException());

                    if (_M_TaskState == _Canceled)
                    {
                        // If the task has finished cancelling there should not be any continuation records in the array.
                        return false;
                    }
                    else
                    {
                        _CONCRT_ASSERT(_M_TaskState != _Completed);
                        _M_exceptionHolder = _ExceptionHolder;
                    }
                }
                else
                {
                    // Completed is a non-cancellable state, and if this is an asynchronous cancel, we're unable to do better than the last async cancel
                    // which is to say, cancellation is already initiated, so return early.
                    if (_IsCompleted() || _IsCanceled() || (_IsPendingCancel() && !_SynchronousCancel))
                    {
                        _CONCRT_ASSERT(!_IsCompleted() || !_HasUserException());
                        return false;
                    }
                    _CONCRT_ASSERT(!_SynchronousCancel || !_HasUserException());
                }

#if _MSC_VER >= 1800
                if (_SynchronousCancel)
#else
                if (_SynchronousCancel || _IsCreated())
#endif
                {
                    // Be aware that this set must be done BEFORE _M_Scheduled being set, or race will happen between this and wait()
                    _M_TaskState = _Canceled;
#if _MSC_VER < 1800
                    _M_Scheduled.set();
#endif

                    // Cancellation completes the task, so all dependent tasks must be run to cancel them
                    // They are canceled when they begin running (see _RunContinuation) and see that their
                    // ancestor has been canceled.
                    _Do = _RunContinuations;
                }
                else
                {
#if _MSC_VER >= 1800
                    _CONCRT_ASSERT(!_UserException);

                    if (_IsStarted())
                    {
                        // should not initiate cancellation under a lock
                        _Do = _Cancel;
                    }

                    // The _M_TaskState variable transitions to _Canceled when cancellation is completed (the task is not executing user code anymore).
                    // In the case of a synchronous cancel, this can happen immediately, whereas with an asynchronous cancel, the task has to move from
                    // _Started to _PendingCancel before it can move to _Canceled when it is finished executing.
                    _M_TaskState = _PendingCancel;

                    _M_taskEventLogger._LogCancelTask();
                }
            }

            switch (_Do)
            {
            case _Cancel:
            {
#else
                    _CONCRT_ASSERT(_IsStarted() && !_UserException);
#endif
                    // The _M_TaskState variable transitions to _Canceled when cancellation is completed (the task is not executing user code anymore).
                    // In the case of a synchronous cancel, this can happen immediately, whereas with an asynchronous cancel, the task has to move from
                    // _Started to _PendingCancel before it can move to _Canceled when it is finished executing.
                    _M_TaskState = _PendingCancel;
                    if (_M_unwrapped_async_op != nullptr)
                    {
                        // We will only try to cancel async operation but not unwrapped tasks, since unwrapped tasks cannot be canceled without its token.
                        if (_M_unwrapped_async_op)    _M_unwrapped_async_op->Cancel();
                    }
#if _MSC_VER >= 1800
                        _M_TaskCollection._Cancel();
                        break;
#else
                // Optimistic trying for cancelation
                if (_M_pTaskCollection != nullptr)
                {
                    _M_pTaskCollection->_Cancel();
                }
#endif
                }
#if _MSC_VER < 1800
            }
#endif

            // Only execute continuations and mark the task as completed if we were able to move the task to the _Canceled state.
#if _MSC_VER >= 1800
            case _RunContinuations:
            {
                _M_TaskCollection._Complete();
#else
            if (_RunContinuations)
            {
                _M_Completed.set();
#endif

                if (_M_Continuations)
                {
                    // Scheduling cancellation with automatic inlining.
                    details::_ScheduleFuncWithAutoInline([=]() -> HRESULT { _RunTaskContinuations(); return S_OK; }, Concurrency::details::_DefaultAutoInline);
                }
#if _MSC_VER >= 1800
                break;
            }
#endif
            }
            return true;
        }
        void _FinalizeAndRunContinuations(_ReturnType _Result)
        {

#if _MSC_VER >= 1800
            _M_Result.Set(_Result);
#else
            _M_Result = _Result;
            _M_ResultContext = _ResultContext<_ReturnType>::_GetContext(_M_fRuntimeAggregate);
#endif
            {
                //
                // Hold this lock to ensure continuations being concurrently either get added
                // to the _M_Continuations vector or wait for the result
                //
                scoped_lock _LockHolder(_M_ContinuationsCritSec);

                // A task could still be in the _Created state if it was created with a task_completion_event.
                // It could also be in the _Canceled state for the same reason.
                _CONCRT_ASSERT(!_HasUserException() && !_IsCompleted());
                if (_IsCanceled())
                {
                    return;
                }

                // Always transition to "completed" state, even in the face of unacknowledged pending cancellation
                _M_TaskState = _Completed;
            }
#if _MSC_VER >= 1800
            _M_TaskCollection._Complete();
#else
            _M_Completed.set();
#endif
            _RunTaskContinuations();
        }
        //
        // This method is invoked when the starts executing. The task returns early if this method returns true.
        //
        bool _TransitionedToStarted()
        {
            scoped_lock _LockHolder(_M_ContinuationsCritSec);
#if _MSC_VER >= 1800
            // Canceled state could only result from antecedent task's canceled state, but that code path will not reach here.
            _ASSERT(!_IsCanceled());
            if (_IsPendingCancel())
#else
            if (_IsCanceled())
#endif
            {
                return false;
            }
            _CONCRT_ASSERT(_IsCreated());
            _M_TaskState = _Started;
            return true;
        }
        void _SetUnwrappedAsyncOp(_AsyncOperationType* _AsyncOp)
        {
            scoped_lock _LockHolder(_M_ContinuationsCritSec);
            // Cancel the async operation if the task itself is canceled, since the thread that canceled the task missed it.
            if (_IsPendingCancel())
            {
                _CONCRT_ASSERT(!_IsCanceled());
                if (_AsyncOp) _AsyncOp->Cancel();
            }
            else
            {
                _M_unwrapped_async_op = _AsyncOp;
            }
        }
#if _MSC_VER >= 1800
        // Return true if the task has reached a terminal state
        bool _IsDone()
        {
            return _IsCompleted() || _IsCanceled();
        }
#endif
        _ReturnType _GetResult()
        {
#if _MSC_VER >= 1800
            return _M_Result.Get();
#else
            return _ResultContext<_ReturnType>::_GetValue(_M_Result, _M_ResultContext, _M_fRuntimeAggregate);
#endif
        }
#if _MSC_VER >= 1800
        _ResultHolder<_ReturnType>                 _M_Result;        // this means that the result type must have a public default ctor.
#else
        _ReturnType                                 _M_Result;        // this means that the result type must have a public default ctor.
#endif
        Microsoft::WRL::ComPtr<_AsyncOperationType> _M_unwrapped_async_op;
#if _MSC_VER < 1800
        _ContextCallback                            _M_ResultContext;
#endif
    };

    template<typename _ResultType>
    struct _Task_completion_event_impl
    {
#if _MSC_VER >= 1800
    private:
        _Task_completion_event_impl(const _Task_completion_event_impl&);
        _Task_completion_event_impl& operator=(const _Task_completion_event_impl&);

    public:
#endif
        typedef std::vector<typename _Task_ptr<_ResultType>::_Type> _TaskList;

        _Task_completion_event_impl() : _M_fHasValue(false), _M_fIsCanceled(false)
        {
        }

        bool _HasUserException()
        {
            return _M_exceptionHolder != nullptr;
        }

        ~_Task_completion_event_impl()
        {
            for (auto _TaskIt = _M_tasks.begin(); _TaskIt != _M_tasks.end(); ++_TaskIt)
            {
                _CONCRT_ASSERT(!_M_fHasValue && !_M_fIsCanceled);
                // Cancel the tasks since the event was never signaled or canceled.
                (*_TaskIt)->_Cancel(true);
            }
        }

        // We need to protect the loop over the array, so concurrent_vector would not have helped
        _TaskList                           _M_tasks;
        critical_section                    _M_taskListCritSec;
#if _MSC_VER >= 1800
        _ResultHolder<_ResultType>         _M_value;
#else
        _ResultType                         _M_value;
#endif
        std::shared_ptr<_ExceptionHolder>   _M_exceptionHolder;
        bool                                _M_fHasValue;
        bool                                _M_fIsCanceled;
    };

    // Utility method for dealing with void functions
    inline std::function<HRESULT(_Unit_type*)> _MakeVoidToUnitFunc(const std::function<HRESULT(void)>& _Func)
    {
        return [=](_Unit_type* retVal) -> HRESULT { HRESULT hr = _Func(); *retVal = _Unit_type(); return hr; };
    }

    template <typename _Type>
    std::function<HRESULT(_Unit_type, _Type*)> _MakeUnitToTFunc(const std::function<HRESULT(_Type*)>& _Func)
    {
        return [=](_Unit_type, _Type* retVal) -> HRESULT { HRESULT hr = _Func(retVal); return hr;  };
    }

    template <typename _Type>
    std::function<HRESULT(_Type, _Unit_type*)> _MakeTToUnitFunc(const std::function<HRESULT(_Type)>& _Func)
    {
        return[=](_Type t, _Unit_type* retVal) -> HRESULT { HRESULT hr = _Func(t); *retVal = _Unit_type(); return hr;  };
    }

    inline std::function<HRESULT(_Unit_type, _Unit_type*)> _MakeUnitToUnitFunc(const std::function<HRESULT(void)>& _Func)
    {
        return [=](_Unit_type, _Unit_type* retVal) -> HRESULT { HRESULT hr = _Func(); *retVal = _Unit_type(); return hr;  };
    }
}


/// <summary>
///     The <c>task_completion_event</c> class allows you to delay the execution of a task until a condition is satisfied,
///     or start a task in response to an external event.
/// </summary>
/// <typeparam name="_ResultType">
///     The result type of this <c>task_completion_event</c> class.
/// </typeparam>
/// <remarks>
///     Use a task created from a task completion event when your scenario requires you to create a task that will complete, and
///     thereby have its continuations scheduled for execution, at some point in the future. The <c>task_completion_event</c> must
///     have the same type as the task you create, and calling the set method on the task completion event with a value of that type
///     will cause the associated task to complete, and provide that value as a result to its continuations.
///     <para>If the task completion event is never signaled, any tasks created from it will be canceled when it is destructed.</para>
///     <para><c>task_completion_event</c> behaves like a smart pointer, and should be passed by value.</para>
/// </remarks>
/// <seealso cref="task Class"/>
/**/
template<typename _ResultType>
class task_completion_event
{
public:
    /// <summary>
    ///     Constructs a <c>task_completion_event</c> object.
    /// </summary>
    /**/
    task_completion_event() : _M_Impl(std::make_shared<details::_Task_completion_event_impl<_ResultType>>())
    {
    }

    /// <summary>
    ///     Sets the task completion event.
    /// </summary>
    /// <param name="_Result">
    ///     The result to set this event with.
    /// </param>
    /// <returns>
    ///     The method returns <c>true</c> if it was successful in setting the event. It returns <c>false</c> if the event is already set.
    /// </returns>
    /// <remarks>
    ///     In the presence of multiple or concurrent calls to <c>set</c>, only the first call will succeed and its result (if any) will be stored in the
    ///     task completion event. The remaining sets are ignored and the method will return false. When you set a task completion event, all the
    ///     tasks created from that event will immediately complete, and its continuations, if any, will be scheduled. Task completion objects that have
    ///     a <typeparamref name="_ResultType"/> other than <c>void</c> will pass the value <paramref value="_Result"/> to their continuations.
    /// </remarks>
    /**/
    bool set(_ResultType _Result) const // 'const' (even though it's not deep) allows to safely pass events by value into lambdas
    {
        // Subsequent sets are ignored. This makes races to set benign: the first setter wins and all others are ignored.
        if (_IsTriggered())
        {
            return false;
        }

        _TaskList _Tasks;
        bool _RunContinuations = false;
        {
            details::scoped_lock _LockHolder(_M_Impl->_M_taskListCritSec);

            if (!_IsTriggered())
            {
#if _MSC_VER >= 1800
                _M_Impl->_M_value.Set(_Result);
#else
                _M_Impl->_M_value = _Result;
#endif
                _M_Impl->_M_fHasValue = true;

                _Tasks.swap(_M_Impl->_M_tasks);
                _RunContinuations = true;
            }
        }

        if (_RunContinuations)
        {
            for (auto _TaskIt = _Tasks.begin(); _TaskIt != _Tasks.end(); ++_TaskIt)
            {
#if _MSC_VER >= 1800
                // If current task was cancelled by a cancellation_token, it would be in cancel pending state.
                if ((*_TaskIt)->_IsPendingCancel())
                    (*_TaskIt)->_Cancel(true);
                else
                {
                    // Tasks created with task_completion_events can be marked as async, (we do this in when_any and when_all
                    // if one of the tasks involved is an async task). Since continuations of async tasks can execute inline, we
                    // need to run continuations after the lock is released.
                    (*_TaskIt)->_FinalizeAndRunContinuations(_M_Impl->_M_value.Get());
                }
#else
                // Tasks created with task_completion_events can be marked as async, (we do this in when_any and when_all
                // if one of the tasks involved is an async task). Since continuations of async tasks can execute inline, we
                // need to run continuations after the lock is released.
                (*_TaskIt)->_FinalizeAndRunContinuations(_M_Impl->_M_value);
#endif
            }
            if (_M_Impl->_HasUserException())
            {
                _M_Impl->_M_exceptionHolder.reset();
            }
            return true;
        }

        return false;
    }
#if _MSC_VER >= 1800

    template<typename _E>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        bool set_exception(_E _Except) const // 'const' (even though it's not deep) allows to safely pass events by value into lambdas
    {
            // It is important that _CAPTURE_CALLSTACK() evaluate to the instruction after the call instruction for set_exception.
            return _Cancel(std::make_exception_ptr(_Except), _CAPTURE_CALLSTACK());
    }
#endif

    /// <summary>
    ///     Propagates an exception to all tasks associated with this event.
    /// </summary>
    /// <param>
    ///     The exception_ptr that indicates the exception to set this event with.
    /// </param>
    /**/
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        bool set_exception(std::exception_ptr _ExceptionPtr) const // 'const' (even though it's not deep) allows to safely pass events by value into lambdas
    {
            // It is important that _ReturnAddress() evaluate to the instruction after the call instruction for set_exception.
#if _MSC_VER >= 1800
            return _Cancel(_ExceptionPtr, _CAPTURE_CALLSTACK());
#else
            return _Cancel(_ExceptionPtr, _ReturnAddress());
#endif
    }

    /// <summary>
    ///     Internal method to cancel the task_completion_event. Any task created using this event will be marked as canceled if it has
    ///     not already been set.
    /// </summary>
    bool _Cancel() const
    {
        // Cancel with the stored exception if one exists.
        return _CancelInternal();
    }

    /// <summary>
    ///     Internal method to cancel the task_completion_event with the exception provided. Any task created using this event will be canceled
    ///     with the same exception.
    /// </summary>
    template<typename _ExHolderType>
#if _MSC_VER >= 1800
    bool _Cancel(_ExHolderType _ExHolder, const details::_TaskCreationCallstack &_SetExceptionAddressHint = details::_TaskCreationCallstack()) const
#else
    bool _Cancel(_ExHolderType _ExHolder, void* _SetExceptionAddressHint = nullptr) const
#endif
    {
        (void)_SetExceptionAddressHint;
        bool _Canceled;
#if _MSC_VER >= 1800
        if(_StoreException(_ExHolder, _SetExceptionAddressHint))
#else
        if (_StoreException(_ExHolder))
#endif
        {
            _Canceled = _CancelInternal();
            _CONCRT_ASSERT(_Canceled);
        }
        else
        {
            _Canceled = false;
        }
        return _Canceled;
    }

    /// <summary>
    ///     Internal method that stores an exception in the task completion event. This is used internally by when_any.
    ///     Note, this does not cancel the task completion event. A task completion event with a stored exception
    ///     can bet set() successfully. If it is canceled, it will cancel with the stored exception, if one is present.
    /// </summary>
    template<typename _ExHolderType>
#if _MSC_VER >= 1800
    bool _StoreException(_ExHolderType _ExHolder, const details::_TaskCreationCallstack &_SetExceptionAddressHint = details::_TaskCreationCallstack()) const
#else
    bool _StoreException(_ExHolderType _ExHolder, void* _SetExceptionAddressHint = nullptr) const
#endif
    {
        details::scoped_lock _LockHolder(_M_Impl->_M_taskListCritSec);
        if (!_IsTriggered() && !_M_Impl->_HasUserException())
        {
            // Create the exception holder only if we have ensured there we will be successful in setting it onto the
            // task completion event. Failing to do so will result in an unobserved task exception.
            _M_Impl->_M_exceptionHolder = _ToExceptionHolder(_ExHolder, _SetExceptionAddressHint);
            return true;
        }
        return false;
    }

    /// <summary>
    ///     Tests whether current event has been either Set, or Canceled.
    /// </summary>
    bool _IsTriggered() const
    {
        return _M_Impl->_M_fHasValue || _M_Impl->_M_fIsCanceled;
    }

private:

#if _MSC_VER >= 1800
    static std::shared_ptr<details::_ExceptionHolder> _ToExceptionHolder(const std::shared_ptr<details::_ExceptionHolder>& _ExHolder, const details::_TaskCreationCallstack&)
#else
    static std::shared_ptr<details::_ExceptionHolder> _ToExceptionHolder(const std::shared_ptr<details::_ExceptionHolder>& _ExHolder, void*)
#endif
    {
        return _ExHolder;
    }

#if _MSC_VER >= 1800
    static std::shared_ptr<details::_ExceptionHolder> _ToExceptionHolder(std::exception_ptr _ExceptionPtr, const details::_TaskCreationCallstack &_SetExceptionAddressHint)
#else
    static std::shared_ptr<details::_ExceptionHolder> _ToExceptionHolder(std::exception_ptr _ExceptionPtr, void* _SetExceptionAddressHint)
#endif
    {
        return std::make_shared<details::_ExceptionHolder>(_ExceptionPtr, _SetExceptionAddressHint);
    }

    template <typename T> friend class task; // task can register itself with the event by calling the private _RegisterTask
    template <typename T> friend class task_completion_event;

    typedef typename details::_Task_completion_event_impl<_ResultType>::_TaskList _TaskList;

    /// <summary>
    ///    Cancels the task_completion_event.
    /// </summary>
    bool _CancelInternal() const
    {
        // Cancellation of task completion events is an internal only utility. Our usage is such that _CancelInternal
        // will never be invoked if the task completion event has been set.
        _CONCRT_ASSERT(!_M_Impl->_M_fHasValue);
        if (_M_Impl->_M_fIsCanceled)
        {
            return false;
        }

        _TaskList _Tasks;
        bool _Cancel = false;
        {
            details::scoped_lock _LockHolder(_M_Impl->_M_taskListCritSec);
            _CONCRT_ASSERT(!_M_Impl->_M_fHasValue);
            if (!_M_Impl->_M_fIsCanceled)
            {
                _M_Impl->_M_fIsCanceled = true;
                _Tasks.swap(_M_Impl->_M_tasks);
                _Cancel = true;
            }
        }

        bool _UserException = _M_Impl->_HasUserException();

        if (_Cancel)
        {
            for (auto _TaskIt = _Tasks.begin(); _TaskIt != _Tasks.end(); ++_TaskIt)
            {
                // Need to call this after the lock is released. See comments in set().
                if (_UserException)
                {
                    (*_TaskIt)->_CancelWithExceptionHolder(_M_Impl->_M_exceptionHolder, true);
                }
                else
                {
                    (*_TaskIt)->_Cancel(true);
                }
            }
        }
        return _Cancel;
    }

    /// <summary>
    ///     Register a task with this event. This function is called when a task is constructed using
    ///     a task_completion_event.
    /// </summary>
    void _RegisterTask(const typename details::_Task_ptr<_ResultType>::_Type & _TaskParam)
    {
        details::scoped_lock _LockHolder(_M_Impl->_M_taskListCritSec);
#if _MSC_VER < 1800
        _TaskParam->_SetScheduledEvent();
#endif
        //If an exception was already set on this event, then cancel the task with the stored exception.
        if (_M_Impl->_HasUserException())
        {
            _TaskParam->_CancelWithExceptionHolder(_M_Impl->_M_exceptionHolder, true);
        }
        else if (_M_Impl->_M_fHasValue)
        {
#if _MSC_VER >= 1800
            _TaskParam->_FinalizeAndRunContinuations(_M_Impl->_M_value.Get());
#else
            _TaskParam->_FinalizeAndRunContinuations(_M_Impl->_M_value);
#endif
        }
        else
        {
            _M_Impl->_M_tasks.push_back(_TaskParam);
        }
    }

    std::shared_ptr<details::_Task_completion_event_impl<_ResultType>> _M_Impl;
};

/// <summary>
///     The <c>task_completion_event</c> class allows you to delay the execution of a task until a condition is satisfied,
///     or start a task in response to an external event.
/// </summary>
/// <remarks>
///     Use a task created from a task completion event when your scenario requires you to create a task that will complete, and
///     thereby have its continuations scheduled for execution, at some point in the future. The <c>task_completion_event</c> must
///     have the same type as the task you create, and calling the set method on the task completion event with a value of that type
///     will cause the associated task to complete, and provide that value as a result to its continuations.
///     <para>If the task completion event is never signaled, any tasks created from it will be canceled when it is destructed.</para>
///     <para><c>task_completion_event</c> behaves like a smart pointer, and should be passed by value.</para>
/// </remarks>
/// <seealso cref="task Class"/>
/**/
template<>
class task_completion_event<void>
{
public:
    /// <summary>
    ///     Sets the task completion event.
    /// </summary>
    /// <returns>
    ///     The method returns <c>true</c> if it was successful in setting the event. It returns <c>false</c> if the event is already set.
    /// </returns>
    /// <remarks>
    ///     In the presence of multiple or concurrent calls to <c>set</c>, only the first call will succeed and its result (if any) will be stored in the
    ///     task completion event. The remaining sets are ignored and the method will return false. When you set a task completion event, all the
    ///     tasks created from that event will immediately complete, and its continuations, if any, will be scheduled. Task completion objects that have
    ///     a <typeparamref name="_ResultType"/> other than <c>void</c> will pass the value <paramref value="_Result"/> to their continuations.
    /// </remarks>
    /**/
    bool set() const // 'const' (even though it's not deep) allows to safely pass events by value into lambdas
    {
        return _M_unitEvent.set(details::_Unit_type());
    }
#if _MSC_VER >= 1800

    template<typename _E>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        bool set_exception(_E _Except) const // 'const' (even though it's not deep) allows to safely pass events by value into lambdas
    {
            return _M_unitEvent._Cancel(std::make_exception_ptr(_Except), _CAPTURE_CALLSTACK());
    }
#endif

    /// <summary>
    ///     Propagates an exception to all tasks associated with this event.
    /// </summary>
    /// <param>
    ///     The exception_ptr that indicates the exception to set this event with.
    /// </param>
    /**/
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        bool set_exception(std::exception_ptr _ExceptionPtr) const // 'const' (even though it's not deep) allows to safely pass events by value into lambdas
    {
        // It is important that _ReturnAddress() evaluate to the instruction after the call instruction for set_exception.
#if _MSC_VER >= 1800
            return _M_unitEvent._Cancel(_ExceptionPtr, _CAPTURE_CALLSTACK());
#else
            return _M_unitEvent._Cancel(_ExceptionPtr, _ReturnAddress());
#endif
    }

    /// <summary>
    ///     Cancel the task_completion_event. Any task created using this event will be marked as canceled if it has
    ///     not already been set.
    /// </summary>
    void _Cancel() const // 'const' (even though it's not deep) allows to safely pass events by value into lambdas
    {
        _M_unitEvent._Cancel();
    }

    /// <summary>
    ///     Cancel the task_completion_event with the exception holder provided. Any task created using this event will be canceled
    ///     with the same exception.
    /// </summary>
    void _Cancel(const std::shared_ptr<details::_ExceptionHolder>& _ExHolder) const
    {
        _M_unitEvent._Cancel(_ExHolder);
    }

    /// <summary>
    ///     Method that stores an exception in the task completion event. This is used internally by when_any.
    ///     Note, this does not cancel the task completion event. A task completion event with a stored exception
    ///     can bet set() successfully. If it is canceled, it will cancel with the stored exception, if one is present.
    /// </summary>
    bool _StoreException(const std::shared_ptr<details::_ExceptionHolder>& _ExHolder) const
    {
        return _M_unitEvent._StoreException(_ExHolder);
    }

    /// <summary>
    ///     Test whether current event has been either Set, or Canceled.
    /// </summary>
    bool _IsTriggered() const
    {
        return _M_unitEvent._IsTriggered();
    }

private:
    template <typename T> friend class task; // task can register itself with the event by calling the private _RegisterTask

    /// <summary>
    ///     Register a task with this event. This function is called when a task is constructed using
    ///     a task_completion_event.
    /// </summary>
    void _RegisterTask(details::_Task_ptr<details::_Unit_type>::_Type _TaskParam)
    {
        _M_unitEvent._RegisterTask(_TaskParam);
    }

    // The void event contains an event a dummy type so common code can be used for events with void and non-void results.
    task_completion_event<details::_Unit_type> _M_unitEvent;
};
namespace details
{
    //
    // Compile-time validation helpers
    //

    // Task constructor validation: issue helpful diagnostics for common user errors. Do not attempt full validation here.
    //
    // Anything callable is fine
    template<typename _ReturnType, typename _Ty>
    auto _IsValidTaskCtor(_Ty _Param, int, int, int, int, int, int, int) -> typename decltype(_Param(), std::true_type());

    // Anything callable with a task return value is fine
    template<typename _ReturnType, typename _Ty>
    auto _IsValidTaskCtor(_Ty _Param, int, int, int, int, int, int, ...) -> typename decltype(_Param(stdx::declval<task<_ReturnType>*>()), std::true_type());

    // Anything callable with a return value is fine
    template<typename _ReturnType, typename _Ty>
    auto _IsValidTaskCtor(_Ty _Param, int, int, int, int, int, ...) -> typename decltype(_Param(stdx::declval<_ReturnType*>()), std::true_type());

    // Anything that has GetResults is fine: this covers AsyncAction*
    template<typename _ReturnType, typename _Ty>
    auto _IsValidTaskCtor(_Ty _Param, int, int, int, int, ...) -> typename decltype(_Param->GetResults(), std::true_type());

    // Anything that has GetResults(TResult_abi*) is fine: this covers AsyncOperation*
    template<typename _ReturnType, typename _Ty>
    auto _IsValidTaskCtor(_Ty _Param, int, int, int, ...) -> typename decltype(_Param->GetResults(stdx::declval<decltype(_GetUnwrappedType(stdx::declval<_Ty>()))*>()), std::true_type());

    // Allow parameters with set: this covers task_completion_event
    template<typename _ReturnType, typename _Ty>
    auto _IsValidTaskCtor(_Ty _Param, int, int, ...) -> typename decltype(_Param.set(stdx::declval<_ReturnType>()), std::true_type());

    template<typename _ReturnType, typename _Ty>
    auto _IsValidTaskCtor(_Ty _Param, int, ...) -> typename decltype(_Param.set(), std::true_type());

    // All else is invalid
    template<typename _ReturnType, typename _Ty>
    std::false_type _IsValidTaskCtor(_Ty _Param, ...);

    template<typename _ReturnType, typename _Ty>
    void _ValidateTaskConstructorArgs(_Ty _Param)
    {
        (void)_Param;
        static_assert(std::is_same<decltype(details::_IsValidTaskCtor<_ReturnType>(_Param, 0, 0, 0, 0, 0, 0, 0)), std::true_type>::value,
            "incorrect argument for task constructor; can be a callable object, an asynchronous operation, or a task_completion_event"
            );
        static_assert(!(std::is_same<_Ty, _ReturnType>::value && details::_IsIAsyncInfo<_Ty>::_Value),
            "incorrect template argument for task; consider using the return type of the async operation");
    }
    // Helpers for create_async validation
    //
    // A parameter lambda taking no arguments is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, int, int, int, int, int, int, int) -> typename decltype(_Param(), std::true_type());

    // A parameter lambda taking a result argument is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, int, int, int, int, int, int, ...) -> typename decltype(_Param(stdx::declval<_ReturnType*>()), std::true_type());

    // A parameter lambda taking an cancellation_token argument is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, int, int, int, int, int, ...) -> typename decltype(_Param(Concurrency::cancellation_token::none()), std::true_type());

    // A parameter lambda taking an cancellation_token argument and a result argument is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, int, int, int, int, ...) -> typename decltype(_Param(Concurrency::cancellation_token::none(), stdx::declval<_ReturnType*>()), std::true_type());

    // A parameter lambda taking a progress report argument is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, int, int, int, ...) -> typename decltype(_Param(details::_ProgressReporterCtorArgType()), std::true_type());

    // A parameter lambda taking a progress report argument and a result argument is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, int, int, ...) -> typename decltype(_Param(details::_ProgressReporterCtorArgType(), stdx::declval<_ReturnType*>()), std::true_type());

    // A parameter lambda taking a progress report and a cancellation_token argument is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, int, ...) -> typename decltype(_Param(details::_ProgressReporterCtorArgType(), Concurrency::cancellation_token::none()), std::true_type());

    // A parameter lambda taking a progress report and a cancellation_token argument and a result argument is valid
    template<typename _ReturnType, typename _Ty>
    static auto _IsValidCreateAsync(_Ty _Param, int, ...) -> typename decltype(_Param(details::_ProgressReporterCtorArgType(), Concurrency::cancellation_token::none(), stdx::declval<_ReturnType*>()), std::true_type());

    // All else is invalid
    template<typename _ReturnType, typename _Ty>
    static std::false_type _IsValidCreateAsync(_Ty _Param, ...);
}

/// <summary>
///     The Parallel Patterns Library (PPL) <c>task</c> class. A <c>task</c> object represents work that can be executed asynchronously,
///     and concurrently with other tasks and parallel work produced by parallel algorithms in the Concurrency Runtime. It produces
///     a result of type <typeparamref name="_ResultType"/> on successful completion. Tasks of type <c>task&lt;void&gt;</c> produce no result.
///     A task can be waited upon and canceled independently of other tasks. It can also be composed with other tasks using
///     continuations(<c>then</c>), and join(<c>when_all</c>) and choice(<c>when_any</c>) patterns.
/// </summary>
/// <typeparam name="_ReturnType">
///     The result type of this task.
/// </typeparam>
/// <remarks>
///     For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.
/// </remarks>
/**/
template<typename _ReturnType>
class task
{
public:
    /// <summary>
    ///     The type of the result an object of this class produces.
    /// </summary>
    /**/
    typedef _ReturnType result_type;

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    task() : _M_Impl(nullptr)
    {
        // The default constructor should create a task with a nullptr impl. This is a signal that the
        // task is not usable and should throw if any wait(), get() or then() APIs are used.
    }

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <typeparam name="_Ty">
    ///     The type of the parameter from which the task is to be constructed.
    /// </typeparam>
    /// <param name="_Param">
    ///     The parameter from which the task is to be constructed. This could be a lambda, a function object, a <c>task_completion_event&lt;result_type&gt;</c>
    ///     object, or a Windows::Foundation::IAsyncInfo if you are using tasks in your Windows Store app. The lambda or function
    ///     object should be a type equivalent to <c>std::function&lt;X(void)&gt;</c>, where X can be a variable of type <c>result_type</c>,
    ///     <c>task&lt;result_type&gt;</c>, or a Windows::Foundation::IAsyncInfo in Windows Store apps.
    /// </param>
    /// <param name="_Token">
    ///     The cancellation token to associate with this task. A task created without a cancellation token cannot be canceled. It implicitly receives
    ///     the token <c>cancellation_token::none()</c>.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Ty>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        explicit task(_Ty _Param)
    {
#if _MSC_VER >= 1800
            task_options _TaskOptions;
#endif
            details::_ValidateTaskConstructorArgs<_ReturnType, _Ty>(_Param);

#if _MSC_VER >= 1800
            _CreateImpl(_TaskOptions.get_cancellation_token()._GetImplValue(), _TaskOptions.get_scheduler());
#else
            _CreateImpl(Concurrency::cancellation_token::none()._GetImplValue());
#endif
            // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of the task constructor.
#if _MSC_VER >= 1800
            _SetTaskCreationCallstack(_CAPTURE_CALLSTACK());
#else
            _SetTaskCreationAddressHint(_ReturnAddress());
#endif
            _TaskInitMaybeFunctor(_Param, details::_IsCallable<_ReturnType>(_Param, 0, 0, 0));
        }

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <typeparam name="_Ty">
    ///     The type of the parameter from which the task is to be constructed.
    /// </typeparam>
    /// <param name="_Param">
    ///     The parameter from which the task is to be constructed. This could be a lambda, a function object, a <c>task_completion_event&lt;result_type&gt;</c>
    ///     object, or a Windows::Foundation::IAsyncInfo if you are using tasks in your Windows Store app. The lambda or function
    ///     object should be a type equivalent to <c>std::function&lt;X(void)&gt;</c>, where X can be a variable of type <c>result_type</c>,
    ///     <c>task&lt;result_type&gt;</c>, or a Windows::Foundation::IAsyncInfo in Windows Store apps.
    /// </param>
    /// <param name="_Token">
    ///     The cancellation token to associate with this task. A task created without a cancellation token cannot be canceled. It implicitly receives
    ///     the token <c>cancellation_token::none()</c>.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Ty>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
#if _MSC_VER >= 1800
        explicit task(_Ty _Param, const task_options &_TaskOptions)
#else
        explicit task(_Ty _Param, Concurrency::cancellation_token _Token)
#endif
    {
            details::_ValidateTaskConstructorArgs<_ReturnType, _Ty>(_Param);

#if _MSC_VER >= 1800
            _CreateImpl(_TaskOptions.get_cancellation_token()._GetImplValue(), _TaskOptions.get_scheduler());
#else
            _CreateImpl(_Token._GetImplValue());
#endif
            // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of the task constructor.
#if _MSC_VER >= 1800
            _SetTaskCreationCallstack(details::_get_internal_task_options(_TaskOptions)._M_hasPresetCreationCallstack ? details::_get_internal_task_options(_TaskOptions)._M_presetCreationCallstack : _CAPTURE_CALLSTACK());
#else
            _SetTaskCreationAddressHint(_ReturnAddress());
#endif
            _TaskInitMaybeFunctor(_Param, details::_IsCallable<_ReturnType>(_Param, 0, 0, 0));
        }

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    task(const task& _Other) : _M_Impl(_Other._M_Impl) {}

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    task(task&& _Other) : _M_Impl(std::move(_Other._M_Impl)) {}

    /// <summary>
    ///     Replaces the contents of one <c>task</c> object with another.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     As <c>task</c> behaves like a smart pointer, after a copy assignment, this <c>task</c> objects represents the same
    ///     actual task as <paramref name="_Other"/> does.
    /// </remarks>
    /**/
    task& operator=(const task& _Other)
    {
        if (this != &_Other)
        {
            _M_Impl = _Other._M_Impl;
        }
        return *this;
    }

    /// <summary>
    ///     Replaces the contents of one <c>task</c> object with another.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     As <c>task</c> behaves like a smart pointer, after a copy assignment, this <c>task</c> objects represents the same
    ///     actual task as <paramref name="_Other"/> does.
    /// </remarks>
    /**/
    task& operator=(task&& _Other)
    {
        if (this != &_Other)
        {
            _M_Impl = std::move(_Other._M_Impl);
        }
        return *this;
    }

    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        auto then(const _Function& _Func) const -> typename details::_ContinuationTypeTraits<_Function, _ReturnType>::_TaskOfType
    {
#if _MSC_VER >= 1800
            task_options _TaskOptions;
            details::_get_internal_task_options(_TaskOptions)._set_creation_callstack(_CAPTURE_CALLSTACK());
            return _ThenImpl<_ReturnType, _Function>(_Func, _TaskOptions);
#else
            auto _ContinuationTask = _ThenImpl<_ReturnType, _Function>(_Func, nullptr, task_continuation_context::use_default());
            // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
            _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
            return _ContinuationTask;
#endif
        }

    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <param name="_CancellationToken">
    ///     The cancellation token to associate with the continuation task. A continuation task that is created without a cancellation token will inherit
    ///     the token of its antecedent task.
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
#if _MSC_VER >= 1800
        auto then(const _Function& _Func, task_options _TaskOptions) const -> typename details::_ContinuationTypeTraits<_Function, _ReturnType>::_TaskOfType
#else
        auto then(const _Function& _Func, Concurrency::cancellation_token _CancellationToken) const -> typename details::_ContinuationTypeTraits<_Function, _ReturnType>::_TaskOfType
#endif
    {
#if _MSC_VER >= 1800
            details::_get_internal_task_options(_TaskOptions)._set_creation_callstack(_CAPTURE_CALLSTACK());
            return _ThenImpl<_ReturnType, _Function>(_Func, _TaskOptions);
#else
        auto _ContinuationTask = _ThenImpl<_ReturnType, _Function>(_Func, _CancellationToken._GetImplValue(), task_continuation_context::use_default());
        // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
        _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
        return _ContinuationTask;
#endif
    }
#if _MSC_VER < 1800
    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <param name="_ContinuationContext">
    ///     A variable that specifies where the continuation should execute. This variable is only useful when used in a
    ///     Windows Store app. For more information, see <see cref="task_continuation_context Class">task_continuation_context</see>
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        auto then(const _Function& _Func, task_continuation_context _ContinuationContext) const -> typename details::_ContinuationTypeTraits<_Function, _ReturnType>::_TaskOfType
    {
        auto _ContinuationTask = _ThenImpl<_ReturnType, _Function>(_Func, nullptr, _ContinuationContext);
        // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
        _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
        return _ContinuationTask;
    }
#endif
    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <param name="_CancellationToken">
    ///     The cancellation token to associate with the continuation task. A continuation task that is created without a cancellation token will inherit
    ///     the token of its antecedent task.
    /// </param>
    /// <param name="_ContinuationContext">
    ///     A variable that specifies where the continuation should execute. This variable is only useful when used in a
    ///     Windows Store app. For more information, see <see cref="task_continuation_context Class">task_continuation_context</see>
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        auto then(const _Function& _Func, Concurrency::cancellation_token _CancellationToken, task_continuation_context _ContinuationContext) const -> typename details::_ContinuationTypeTraits<_Function, _ReturnType>::_TaskOfType
    {
#if _MSC_VER >= 1800
            task_options _TaskOptions(_CancellationToken, _ContinuationContext);
            details::_get_internal_task_options(_TaskOptions)._set_creation_callstack(_CAPTURE_CALLSTACK());
            return _ThenImpl<_ReturnType, _Function>(_Func, _TaskOptions);
#else
            auto _ContinuationTask = _ThenImpl<_ReturnType, _Function>(_Func, _CancellationToken._GetImplValue(), _ContinuationContext);
            // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
            _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
            return _ContinuationTask;
#endif
        }

    /// <summary>
    ///     Waits for this task to reach a terminal state. It is possible for <c>wait</c> to execute the task inline, if all of the tasks
    ///     dependencies are satisfied, and it has not already been picked up for execution by a background worker.
    /// </summary>
    /// <returns>
    ///     A <c>task_status</c> value which could be either <c>completed</c> or <c>canceled</c>. If the task encountered an exception
    ///     during execution, or an exception was propagated to it from an antecedent task, <c>wait</c> will throw that exception.
    /// </returns>
    /**/
    task_status wait() const
    {
        if (_M_Impl == nullptr)
        {
            throw Concurrency::invalid_operation("wait() cannot be called on a default constructed task.");
        }

        return _M_Impl->_Wait();
    }

    /// <summary>
    ///     Returns the result this task produced. If the task is not in a terminal state, a call to <c>get</c> will wait for the task to
    ///     finish. This method does not return a value when called on a task with a <c>result_type</c> of <c>void</c>.
    /// </summary>
    /// <returns>
    ///     The result of the task.
    /// </returns>
    /// <remarks>
    ///     If the task is canceled, a call to <c>get</c> will throw a <see cref="task_canceled Class">task_canceled</see> exception. If the task
    ///     encountered an different exception or an exception was propagated to it from an antecedent task, a call to <c>get</c> will throw that exception.
    /// </remarks>
    /**/
    _ReturnType get() const
    {
        if (_M_Impl == nullptr)
        {
            throw Concurrency::invalid_operation("get() cannot be called on a default constructed task.");
        }

        if (_M_Impl->_Wait() == Concurrency::canceled)
        {
            throw Concurrency::task_canceled();
        }

        return _M_Impl->_GetResult();
    }
#if _MSC_VER >= 1800
    /// <summary>
    ///     Determines if the task is completed.
    /// </summary>
    /// <returns>
    ///     True if the task has completed, false otherwise.
    /// </returns>
    /// <remarks>
    ///     The function returns true if the task is completed or canceled (with or without user exception).
    /// </remarks>
    bool is_done() const
    {
        if (!_M_Impl)
        {
            throw Concurrency::invalid_operation("is_done() cannot be called on a default constructed task.");
        }

        return _M_Impl->_IsDone();
    }

    /// <summary>
    ///     Returns the scheduler for this task
    /// </summary>
    /// <returns>
    ///     A pointer to the scheduler
    /// </returns>
    Concurrency::scheduler_ptr scheduler() const
    {
        if (!_M_Impl)
        {
            throw Concurrency::invalid_operation("scheduler() cannot be called on a default constructed task.");
        }

        return _M_Impl->_GetScheduler();
    }
#endif
    /// <summary>
    ///     Determines whether the task unwraps a Windows Runtime <c>IAsyncInfo</c> interface or is descended from such a task.
    /// </summary>
    /// <returns>
    ///     <c>true</c> if the task unwraps an <c>IAsyncInfo</c> interface or is descended from such a task, <c>false</c> otherwise.
    /// </returns>
    /**/
    bool is_apartment_aware() const
    {
        if (_M_Impl == nullptr)
        {
            throw Concurrency::invalid_operation("is_apartment_aware() cannot be called on a default constructed task.");
        }
        return _M_Impl->_IsApartmentAware();
    }

    /// <summary>
    ///     Determines whether two <c>task</c> objects represent the same internal task.
    /// </summary>
    /// <returns>
    ///     <c>true</c> if the objects refer to the same underlying task, and <c>false</c> otherwise.
    /// </returns>
    /**/
    bool operator==(const task<_ReturnType>& _Rhs) const
    {
        return (_M_Impl == _Rhs._M_Impl);
    }

    /// <summary>
    ///     Determines whether two <c>task</c> objects represent different internal tasks.
    /// </summary>
    /// <returns>
    ///     <c>true</c> if the objects refer to different underlying tasks, and <c>false</c> otherwise.
    /// </returns>
    /**/
    bool operator!=(const task<_ReturnType>& _Rhs) const
    {
        return !operator==(_Rhs);
    }

    /// <summary>
    ///     Create an underlying task implementation.
    /// </summary>
#if _MSC_VER >= 1800
    void _CreateImpl(Concurrency::details::_CancellationTokenState * _Ct, Concurrency::scheduler_ptr _Scheduler)
#else
    void _CreateImpl(Concurrency::details::_CancellationTokenState * _Ct)
#endif
    {
        _CONCRT_ASSERT(_Ct != nullptr);
#if _MSC_VER >= 1800
        _M_Impl = details::_Task_ptr<_ReturnType>::_Make(_Ct, _Scheduler);
#else
        _M_Impl = details::_Task_ptr<_ReturnType>::_Make(_Ct);
#endif
        if (_Ct != Concurrency::details::_CancellationTokenState::_None())
        {
#if _MSC_VER >= 1800
            _M_Impl->_RegisterCancellation(_M_Impl);
#else
            _M_Impl->_RegisterCancellation();
#endif
        }
    }

    /// <summary>
    ///     Return the underlying implementation for this task.
    /// </summary>
    const typename details::_Task_ptr<_ReturnType>::_Type & _GetImpl() const
    {
        return _M_Impl;
    }

    /// <summary>
    ///     Set the implementation of the task to be the supplied implementaion.
    /// </summary>
    void _SetImpl(const typename details::_Task_ptr<_ReturnType>::_Type & _Impl)
    {
        _CONCRT_ASSERT(_M_Impl == nullptr);
        _M_Impl = _Impl;
    }

    /// <summary>
    ///     Set the implementation of the task to be the supplied implementaion using a move instead of a copy.
    /// </summary>
    void _SetImpl(typename details::_Task_ptr<_ReturnType>::_Type && _Impl)
    {
        _CONCRT_ASSERT(_M_Impl == nullptr);
        _M_Impl = std::move(_Impl);
    }

    /// <summary>
    ///     Sets a property determining whether the task is apartment aware.
    /// </summary>
    void _SetAsync(bool _Async = true)
    {
        _GetImpl()->_SetAsync(_Async);
    }

    /// <summary>
    ///     Sets a field in the task impl to the return address for calls to the task constructors and the then method.
    /// </summary>
#if _MSC_VER >= 1800
    void _SetTaskCreationCallstack(const details::_TaskCreationCallstack &_callstack)
    {
        _GetImpl()->_SetTaskCreationCallstack(_callstack);
    }
#else
    void _SetTaskCreationAddressHint(void* _Address)
    {
        _GetImpl()->_SetTaskCreationAddressHint(_Address);
    }
#endif
    /// <summary>
    ///     An internal version of then that takes additional flags and always execute the continuation inline by default.
    ///     When _ForceInline is set to false, continuations inlining will be limited to default _DefaultAutoInline.
    ///     This function is Used for runtime internal continuations only.
    /// </summary>
    template<typename _Function>
#if _MSC_VER >= 1800
    auto _Then(const _Function& _Func, Concurrency::details::_CancellationTokenState *_PTokenState,
        details::_TaskInliningMode _InliningMode = Concurrency::details::_ForceInline) const -> typename details::_ContinuationTypeTraits<_Function, _ReturnType>::_TaskOfType
    {
        // inherit from antecedent
        auto _Scheduler = _GetImpl()->_GetScheduler();

        return _ThenImpl<_ReturnType, _Function>(_Func, _PTokenState, task_continuation_context::use_default(), _Scheduler, _CAPTURE_CALLSTACK(), _InliningMode);
    }
#else
    auto _Then(const _Function& _Func, Concurrency::details::_CancellationTokenState *_PTokenState, bool _Aggregating,
        details::_TaskInliningMode _InliningMode = Concurrency::details::_ForceInline) const -> typename details::_ContinuationTypeTraits<_Function, _ReturnType>::_TaskOfType
    {
        return _ThenImpl<_ReturnType, _Function>(_Func, _PTokenState, task_continuation_context::use_default(), _Aggregating, _InliningMode);
    }
#endif

private:
    template <typename T> friend class task;

    // A helper class template that transforms an intial task lambda returns void into a lambda that returns a non-void type (details::_Unit_type is used
    // to substitute for void). This is to minimize the special handling required for 'void'.
    template<typename _RetType>
    class _Init_func_transformer
    {
    public:
        static auto _Perform(std::function<HRESULT(_RetType*)> _Func) -> decltype(_Func)
        {
            return _Func;
        }
    };

    template<>
    class _Init_func_transformer<void>
    {
    public:
        static auto _Perform(std::function<HRESULT(void)> _Func) -> decltype(details::_MakeVoidToUnitFunc(_Func))
        {
            return details::_MakeVoidToUnitFunc(_Func);
        }
    };

    // The task handle type used to construct an 'initial task' - a task with no dependents.
    template <typename _InternalReturnType, typename _Function, typename _TypeSelection>
    struct _InitialTaskHandle :
        details::_PPLTaskHandle<_ReturnType, _InitialTaskHandle<_InternalReturnType, _Function, _TypeSelection>, details::_UnrealizedChore>
    {
        _Function _M_function;
        _InitialTaskHandle(const typename details::_Task_ptr<_ReturnType>::_Type & _TaskImpl, const _Function & _Function) : _M_function(_Function), _PPLTaskHandle(_TaskImpl)
        {
        }
        virtual ~_InitialTaskHandle() {}

#if _MSC_VER >= 1800
        template <typename _Func, typename _RetArg>
        auto _LogWorkItemAndInvokeUserLambda(_Func && _func, _RetArg && _retArg) const -> decltype(_func(std::forward<_RetArg>(_retArg)))
        {
            details::_TaskWorkItemRAIILogger _LogWorkItem(this->_M_pTask->_M_taskEventLogger);
            return _func(std::forward<_RetArg>(_retArg));
        }
#endif

        void _Perform() const
        {
            _Init(_TypeSelection());
        }
#if _MSC_VER >= 1800

        void _SyncCancelAndPropagateException() const
        {
            this->_M_pTask->_Cancel(true);
        }
#endif
        //
        // Overload 0: returns _InternalReturnType
        //
        // This is the most basic task with no unwrapping
        //
        void _Init(details::_TypeSelectorNoAsync) const
        {
            _ReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Init_func_transformer<_InternalReturnType>::_Perform(_M_function), &retVal);
#else
            HRESULT hr = _Init_func_transformer<_InternalReturnType>::_Perform(_M_function)(&retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            _M_pTask->_FinalizeAndRunContinuations(retVal);
        }

        //
        // Overload 1: returns IAsyncOperation<_InternalReturnType>*
        //                   or
        //             returns task<_InternalReturnType>
        //
        // This is task whose functor returns an async operation or a task which will be unwrapped for continuation
        // Depending on the output type, the right _AsyncInit gets invoked
        //
        void _Init(details::_TypeSelectorAsyncTask) const
        {
            task<_InternalReturnType> retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, &retVal);
#else
            HRESULT hr = _M_function(&retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_ReturnType, _InternalReturnType>(_M_pTask, retVal);
        }
        void _Init(details::_TypeSelectorAsyncOperation) const
        {
            _ReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, &retVal);
#else
            HRESULT hr = _M_function(&retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_ReturnType, _InternalReturnType>(_M_pTask,
                Microsoft::WRL::Make<details::_IAsyncOperationToAsyncOperationConverter<_InternalReturnType>>(retVal).Get());
        }

        //
        // Overload 2: returns IAsyncAction*
        //
        // This is task whose functor returns an async action which will be unwrapped for continuation
        //
        void _Init(details::_TypeSelectorAsyncAction) const
        {
            _ReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, &retVal);
#else
            HRESULT hr = _M_function(&retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_ReturnType, _InternalReturnType>(_M_pTask, Microsoft::WRL::Make<details::_IAsyncActionToAsyncOperationConverter>(retVal).Get());
        }

        //
        // Overload 3: returns IAsyncOperationWithProgress<_InternalReturnType, _ProgressType>*
        //
        // This is task whose functor returns an async operation with progress which will be unwrapped for continuation
        //
        void _Init(details::_TypeSelectorAsyncOperationWithProgress) const
        {
            typedef details::_GetProgressType<decltype(_M_function())>::_Value _ProgressType;
            _ReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, &retVal);
#else
            HRESULT hr = _M_function(&retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_ReturnType, _InternalReturnType>(_M_pTask,
                Microsoft::WRL::Make<details::_IAsyncOperationWithProgressToAsyncOperationConverter<_InternalReturnType, _ProgressType>>(retVal).Get());
        }

        //
        // Overload 4: returns IAsyncActionWithProgress<_ProgressType>*
        //
        // This is task whose functor returns an async action with progress which will be unwrapped for continuation
        //
        void _Init(details::_TypeSelectorAsyncActionWithProgress) const
        {
            typedef details::_GetProgressType<decltype(_M_function())>::_Value _ProgressType;
            _ReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, &retVal);
#else
            HRESULT hr = _M_function(&retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_ReturnType, _InternalReturnType>(_M_pTask,
                Microsoft::WRL::Make<details::_IAsyncActionWithProgressToAsyncOperationConverter<_ProgressType>>(retVal).Get());
        }
    };

    /// <summary>
    ///     A helper class template that transforms a continuation lambda that either takes or returns void, or both, into a lambda that takes and returns a
    ///     non-void type (details::_Unit_type is used to substitute for void). This is to minimize the special handling required for 'void'.
    /// </summary>
    template<typename _InpType, typename _OutType>
    class _Continuation_func_transformer
    {
    public:
        static auto _Perform(std::function<HRESULT(_InpType, _OutType*)> _Func) -> decltype(_Func)
        {
            return _Func;
        }
    };

    template<typename _OutType>
    class _Continuation_func_transformer<void, _OutType>
    {
    public:
        static auto _Perform(std::function<HRESULT(_OutType*)> _Func) -> decltype(details::_MakeUnitToTFunc<_OutType>(_Func))
        {
            return details::_MakeUnitToTFunc<_OutType>(_Func);
        }
    };

    template<typename _InType>
    class _Continuation_func_transformer<_InType, void>
    {
    public:
        static auto _Perform(std::function<HRESULT(_InType)> _Func) -> decltype(details::_MakeTToUnitFunc<_InType>(_Func))
        {
            return details::_MakeTToUnitFunc<_InType>(_Func);
        }
    };

    template<>
    class _Continuation_func_transformer<void, void>
    {
    public:
        static auto _Perform(std::function<HRESULT(void)> _Func) -> decltype(details::_MakeUnitToUnitFunc(_Func))
        {
            return details::_MakeUnitToUnitFunc(_Func);
        }
    };
    /// <summary>
    ///     The task handle type used to create a 'continuation task'.
    /// </summary>
    template <typename _InternalReturnType, typename _ContinuationReturnType, typename _Function, typename _IsTaskBased, typename _TypeSelection>
    struct _ContinuationTaskHandle :
        details::_PPLTaskHandle<typename details::_NormalizeVoidToUnitType<_ContinuationReturnType>::_Type,
        _ContinuationTaskHandle<_InternalReturnType, _ContinuationReturnType, _Function, _IsTaskBased, _TypeSelection>, details::_ContinuationTaskHandleBase>
    {
        typedef typename details::_NormalizeVoidToUnitType<_ContinuationReturnType>::_Type _NormalizedContinuationReturnType;

        typename details::_Task_ptr<_ReturnType>::_Type _M_ancestorTaskImpl;
        _Function _M_function;

        _ContinuationTaskHandle(const typename details::_Task_ptr<_ReturnType>::_Type & _AncestorImpl,
            const typename details::_Task_ptr<_NormalizedContinuationReturnType>::_Type & _ContinuationImpl,
            const _Function & _Func, const task_continuation_context & _Context, details::_TaskInliningMode _InliningMode) :
#if _MSC_VER >= 1800
            details::_PPLTaskHandle<typename details::_NormalizeVoidToUnitType<_ContinuationReturnType>::_Type,
            _ContinuationTaskHandle<_InternalReturnType, _ContinuationReturnType, _Function, _IsTaskBased, _TypeSelection>, details::_ContinuationTaskHandleBase>
            ::_PPLTaskHandle(_ContinuationImpl)
            , _M_ancestorTaskImpl(_AncestorImpl)
            , _M_function(_Func)
#else
            _M_ancestorTaskImpl(_AncestorImpl), _PPLTaskHandle(_ContinuationImpl), _M_function(_Func)
#endif
        {
            _M_isTaskBasedContinuation = _IsTaskBased::value;
            _M_continuationContext = _Context;
            _M_continuationContext._Resolve(_AncestorImpl->_IsApartmentAware());
            _M_inliningMode = _InliningMode;
        }

        virtual ~_ContinuationTaskHandle() {}

#if _MSC_VER >= 1800
        template <typename _Func, typename _Arg, typename _RetArg>
        auto _LogWorkItemAndInvokeUserLambda(_Func && _func, _Arg && _value, _RetArg && _retArg) const -> decltype(_func(std::forward<_Arg>(_value), std::forward<_RetArg>(_retArg)))
        {
            details::_TaskWorkItemRAIILogger _LogWorkItem(this->_M_pTask->_M_taskEventLogger);
            return _func(std::forward<_Arg>(_value), std::forward<_RetArg>(_retArg));
        }
#endif

        void _Perform() const
        {
            _Continue(_IsTaskBased(), _TypeSelection());
        }

#if _MSC_VER >= 1800
        void _SyncCancelAndPropagateException() const
        {
            if (_M_ancestorTaskImpl->_HasUserException())
            {
                // If the ancestor encountered an exception, transfer the exception to the continuation
                // This traverses down the tree to propagate the exception.
                this->_M_pTask->_CancelWithExceptionHolder(_M_ancestorTaskImpl->_GetExceptionHolder(), true);
            }
            else
            {
                // If the ancestor was canceled, then your own execution should be canceled.
                // This traverses down the tree to cancel it.
                this->_M_pTask->_Cancel(true);
            }
        }
#endif

        //
        // Overload 0-0: _InternalReturnType -> _TaskType
        //
        // This is a straight task continuation which simply invokes its target with the ancestor's completion argument
        //
        void _Continue(std::false_type, details::_TypeSelectorNoAsync) const
        {
            _NormalizedContinuationReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Continuation_func_transformer<_InternalReturnType, _ContinuationReturnType>::_Perform(_M_function), _M_ancestorTaskImpl->_GetResult(), &retVal);
#else
            HRESULT hr =_Continuation_func_transformer<_InternalReturnType, _ContinuationReturnType>::_Perform(_M_function)(_M_ancestorTaskImpl->_GetResult(), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            _M_pTask->_FinalizeAndRunContinuations(retVal);
        }

        //
        // Overload 0-1: _InternalReturnType -> IAsyncOperation<_TaskType>*
        //               or
        //               _InternalReturnType -> task<_TaskType>
        //
        // This is a straight task continuation which returns an async operation or a task which will be unwrapped for continuation
        // Depending on the output type, the right _AsyncInit gets invoked
        //
        void _Continue(std::false_type, details::_TypeSelectorAsyncTask) const
        {
            typedef typename details::_FunctionTypeTraits<_Function, _InternalReturnType>::_FuncRetType _FuncOutputType;
            _FuncOutputType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function), _M_ancestorTaskImpl->_GetResult(), &retVal);
#else
            HRESULT hr = _Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function)(_M_ancestorTaskImpl->_GetResult(), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(
                _M_pTask,
                retVal
                );
        }
        void _Continue(std::false_type, details::_TypeSelectorAsyncOperation) const
        {
            typedef typename details::_FunctionTypeTraits<_Function, _InternalReturnType>::_FuncRetType _FuncOutputType;
            _FuncOutputType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function), _M_ancestorTaskImpl->_GetResult(), &retVal);
#else
            HRESULT hr = _Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function)(_M_ancestorTaskImpl->_GetResult(), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(
                _M_pTask,
                Microsoft::WRL::Make<details::_IAsyncOperationToAsyncOperationConverter<_ContinuationReturnType>>(retVal).Get());
        }

        //
        // Overload 0-2: _InternalReturnType -> IAsyncAction*
        //
        // This is a straight task continuation which returns an async action which will be unwrapped for continuation
        //
        void _Continue(std::false_type, details::_TypeSelectorAsyncAction) const
        {
            typedef details::_FunctionTypeTraits<_Function, _InternalReturnType>::_FuncRetType _FuncOutputType;
            _FuncOutputType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function), _M_ancestorTaskImpl->_GetResult(), &retVal);
#else
            HRESULT hr = _Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function)(_M_ancestorTaskImpl->_GetResult(), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(
                _M_pTask,
                Microsoft::WRL::Make<details::_IAsyncActionToAsyncOperationConverter>(
                retVal).Get());
        }

        //
        // Overload 0-3: _InternalReturnType -> IAsyncOperationWithProgress<_TaskType, _ProgressType>*
        //
        // This is a straight task continuation which returns an async operation with progress which will be unwrapped for continuation
        //
        void _Continue(std::false_type, details::_TypeSelectorAsyncOperationWithProgress) const
        {
            typedef details::_FunctionTypeTraits<_Function, _InternalReturnType>::_FuncRetType _FuncOutputType;

            _FuncOutputType _OpWithProgress;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function), _M_ancestorTaskImpl->_GetResult(), &_OpWithProgress);
#else
            HRESULT hr = _Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function)(_M_ancestorTaskImpl->_GetResult(), &_OpWithProgress);
#endif
            typedef details::_GetProgressType<decltype(_OpWithProgress)>::_Value _ProgressType;

            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(
                _M_pTask,
                Microsoft::WRL::Make<details::_IAsyncOperationWithProgressToAsyncOperationConverter<_ContinuationReturnType, _ProgressType>>(_OpWithProgress).Get());
        }

        //
        // Overload 0-4: _InternalReturnType -> IAsyncActionWithProgress<_ProgressType>*
        //
        // This is a straight task continuation which returns an async action with progress which will be unwrapped for continuation
        //
        void _Continue(std::false_type, details::_TypeSelectorAsyncActionWithProgress) const
        {
            typedef details::_FunctionTypeTraits<_Function, _InternalReturnType>::_FuncRetType _FuncOutputType;

            _FuncOutputType _OpWithProgress;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function), _M_ancestorTaskImpl->_GetResult(), &_OpWithProgress);
#else
            HRESULT hr = _Continuation_func_transformer<_InternalReturnType, _FuncOutputType>::_Perform(_M_function)(_M_ancestorTaskImpl->_GetResult(), &_OpWithProgress);
#endif
            typedef details::_GetProgressType<decltype(_OpWithProgress)>::_Value _ProgressType;

            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(
                _M_pTask,
                Microsoft::WRL::Make<details::_IAsyncActionWithProgressToAsyncOperationConverter<_ProgressType>>(_OpWithProgress).Get());
        }


        //
        // Overload 1-0: task<_InternalReturnType> -> _TaskType
        //
        // This is an exception handling type of continuation which takes the task rather than the task's result.
        //
        void _Continue(std::true_type, details::_TypeSelectorNoAsync) const
        {
            typedef task<_InternalReturnType> _FuncInputType;
            task<_InternalReturnType> _ResultTask;
            _ResultTask._SetImpl(std::move(_M_ancestorTaskImpl));
            _NormalizedContinuationReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_Continuation_func_transformer<_FuncInputType, _ContinuationReturnType>::_Perform(_M_function), std::move(_ResultTask), &retVal);
#else
            HRESULT hr = _Continuation_func_transformer<_FuncInputType, _ContinuationReturnType>::_Perform(_M_function)(std::move(_ResultTask), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            _M_pTask->_FinalizeAndRunContinuations(retVal);
        }

        //
        // Overload 1-1: task<_InternalReturnType> -> IAsyncOperation<_TaskType>^
        //                                            or
        //                                            task<_TaskType>
        //
        // This is an exception handling type of continuation which takes the task rather than
        // the task's result. It also returns an async operation or a task which will be unwrapped
        // for continuation
        //
        void _Continue(std::true_type, details::_TypeSelectorAsyncTask) const
        {
            // The continuation takes a parameter of type task<_Input>, which is the same as the ancestor task.
            task<_InternalReturnType> _ResultTask;
            _ResultTask._SetImpl(std::move(_M_ancestorTaskImpl));
            _ContinuationReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, std::move(_ResultTask), &retVal);
#else
            HRESULT hr = _M_function(std::move(_ResultTask), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(_M_pTask, retVal);
        }
        void _Continue(std::true_type, details::_TypeSelectorAsyncOperation) const
        {
            // The continuation takes a parameter of type task<_Input>, which is the same as the ancestor task.
            task<_InternalReturnType> _ResultTask;
            _ResultTask._SetImpl(std::move(_M_ancestorTaskImpl));
            _ContinuationReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, std::move(_ResultTask), &retVal);
#else
            HRESULT hr = _M_function(std::move(_ResultTask), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(_M_pTask,
                Microsoft::WRL::Make<details::_IAsyncOperationToAsyncOperationConverter<_ContinuationReturnType>>(retVal));
        }

        //
        // Overload 1-2: task<_InternalReturnType> -> IAsyncAction*
        //
        // This is an exception handling type of continuation which takes the task rather than
        // the task's result. It also returns an async action which will be unwrapped for continuation
        //
        void _Continue(std::true_type, details::_TypeSelectorAsyncAction) const
        {
            // The continuation takes a parameter of type task<_Input>, which is the same as the ancestor task.
            task<_InternalReturnType> _ResultTask;
            _ResultTask._SetImpl(std::move(_M_ancestorTaskImpl));
            _ContinuationReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, std::move(_ResultTask), &retVal);
#else
            HRESULT hr = _M_function(std::move(_ResultTask), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(_M_pTask,
                Microsoft::WRL::Make<details::_IAsyncActionToAsyncOperationConverter>(retVal));
        }

        //
        // Overload 1-3: task<_InternalReturnType> -> IAsyncOperationWithProgress<_TaskType, _ProgressType>*
        //
        // This is an exception handling type of continuation which takes the task rather than
        // the task's result. It also returns an async operation with progress which will be unwrapped
        // for continuation
        //
        void _Continue(std::true_type, details::_TypeSelectorAsyncOperationWithProgress) const
        {
            // The continuation takes a parameter of type task<_Input>, which is the same as the ancestor task.
            task<_InternalReturnType> _ResultTask;
            _ResultTask._SetImpl(std::move(_M_ancestorTaskImpl));

            typedef details::_GetProgressType<decltype(_M_function(_ResultTask))>::_Value _ProgressType;
            _ContinuationReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, std::move(_ResultTask), &retVal);
#else
            HRESULT hr = _M_function(std::move(_ResultTask), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(_M_pTask,
                Microsoft::WRL::Make<details::_IAsyncOperationWithProgressToAsyncOperationConverter<_ContinuationReturnType, _ProgressType>>(retVal));
        }

        //
        // Overload 1-4: task<_InternalReturnType> -> IAsyncActionWithProgress<_ProgressType>*
        //
        // This is an exception handling type of continuation which takes the task rather than
        // the task's result. It also returns an async operation with progress which will be unwrapped
        // for continuation
        //
        void _Continue(std::true_type, details::_TypeSelectorAsyncActionWithProgress) const
        {
            // The continuation takes a parameter of type task<_Input>, which is the same as the ancestor task.
            task<_InternalReturnType> _ResultTask;
            _ResultTask._SetImpl(std::move(_M_ancestorTaskImpl));

            typedef details::_GetProgressType<decltype(_M_function(_ResultTask))>::_Value _ProgressType;
            _ContinuationReturnType retVal;
#if _MSC_VER >= 1800
            HRESULT hr = _LogWorkItemAndInvokeUserLambda(_M_function, std::move(_ResultTask), &retVal);
#else
            HRESULT hr = _M_function(std::move(_ResultTask), &retVal);
#endif
            if (FAILED(hr)) throw std::make_exception_ptr(hr);
            details::_Task_impl_base::_AsyncInit<_NormalizedContinuationReturnType, _ContinuationReturnType>(_M_pTask,
                Microsoft::WRL::Make<details::_IAsyncActionWithProgressToAsyncOperationConverter<_ProgressType>>(retVal));
        }
    };
    /// <summary>
    ///     Initializes a task using a lambda, function pointer or function object.
    /// </summary>
    template<typename _InternalReturnType, typename _Function>
    void _TaskInitWithFunctor(const _Function& _Func)
    {
        typedef details::_InitFunctorTypeTraits<_InternalReturnType, details::_FunctionTypeTraits<_Function, void>::_FuncRetType> _Async_type_traits;

        _M_Impl->_M_fFromAsync = _Async_type_traits::_IsAsyncTask;
        _M_Impl->_M_fUnwrappedTask = _Async_type_traits::_IsUnwrappedTaskOrAsync;
#if _MSC_VER >= 1800
        _M_Impl->_M_taskEventLogger._LogScheduleTask(false);
#endif
        _M_Impl->_ScheduleTask(new _InitialTaskHandle<_InternalReturnType, _Function, typename _Async_type_traits::_AsyncKind>(_GetImpl(), _Func), Concurrency::details::_NoInline);
    }

    /// <summary>
    ///     Initializes a task using a task completion event.
    /// </summary>
    void _TaskInitNoFunctor(task_completion_event<_ReturnType>& _Event)
    {
        _Event._RegisterTask(_M_Impl);
    }

    /// <summary>
    ///     Initializes a task using an asynchronous operation IAsyncOperation<T>*
    /// </summary>
    template<typename _Result, typename _OpType, typename _CompHandlerType, typename _ResultType>
    void _TaskInitAsyncOp(details::_AsyncInfoImpl<_OpType, _CompHandlerType, _ResultType>* _AsyncOp)
    {
        _M_Impl->_M_fFromAsync = true;
#if _MSC_VER < 1800
        _M_Impl->_SetScheduledEvent();
#endif
        // Mark this task as started here since we can set the state in the constructor without acquiring a lock. Once _AsyncInit
        // returns a completion could execute concurrently and the task must be fully initialized before that happens.
        _M_Impl->_M_TaskState = details::_Task_impl_base::_Started;
        // Pass the shared pointer into _AsyncInit for storage in the Async Callback.
        details::_Task_impl_base::_AsyncInit<_ReturnType, _Result>(_M_Impl, _AsyncOp);
    }

    /// <summary>
    ///     Initializes a task using an asynchronous operation IAsyncOperation<T>*
    /// </summary>
    template<typename _Result>
    void _TaskInitNoFunctor(ABI::Windows::Foundation::IAsyncOperation<_Result>* _AsyncOp)
    {
        _TaskInitAsyncOp<_Result>(Microsoft::WRL::Make<details::_IAsyncOperationToAsyncOperationConverter<_Result>>(_AsyncOp).Get());
    }

    /// <summary>
    ///     Initializes a task using an asynchronous operation with progress IAsyncOperationWithProgress<T, P>*
    /// </summary>
    template<typename _Result, typename _Progress>
    void _TaskInitNoFunctor(ABI::Windows::Foundation::IAsyncOperationWithProgress<_Result, _Progress>* _AsyncOp)
    {
        _TaskInitAsyncOp<_Result>(Microsoft::WRL::Make<details::_IAsyncOperationWithProgressToAsyncOperationConverter<_Result, _Progress>>(_AsyncOp).Get());
    }
    /// <summary>
    ///     Initializes a task using a callable object.
    /// </summary>
    template<typename _Function>
    void _TaskInitMaybeFunctor(_Function & _Func, std::true_type)
    {
        _TaskInitWithFunctor<_ReturnType, _Function>(_Func);
    }

    /// <summary>
    ///     Initializes a task using a non-callable object.
    /// </summary>
    template<typename _Ty>
    void _TaskInitMaybeFunctor(_Ty & _Param, std::false_type)
    {
        _TaskInitNoFunctor(_Param);
    }
#if _MSC_VER >= 1800
    template<typename _InternalReturnType, typename _Function>
    auto _ThenImpl(const _Function& _Func, const task_options& _TaskOptions) const -> typename details::_ContinuationTypeTraits<_Function, _InternalReturnType>::_TaskOfType
    {
        if (!_M_Impl)
        {
            throw Concurrency::invalid_operation("then() cannot be called on a default constructed task.");
        }

        Concurrency::details::_CancellationTokenState *_PTokenState = _TaskOptions.has_cancellation_token() ? _TaskOptions.get_cancellation_token()._GetImplValue() : nullptr;
        auto _Scheduler = _TaskOptions.has_scheduler() ? _TaskOptions.get_scheduler() : _GetImpl()->_GetScheduler();
        auto _CreationStack = details::_get_internal_task_options(_TaskOptions)._M_hasPresetCreationCallstack ? details::_get_internal_task_options(_TaskOptions)._M_presetCreationCallstack : details::_TaskCreationCallstack();
        return _ThenImpl<_InternalReturnType, _Function>(_Func, _PTokenState, _TaskOptions.get_continuation_context(), _Scheduler, _CreationStack);
    }
#endif
    /// <summary>
    ///     The one and only implementation of then for void and non-void tasks.
    /// </summary>
    template<typename _InternalReturnType, typename _Function>
#if _MSC_VER >= 1800
    auto _ThenImpl(const _Function& _Func, Concurrency::details::_CancellationTokenState *_PTokenState, const task_continuation_context& _ContinuationContext, Concurrency::scheduler_ptr _Scheduler, details::_TaskCreationCallstack _CreationStack,
        details::_TaskInliningMode _InliningMode = Concurrency::details::_NoInline) const -> typename details::_ContinuationTypeTraits<_Function, _InternalReturnType>::_TaskOfType
#else
    auto _ThenImpl(const _Function& _Func, Concurrency::details::_CancellationTokenState *_PTokenState, const task_continuation_context& _ContinuationContext,
        bool _Aggregating = false, details::_TaskInliningMode _InliningMode = Concurrency::details::_NoInline) const -> typename details::_ContinuationTypeTraits<_Function, _InternalReturnType>::_TaskOfType
#endif
    {
        if (_M_Impl == nullptr)
        {
            throw Concurrency::invalid_operation("then() cannot be called on a default constructed task.");
        }

        typedef details::_FunctionTypeTraits<_Function, _InternalReturnType> _Function_type_traits;
        typedef details::_TaskTypeTraits<typename _Function_type_traits::_FuncRetType> _Async_type_traits;
        typedef typename _Async_type_traits::_TaskRetType _TaskType;

        //
        // A **nullptr** token state indicates that it was not provided by the user. In this case, we inherit the antecedent's token UNLESS this is a
        // an exception handling continuation. In that case, we break the chain with a _None. That continuation is never canceled unless the user
        // explicitly passes the same token.
        //
        if (_PTokenState == nullptr)
        {
#if _MSC_VER >= 1800
            if (_Function_type_traits::_Takes_task::value)
#else
            if (_Function_type_traits::_Takes_task())
#endif
            {
                _PTokenState = Concurrency::details::_CancellationTokenState::_None();
            }
            else
            {
                _PTokenState = _GetImpl()->_M_pTokenState;
            }
        }

        task<_TaskType> _ContinuationTask;
#if _MSC_VER >= 1800
        _ContinuationTask._CreateImpl(_PTokenState, _Scheduler);
#else
        _ContinuationTask._CreateImpl(_PTokenState);
#endif
        _ContinuationTask._GetImpl()->_M_fFromAsync = (_GetImpl()->_M_fFromAsync || _Async_type_traits::_IsAsyncTask);
#if _MSC_VER < 1800
        _ContinuationTask._GetImpl()->_M_fRuntimeAggregate = _Aggregating;
#endif
        _ContinuationTask._GetImpl()->_M_fUnwrappedTask = _Async_type_traits::_IsUnwrappedTaskOrAsync;
#if _MSC_VER >= 1800
        _ContinuationTask._SetTaskCreationCallstack(_CreationStack);
#endif
        _GetImpl()->_ScheduleContinuation(new _ContinuationTaskHandle<_InternalReturnType, _TaskType, _Function, typename _Function_type_traits::_Takes_task, typename _Async_type_traits::_AsyncKind>(
            _GetImpl(), _ContinuationTask._GetImpl(), _Func, _ContinuationContext, _InliningMode));

        return _ContinuationTask;
    }

    // The underlying implementation for this task
    typename details::_Task_ptr<_ReturnType>::_Type _M_Impl;
};

/// <summary>
///     The Parallel Patterns Library (PPL) <c>task</c> class. A <c>task</c> object represents work that can be executed asynchronously,
///     and concurrently with other tasks and parallel work produced by parallel algorithms in the Concurrency Runtime. It produces
///     a result of type <typeparamref name="_ResultType"/> on successful completion. Tasks of type <c>task&lt;void&gt;</c> produce no result.
///     A task can be waited upon and canceled independently of other tasks. It can also be composed with other tasks using
///     continuations(<c>then</c>), and join(<c>when_all</c>) and choice(<c>when_any</c>) patterns.
/// </summary>
/// <remarks>
///     For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.
/// </remarks>
/**/
template<>
class task<void>
{
public:
    /// <summary>
    ///     The type of the result an object of this class produces.
    /// </summary>
    /**/
    typedef void result_type;

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    task() : _M_unitTask()
    {
        // The default constructor should create a task with a nullptr impl. This is a signal that the
        // task is not usable and should throw if any wait(), get() or then() APIs are used.
    }
#if _MSC_VER < 1800
    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <typeparam name="_Ty">
    ///     The type of the parameter from which the task is to be constructed.
    /// </typeparam>
    /// <param name="_Param">
    ///     The parameter from which the task is to be constructed. This could be a lambda, a function object, a <c>task_completion_event&lt;result_type&gt;</c>
    ///     object, or a Windows::Foundation::IAsyncInfo if you are using tasks in your Windows Store app. The lambda or function
    ///     object should be a type equivalent to <c>std::function&lt;X(void)&gt;</c>, where X can be a variable of type <c>result_type</c>,
    ///     <c>task&lt;result_type&gt;</c>, or a Windows::Foundation::IAsyncInfo in Windows Store apps.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Ty>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        explicit task(_Ty _Param)
    {
        details::_ValidateTaskConstructorArgs<void, _Ty>(_Param);

        _M_unitTask._CreateImpl(Concurrency::cancellation_token::none()._GetImplValue());
        // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of the task constructor.
        _M_unitTask._SetTaskCreationAddressHint(_ReturnAddress());

        _TaskInitMaybeFunctor(_Param, details::_IsCallable<void>(_Param, 0, 0, 0));
    }
#endif
    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <typeparam name="_Ty">
    ///     The type of the parameter from which the task is to be constructed.
    /// </typeparam>
    /// <param name="_Param">
    ///     The parameter from which the task is to be constructed. This could be a lambda, a function object, a <c>task_completion_event&lt;result_type&gt;</c>
    ///     object, or a Windows::Foundation::IAsyncInfo if you are using tasks in your Windows Store app. The lambda or function
    ///     object should be a type equivalent to <c>std::function&lt;X(void)&gt;</c>, where X can be a variable of type <c>result_type</c>,
    ///     <c>task&lt;result_type&gt;</c>, or a Windows::Foundation::IAsyncInfo in Windows Store apps.
    /// </param>
    /// <param name="_Token">
    ///     The cancellation token to associate with this task. A task created without a cancellation token cannot be canceled. It implicitly receives
    ///     the token <c>cancellation_token::none()</c>.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Ty>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
#if _MSC_VER >= 1800
        explicit task(_Ty _Param, const task_options& _TaskOptions = task_options())
#else
        explicit task(_Ty _Param, Concurrency::cancellation_token _CancellationToken)
#endif
    {
            details::_ValidateTaskConstructorArgs<void, _Ty>(_Param);
#if _MSC_VER >= 1800
            _M_unitTask._CreateImpl(_TaskOptions.get_cancellation_token()._GetImplValue(), _TaskOptions.get_scheduler());
#else
            _M_unitTask._CreateImpl(_CancellationToken._GetImplValue());
#endif
            // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of the task constructor.
#if _MSC_VER >= 1800
            _M_unitTask._SetTaskCreationCallstack(details::_get_internal_task_options(_TaskOptions)._M_hasPresetCreationCallstack ? details::_get_internal_task_options(_TaskOptions)._M_presetCreationCallstack : _CAPTURE_CALLSTACK());
#else
            _M_unitTask._SetTaskCreationAddressHint(_ReturnAddress());
#endif
            _TaskInitMaybeFunctor(_Param, details::_IsCallable<void>(_Param, 0, 0, 0));
        }

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    task(const task& _Other) : _M_unitTask(_Other._M_unitTask){}

    /// <summary>
    ///     Constructs a <c>task</c> object.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     The default constructor for a <c>task</c> is only present in order to allow tasks to be used within containers.
    ///     A default constructed task cannot be used until you assign a valid task to it. Methods such as <c>get</c>, <c>wait</c> or <c>then</c>
    ///     will throw an <see cref="invalid_argument Class">invalid_argument</see> exception when called on a default constructed task.
    ///     <para>A task that is created from a <c>task_completion_event</c> will complete (and have its continuations scheduled) when the task
    ///     completion event is set.</para>
    ///     <para>The version of the constructor that takes a cancellation token creates a task that can be canceled using the
    ///     <c>cancellation_token_source</c> the token was obtained from. Tasks created without a cancellation token are not cancelable.</para>
    ///     <para>Tasks created from a <c>Windows::Foundation::IAsyncInfo</c> interface or a lambda that returns an <c>IAsyncInfo</c> interface
    ///     reach their terminal state when the enclosed Windows Runtime asynchronous operation or action completes. Similarly, tasks created
    ///     from a lamda that returns a <c>task&lt;result_type&gt;</c> reach their terminal state when the inner task reaches its terminal state,
    ///     and not when the lamda returns.</para>
    ///     <para><c>task</c> behaves like a smart pointer and is safe to pass around by value. It can be accessed by multiple threads
    ///     without the need for locks.</para>
    ///     <para>The constructor overloads that take a Windows::Foundation::IAsyncInfo interface or a lambda returning such an interface, are only available
    ///     to Windows Store apps.</para>
    ///     <para>For more information, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    task(task&& _Other) : _M_unitTask(std::move(_Other._M_unitTask)) {}

    /// <summary>
    ///     Replaces the contents of one <c>task</c> object with another.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     As <c>task</c> behaves like a smart pointer, after a copy assignment, this <c>task</c> objects represents the same
    ///     actual task as <paramref name="_Other"/> does.
    /// </remarks>
    /**/
    task& operator=(const task& _Other)
    {
        if (this != &_Other)
        {
            _M_unitTask = _Other._M_unitTask;
        }
        return *this;
    }

    /// <summary>
    ///     Replaces the contents of one <c>task</c> object with another.
    /// </summary>
    /// <param name="_Other">
    ///     The source <c>task</c> object.
    /// </param>
    /// <remarks>
    ///     As <c>task</c> behaves like a smart pointer, after a copy assignment, this <c>task</c> objects represents the same
    ///     actual task as <paramref name="_Other"/> does.
    /// </remarks>
    /**/
    task& operator=(task&& _Other)
    {
        if (this != &_Other)
        {
            _M_unitTask = std::move(_Other._M_unitTask);
        }
        return *this;
    }
#if _MSC_VER < 1800
    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        auto then(const _Function& _Func) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        auto _ContinuationTask = _M_unitTask._ThenImpl<void, _Function>(_Func, nullptr, task_continuation_context::use_default());
        // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
        _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
        return _ContinuationTask;
    }
#endif
    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <param name="_CancellationToken">
    ///     The cancellation token to associate with the continuation task. A continuation task that is created without a cancellation token will inherit
    ///     the token of its antecedent task.
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
#if _MSC_VER >= 1800
    auto then(const _Function& _Func, task_options _TaskOptions = task_options()) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        details::_get_internal_task_options(_TaskOptions)._set_creation_callstack(_CAPTURE_CALLSTACK());
        return _M_unitTask._ThenImpl<void, _Function>(_Func, _TaskOptions);
    }
#else
    auto then(const _Function& _Func, Concurrency::cancellation_token _CancellationToken) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        auto _ContinuationTask = _M_unitTask._ThenImpl<void, _Function>(_Func, _CancellationToken._GetImplValue(), task_continuation_context::use_default());
        // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
        _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
        return _ContinuationTask;
    }
    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <param name="_ContinuationContext">
    ///     A variable that specifies where the continuation should execute. This variable is only useful when used in a
    ///     Windows Store app. For more information, see <see cref="task_continuation_context Class">task_continuation_context</see>
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
        auto then(const _Function& _Func, task_continuation_context _ContinuationContext) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        auto _ContinuationTask = _M_unitTask._ThenImpl<void, _Function>(_Func, nullptr, _ContinuationContext);
        // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
        _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
        return _ContinuationTask;

    }
#endif
    /// <summary>
    ///     Adds a continuation task to this task.
    /// </summary>
    /// <typeparam name="_Function">
    ///     The type of the function object that will be invoked by this task.
    /// </typeparam>
    /// <param name="_Func">
    ///     The continuation function to execute when this task completes. This continuation function must take as input
    ///     a variable of either <c>result_type</c> or <c>task&lt;result_type&gt;</c>, where <c>result_type</c> is the type
    ///     of the result this task produces.
    /// </param>
    /// <param name="_CancellationToken">
    ///     The cancellation token to associate with the continuation task. A continuation task that is created without a cancellation token will inherit
    ///     the token of its antecedent task.
    /// </param>
    /// <param name="_ContinuationContext">
    ///     A variable that specifies where the continuation should execute. This variable is only useful when used in a
    ///     Windows Store app. For more information, see <see cref="task_continuation_context Class">task_continuation_context</see>
    /// </param>
    /// <returns>
    ///     The newly created continuation task. The result type of the returned task is determined by what <paramref name="_Func"/> returns.
    /// </returns>
    /// <remarks>
    ///     The overloads of <c>then</c> that take a lambda or functor that returns a Windows::Foundation::IAsyncInfo interface, are only available
    ///     to Windows Store apps.
    ///     <para>For more information on how to use task continuations to compose asynchronous work, see <see cref="Task Parallelism (Concurrency Runtime)"/>.</para>
    /// </remarks>
    /**/
    template<typename _Function>
    __declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
#if _MSC_VER >= 1800
    auto then(const _Function& _Func, Concurrency::cancellation_token _CancellationToken, task_continuation_context _ContinuationContext) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        task_options _TaskOptions(_CancellationToken, _ContinuationContext);
        details::_get_internal_task_options(_TaskOptions)._set_creation_callstack(_CAPTURE_CALLSTACK());
        return _M_unitTask._ThenImpl<void, _Function>(_Func, _TaskOptions);
    }
#else
    auto then(const _Function& _Func, Concurrency::cancellation_token _CancellationToken, task_continuation_context _ContinuationContext) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        auto _ContinuationTask = _M_unitTask._ThenImpl<void, _Function>(_Func, _CancellationToken._GetImplValue(), _ContinuationContext);
        // Do not move the next line out of this function. It is important that _ReturnAddress() evaluate to the the call site of then.
        _ContinuationTask._SetTaskCreationAddressHint(_ReturnAddress());
        return _ContinuationTask;
    }
#endif

    /// <summary>
    ///     Waits for this task to reach a terminal state. It is possible for <c>wait</c> to execute the task inline, if all of the tasks
    ///     dependencies are satisfied, and it has not already been picked up for execution by a background worker.
    /// </summary>
    /// <returns>
    ///     A <c>task_status</c> value which could be either <c>completed</c> or <c>canceled</c>. If the task encountered an exception
    ///     during execution, or an exception was propagated to it from an antecedent task, <c>wait</c> will throw that exception.
    /// </returns>
    /**/
    task_status wait() const
    {
        return _M_unitTask.wait();
    }

    /// <summary>
    ///     Returns the result this task produced. If the task is not in a terminal state, a call to <c>get</c> will wait for the task to
    ///     finish. This method does not return a value when called on a task with a <c>result_type</c> of <c>void</c>.
    /// </summary>
    /// <remarks>
    ///     If the task is canceled, a call to <c>get</c> will throw a <see cref="task_canceled Class">task_canceled</see> exception. If the task
    ///     encountered an different exception or an exception was propagated to it from an antecedent task, a call to <c>get</c> will throw that exception.
    /// </remarks>
    /**/
    void get() const
    {
        _M_unitTask.get();
    }
#if _MSC_VER >= 1800

    /// <summary>
    ///     Determines if the task is completed.
    /// </summary>
    /// <returns>
    ///     True if the task has completed, false otherwise.
    /// </returns>
    /// <remarks>
    ///     The function returns true if the task is completed or canceled (with or without user exception).
    /// </remarks>
    bool is_done() const
    {
        return _M_unitTask.is_done();
    }

    /// <summary>
    ///     Returns the scheduler for this task
    /// </summary>
    /// <returns>
    ///     A pointer to the scheduler
    /// </returns>
    Concurrency::scheduler_ptr scheduler() const
    {
        return _M_unitTask.scheduler();
    }
#endif
    /// <summary>
    ///     Determines whether the task unwraps a Windows Runtime <c>IAsyncInfo</c> interface or is descended from such a task.
    /// </summary>
    /// <returns>
    ///     <c>true</c> if the task unwraps an <c>IAsyncInfo</c> interface or is descended from such a task, <c>false</c> otherwise.
    /// </returns>
    /**/
    bool is_apartment_aware() const
    {
        return _M_unitTask.is_apartment_aware();
    }

    /// <summary>
    ///     Determines whether two <c>task</c> objects represent the same internal task.
    /// </summary>
    /// <returns>
    ///     <c>true</c> if the objects refer to the same underlying task, and <c>false</c> otherwise.
    /// </returns>
    /**/
    bool operator==(const task<void>& _Rhs) const
    {
        return (_M_unitTask == _Rhs._M_unitTask);
    }

    /// <summary>
    ///     Determines whether two <c>task</c> objects represent different internal tasks.
    /// </summary>
    /// <returns>
    ///     <c>true</c> if the objects refer to different underlying tasks, and <c>false</c> otherwise.
    /// </returns>
    /**/
    bool operator!=(const task<void>& _Rhs) const
    {
        return !operator==(_Rhs);
    }

    /// <summary>
    ///     Create an underlying task implementation.
    /// </summary>
#if _MSC_VER >= 1800
    void _CreateImpl(Concurrency::details::_CancellationTokenState * _Ct, Concurrency::scheduler_ptr _Scheduler)
    {
        _M_unitTask._CreateImpl(_Ct, _Scheduler);
    }
#else
    void _CreateImpl(Concurrency::details::_CancellationTokenState * _Ct)
    {
        _M_unitTask._CreateImpl(_Ct);
    }
#endif

    /// <summary>
    ///     Return the underlying implementation for this task.
    /// </summary>
    const details::_Task_ptr<details::_Unit_type>::_Type & _GetImpl() const
    {
        return _M_unitTask._M_Impl;
    }

    /// <summary>
    ///     Set the implementation of the task to be the supplied implementaion.
    /// </summary>
    void _SetImpl(const details::_Task_ptr<details::_Unit_type>::_Type & _Impl)
    {
        _M_unitTask._SetImpl(_Impl);
    }

    /// <summary>
    ///     Set the implementation of the task to be the supplied implementaion using a move instead of a copy.
    /// </summary>
    void _SetImpl(details::_Task_ptr<details::_Unit_type>::_Type && _Impl)
    {
        _M_unitTask._SetImpl(std::move(_Impl));
    }

    /// <summary>
    ///     Sets a property determining whether the task is apartment aware.
    /// </summary>
    void _SetAsync(bool _Async = true)
    {
        _M_unitTask._SetAsync(_Async);
    }

    /// <summary>
    ///     Sets a field in the task impl to the return address for calls to the task constructors and the then method.
    /// </summary>
#if _MSC_VER >= 1800
    void _SetTaskCreationCallstack(const details::_TaskCreationCallstack &_callstack)
    {
        _M_unitTask._SetTaskCreationCallstack(_callstack);
    }
#else
    void _SetTaskCreationAddressHint(void* _Address)
    {
        _M_unitTask._SetTaskCreationAddressHint(_Address);
    }
#endif

    /// <summary>
    ///     An internal version of then that takes additional flags and executes the continuation inline. Used for runtime internal continuations only.
    /// </summary>
    template<typename _Function>
#if _MSC_VER >= 1800
    auto _Then(const _Function& _Func, Concurrency::details::_CancellationTokenState *_PTokenState,
        details::_TaskInliningMode _InliningMode = Concurrency::details::_ForceInline) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        // inherit from antecedent
        auto _Scheduler = _GetImpl()->_GetScheduler();

        return _M_unitTask._ThenImpl<void, _Function>(_Func, _PTokenState, task_continuation_context::use_default(), _Scheduler, _CAPTURE_CALLSTACK(), _InliningMode);
    }
#else
    auto _Then(const _Function& _Func, Concurrency::details::_CancellationTokenState *_PTokenState,
        bool _Aggregating, details::_TaskInliningMode _InliningMode = Concurrency::details::_ForceInline) const -> typename details::_ContinuationTypeTraits<_Function, void>::_TaskOfType
    {
        return _M_unitTask._ThenImpl<void, _Function>(_Func, _PTokenState, task_continuation_context::use_default(), _Aggregating, _InliningMode);
    }
#endif

private:
    template <typename T> friend class task;
    template <typename T> friend class task_completion_event;

    /// <summary>
    ///     Initializes a task using a task completion event.
    /// </summary>
    void _TaskInitNoFunctor(task_completion_event<void>& _Event)
    {
        _M_unitTask._TaskInitNoFunctor(_Event._M_unitEvent);
    }
    /// <summary>
    ///     Initializes a task using an asynchronous action IAsyncAction*
    /// </summary>
    void _TaskInitNoFunctor(ABI::Windows::Foundation::IAsyncAction* _AsyncAction)
    {
        _M_unitTask._TaskInitAsyncOp<details::_Unit_type>(Microsoft::WRL::Make<details::_IAsyncActionToAsyncOperationConverter>(_AsyncAction).Get());
    }

    /// <summary>
    ///     Initializes a task using an asynchronous action with progress IAsyncActionWithProgress<_P>*
    /// </summary>
    template<typename _P>
    void _TaskInitNoFunctor(ABI::Windows::Foundation::IAsyncActionWithProgress<_P>* _AsyncActionWithProgress)
    {
        _M_unitTask._TaskInitAsyncOp<details::_Unit_type>(Microsoft::WRL::Make<details::_IAsyncActionWithProgressToAsyncOperationConverter<_P>>(_AsyncActionWithProgress).Get());
    }
    /// <summary>
    ///     Initializes a task using a callable object.
    /// </summary>
    template<typename _Function>
    void _TaskInitMaybeFunctor(_Function & _Func, std::true_type)
    {
        _M_unitTask._TaskInitWithFunctor<void, _Function>(_Func);
    }

    /// <summary>
    ///     Initializes a task using a non-callable object.
    /// </summary>
    template<typename _T>
    void _TaskInitMaybeFunctor(_T & _Param, std::false_type)
    {
        _TaskInitNoFunctor(_Param);
    }

    // The void task contains a task of a dummy type so common code can be used for tasks with void and non-void results.
    task<details::_Unit_type> _M_unitTask;
};

namespace details
{

    /// <summary>
    ///   The following type traits are used for the create_task function.
    /// </summary>

    // Unwrap task<T>
    template<typename _Ty>
    _Ty _GetUnwrappedType(task<_Ty>);

    // Unwrap all supported types
    template<typename _Ty>
    auto _GetUnwrappedReturnType(_Ty _Arg, int) -> decltype(_GetUnwrappedType(_Arg));
    // fallback
    template<typename _Ty>
    _Ty _GetUnwrappedReturnType(_Ty, ...);

    /// <summary>
    ///   <c>_GetTaskType</c> functions will retrieve task type <c>T</c> in <c>task[T](Arg)</c>,
    ///   for given constructor argument <c>Arg</c> and its property "callable".
    ///   It will automatically unwrap argument to get the final return type if necessary.
    /// </summary>

    // Non-Callable
    template<typename _Ty>
    _Ty _GetTaskType(task_completion_event<_Ty>, std::false_type);

    // Non-Callable
    template<typename _Ty>
    auto _GetTaskType(_Ty _NonFunc, std::false_type) -> decltype(_GetUnwrappedType(_NonFunc));

    // Callable
    template<typename _Ty>
    auto _GetTaskType(_Ty _Func, std::true_type) -> decltype(_GetUnwrappedReturnType(stdx::declval<_FunctionTypeTraits<_Ty, void>::_FuncRetType>(), 0));

    // Special callable returns void
    void _GetTaskType(std::function<HRESULT()>, std::true_type);
    struct _BadArgType{};

    template<typename _ReturnType, typename _Ty>
    auto _FilterValidTaskType(_Ty _Param, int) -> decltype(_GetTaskType(_Param, _IsCallable<_ReturnType>(_Param, 0, 0, 0)));

    template<typename _ReturnType, typename _Ty>
    _BadArgType _FilterValidTaskType(_Ty _Param, ...);

    template<typename _ReturnType, typename _Ty>
    struct _TaskTypeFromParam
    {
        typedef decltype(_FilterValidTaskType<_ReturnType>(stdx::declval<_Ty>(), 0)) _Type;
    };
}


/// <summary>
///     Creates a PPL <see cref="task Class">task</c> object. <c>create_task</c> can be used anywhere you would have used a task constructor.
///     It is provided mainly for convenience, because it allows use of the <c>auto</c> keyword while creating tasks.
/// </summary>
/// <typeparam name="_Ty">
///     The type of the parameter from which the task is to be constructed.
/// </typeparam>
/// <param name="_Param">
///     The parameter from which the task is to be constructed. This could be a lambda or function object, a <c>task_completion_event</c>
///     object, a different <c>task</c> object, or a Windows::Foundation::IAsyncInfo interface if you are using tasks in your Windows Store app.
/// </param>
/// <returns>
///     A new task of type <c>T</c>, that is inferred from <paramref name="_Param"/>.
/// </returns>
/// <remarks>
///     The first overload behaves like a task constructor that takes a single parameter.
///     <para>The second overload associates the cancellation token provided with the newly created task. If you use this overload you are not
///     allowed to pass in a different <c>task</c> object as the first parameter.</para>
///     <para>The type of the returned task is inferred from the first parameter to the function. If <paramref name="_Param"/> is a <c>task_completion_event&lt;T&gt;</c>,
///     a <c>task&lt;T&gt;</c>, or a functor that returns either type <c>T</c> or <c>task&lt;T&gt;</c>, the type of the created task is <c>task&lt;T&gt;</c>.
///     <para>In a Windows Store app, if <paramref name="_Param"/> is of type Windows::Foundation::IAsyncOperation&ltT&gt^ or
///     Windows::Foundation::IAsyncOperationWithProgress&ltT,P&gt^, or a functor that returns either of those types, the created task will be of type <c>task&lt;T&gt;</c>.
///     If <paramref name="_Param"/> is of type Windows::Foundation::IAsyncAction^ or Windows::Foundation::IAsyncActionWithProgress&lt;P&gt;^, or a functor
///     that returns either of those types, the created task will have type <c>task&lt;void&gt;</c>.</para>
/// </remarks>
/// <seealso cref="task Class"/>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType, typename _Ty>
__declspec(noinline)
#if _MSC_VER >= 1800
auto create_task(_Ty _Param, task_options _TaskOptions = task_options()) -> task<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type>
#else
auto create_task(_Ty _Param) -> task<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type>
#endif
{
    static_assert(!std::is_same<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type, details::_BadArgType>::value,
        "incorrect argument for create_task; can be a callable object, an asynchronous operation, or a task_completion_event"
        );
#if _MSC_VER >= 1800
    details::_get_internal_task_options(_TaskOptions)._set_creation_callstack(_CAPTURE_CALLSTACK());
    task<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type> _CreatedTask(_Param, _TaskOptions);
#else
    task<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type> _CreatedTask(_Param);
    // Ideally we would like to forceinline create_task, but __forceinline does nothing on debug builds. Therefore, we ask for no inlining
    // and overwrite the creation address hint set by the task constructor. DO NOT REMOVE this next line from create_task. It is
    // essential that _ReturnAddress() evaluate to the instruction right after the call to create_task in client code.
    _CreatedTask._SetTaskCreationAddressHint(_ReturnAddress());
#endif
    return _CreatedTask;
}

/// <summary>
///     Creates a PPL <see cref="task Class">task</c> object. <c>create_task</c> can be used anywhere you would have used a task constructor.
///     It is provided mainly for convenience, because it allows use of the <c>auto</c> keyword while creating tasks.
/// </summary>
/// <typeparam name="_Ty">
///     The type of the parameter from which the task is to be constructed.
/// </typeparam>
/// <param name="_Param">
///     The parameter from which the task is to be constructed. This could be a lambda or function object, a <c>task_completion_event</c>
///     object, a different <c>task</c> object, or a Windows::Foundation::IAsyncInfo interface if you are using tasks in your Windows Store app.
/// </param>
/// <param name="_Token">
///     The cancellation token to associate with the task. When the source for this token is canceled, cancellation will be requested on the task.
/// </param>
/// <returns>
///     A new task of type <c>T</c>, that is inferred from <paramref name="_Param"/>.
/// </returns>
/// <remarks>
///     The first overload behaves like a task constructor that takes a single parameter.
///     <para>The second overload associates the cancellation token provided with the newly created task. If you use this overload you are not
///     allowed to pass in a different <c>task</c> object as the first parameter.</para>
///     <para>The type of the returned task is inferred from the first parameter to the function. If <paramref name="_Param"/> is a <c>task_completion_event&lt;T&gt;</c>,
///     a <c>task&lt;T&gt;</c>, or a functor that returns either type <c>T</c> or <c>task&lt;T&gt;</c>, the type of the created task is <c>task&lt;T&gt;</c>.
///     <para>In a Windows Store app, if <paramref name="_Param"/> is of type Windows::Foundation::IAsyncOperation&ltT&gt^ or
///     Windows::Foundation::IAsyncOperationWithProgress&ltT,P&gt^, or a functor that returns either of those types, the created task will be of type <c>task&lt;T&gt;</c>.
///     If <paramref name="_Param"/> is of type Windows::Foundation::IAsyncAction^ or Windows::Foundation::IAsyncActionWithProgress&lt;P&gt;^, or a functor
///     that returns either of those types, the created task will have type <c>task&lt;void&gt;</c>.</para>
/// </remarks>
/// <seealso cref="task Class"/>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
#if _MSC_VER >= 1800
template<typename _ReturnType>
__declspec(noinline)
task<_ReturnType> create_task(const task<_ReturnType>& _Task)
{
    task<_ReturnType> _CreatedTask(_Task);
    return _CreatedTask;
}
#else
template<typename _ReturnType, typename _Ty>
__declspec(noinline)
auto create_task(_Ty _Param, Concurrency::cancellation_token _Token) -> task<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type>
{
    static_assert(!std::is_same<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type, details::_BadArgType>::value,
        "incorrect argument for create_task; can be a callable object, an asynchronous operation, or a task_completion_event"
        );
    task<typename details::_TaskTypeFromParam<_ReturnType, _Ty>::_Type> _CreatedTask(_Param, _Token);
    // Ideally we would like to forceinline create_task, but __forceinline does nothing on debug builds. Therefore, we ask for no inlining
    // and overwrite the creation address hint set by the task constructor. DO NOT REMOVE this next line from create_task. It is
    // essential that _ReturnAddress() evaluate to the instruction right after the call to create_task in client code.
    _CreatedTask._SetTaskCreationAddressHint(_ReturnAddress());
    return _CreatedTask;
}
#endif

namespace details
{
    template<typename _T>
    task<typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperation<_T>*>()))>::type> _To_task_helper(ABI::Windows::Foundation::IAsyncOperation<_T>* op)
    {
        return task<_T>(op);
    }

    template<typename _T, typename _Progress>
    task<typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationWithProgressSelector(stdx::declval<ABI::Windows::Foundation::IAsyncOperationWithProgress<_T, _Progress>*>()))>::type> _To_task_helper(ABI::Windows::Foundation::IAsyncOperationWithProgress<_T, _Progress>* op)
    {
        return task<_T>(op);
    }

    inline task<void> _To_task_helper(ABI::Windows::Foundation::IAsyncAction* op)
    {
        return task<void>(op);
    }

    template<typename _Progress>
    task<void> _To_task_helper(ABI::Windows::Foundation::IAsyncActionWithProgress<_Progress>* op)
    {
        return task<void>(op);
    }

    template<typename _ProgressType>
    class _ProgressDispatcherBase
    {
    public:

        virtual ~_ProgressDispatcherBase()
        {
        }

        virtual void _Report(const _ProgressType& _Val) = 0;
    };

    template<typename _ProgressType, typename _ClassPtrType>
    class _ProgressDispatcher : public _ProgressDispatcherBase<_ProgressType>
    {
    public:

        virtual ~_ProgressDispatcher()
        {
        }

        _ProgressDispatcher(_ClassPtrType _Ptr) : _M_ptr(_Ptr)
        {
        }

        virtual void _Report(const _ProgressType& _Val)
        {
            _M_ptr->_FireProgress(_Val);
        }

    private:

        _ClassPtrType _M_ptr;
    };
} // namespace details


/// <summary>
///     The progress reporter class allows reporting progress notifications of a specific type. Each progress_reporter object is bound
///     to a particular asynchronous action or operation.
/// </summary>
/// <typeparam name="_ProgressType">
///     The payload type of each progress notification reported through the progress reporter.
/// </typeparam>
/// <remarks>
///     This type is only available to Windows Store apps.
/// </remarks>
/// <seealso cref="create_async Function"/>
/**/
template<typename _ProgressType>
class progress_reporter
{
    typedef std::shared_ptr<details::_ProgressDispatcherBase<_ProgressType>> _PtrType;

public:

    /// <summary>
    ///     Sends a progress report to the asynchronous action or operation to which this progress reporter is bound.
    /// </summary>
    /// <param name="_Val">
    ///     The payload to report through a progress notification.
    /// </param>
    /**/
    void report(const _ProgressType& _Val) const
    {
        _M_dispatcher->_Report(_Val);
    }

    template<typename _ClassPtrType>
    static progress_reporter _CreateReporter(_ClassPtrType _Ptr)
    {
        progress_reporter _Reporter;
        details::_ProgressDispatcherBase<_ProgressType> *_PDispatcher = new details::_ProgressDispatcher<_ProgressType, _ClassPtrType>(_Ptr);
        _Reporter._M_dispatcher = _PtrType(_PDispatcher);
        return _Reporter;
    }
    progress_reporter() {}

private:
    progress_reporter(details::_ProgressReporterCtorArgType);

    _PtrType _M_dispatcher;
};

namespace details
{
    //
    // maps internal definitions for AsyncStatus and defines states that are not client visible
    //
    enum _AsyncStatusInternal
    {
        _AsyncCreated = -1,  // externally invisible
        // client visible states (must match AsyncStatus exactly)
        _AsyncStarted = ABI::Windows::Foundation::AsyncStatus::Started, // 0
        _AsyncCompleted = ABI::Windows::Foundation::AsyncStatus::Completed, // 1
        _AsyncCanceled = ABI::Windows::Foundation::AsyncStatus::Canceled, // 2
        _AsyncError = ABI::Windows::Foundation::AsyncStatus::Error, // 3
        // non-client visible internal states
        _AsyncCancelPending,
        _AsyncClosed,
        _AsyncUndefined
    };

    //
    // designates whether the "GetResults" method returns a single result (after complete fires) or multiple results
    // (which are progressively consumable between Start state and before Close is called)
    //
    enum _AsyncResultType
    {
        SingleResult = 0x0001,
        MultipleResults = 0x0002
    };

    template<typename _T>
    struct _ProgressTypeTraits
    {
        static const bool _TakesProgress = false;
        typedef void _ProgressType;
    };

    template<typename _T>
    struct _ProgressTypeTraits<progress_reporter<_T>>
    {
        static const bool _TakesProgress = true;
        typedef typename _T _ProgressType;
    };

    template<typename _T, bool bTakesToken = std::is_same<_T, Concurrency::cancellation_token>::value, bool bTakesProgress = _ProgressTypeTraits<_T>::_TakesProgress>
    struct _TokenTypeTraits
    {
        static const bool _TakesToken = false;
        typedef typename _T _ReturnType;
    };

    template<typename _T>
    struct _TokenTypeTraits<_T, false, true>
    {
        static const bool _TakesToken = false;
        typedef void _ReturnType;
    };

    template<typename _T>
    struct _TokenTypeTraits<_T, true, false>
    {
        static const bool _TakesToken = true;
        typedef void _ReturnType;
    };

    template<typename _T, size_t count = _FunctorTypeTraits<_T>::_ArgumentCount>
    struct _CAFunctorOptions
    {
        static const bool _TakesProgress = false;
        static const bool _TakesToken = false;
        typedef void _ProgressType;
        typedef void _ReturnType;
    };

    template<typename _T>
    struct _CAFunctorOptions<_T, 1>
    {
    private:

        typedef typename _FunctorTypeTraits<_T>::_Argument1Type _Argument1Type;

    public:

        static const bool _TakesProgress = _ProgressTypeTraits<_Argument1Type>::_TakesProgress;
        static const bool _TakesToken = _TokenTypeTraits<_Argument1Type>::_TakesToken;
        typedef typename _ProgressTypeTraits<_Argument1Type>::_ProgressType _ProgressType;
        typedef typename _TokenTypeTraits<_Argument1Type>::_ReturnType _ReturnType;
    };

    template<typename _T>
    struct _CAFunctorOptions<_T, 2>
    {
    private:

        typedef typename _FunctorTypeTraits<_T>::_Argument1Type _Argument1Type;
        typedef typename _FunctorTypeTraits<_T>::_Argument2Type _Argument2Type;

    public:

        static const bool _TakesProgress = _ProgressTypeTraits<_Argument1Type>::_TakesProgress;
        static const bool _TakesToken = !_TakesProgress ? true : _TokenTypeTraits<_Argument2Type>::_TakesToken;
        typedef typename _ProgressTypeTraits<_Argument1Type>::_ProgressType _ProgressType;
        typedef typename _TokenTypeTraits<_Argument2Type>::_ReturnType _ReturnType;
    };

    template<typename _T>
    struct _CAFunctorOptions<_T, 3>
    {
    private:

        typedef typename _FunctorTypeTraits<_T>::_Argument1Type _Argument1Type;

    public:

        static const bool _TakesProgress = true;
        static const bool _TakesToken = true;
        typedef typename _ProgressTypeTraits<_Argument1Type>::_ProgressType _ProgressType;
        typedef typename _FunctorTypeTraits<_T>::_Argument3Type _ReturnType;
    };

    class _Zip
    {
    };

    // ***************************************************************************
    // Async Operation Task Generators
    //

    //
    // Functor returns an IAsyncInfo - result needs to be wrapped in a task:
    //
    template<typename _AsyncSelector, typename _ReturnType>
    struct _SelectorTaskGenerator
    {
#if _MSC_VER >= 1800
        template<typename _Function>
        static task<_ReturnType> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>(_Func(_pRet), _taskOptinos);
        }

        template<typename _Function>
        static task<_ReturnType> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>(_Func(_Cts.get_token(), _pRet), _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>(_Func(_Progress, _pRet), _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>(_Func(_Progress, _Cts.get_token(), _pRet), _taskOptinos);
        }
#else
        template<typename _Function>
        static task<_ReturnType> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>(_Func(_pRet), _Cts.get_token());
        }

        template<typename _Function>
        static task<_ReturnType> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>(_Func(_Cts.get_token(), _pRet), _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>(_Func(_Progress, _pRet), _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>(_Func(_Progress, _Cts.get_token(), _pRet), _Cts.get_token());
        }
#endif
    };

    template<typename _AsyncSelector>
    struct _SelectorTaskGenerator<_AsyncSelector, void>
    {
#if _MSC_VER >= 1800
        template<typename _Function>
        static task<void> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>(_Func(), _taskOptinos);
        }

        template<typename _Function>
        static task<void> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>(_Func(_Cts.get_token()), _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>(_Func(_Progress), _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>(_Func(_Progress, _Cts.get_token()), _taskOptinos);
        }
#else
        template<typename _Function>
        static task<void> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>(_Func(), _Cts.get_token());
        }

        template<typename _Function>
        static task<void> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>(_Func(_Cts.get_token()), _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>(_Func(_Progress), _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>(_Func(_Progress, _Cts.get_token()), _Cts.get_token());
        }
#endif
    };

#if _MSC_VER < 1800
    // For create_async lambdas that return a (non-task) result, we oversubscriber the current task for the duration of the
    // lambda.
    struct _Task_generator_oversubscriber
    {
        _Task_generator_oversubscriber()
        {
            Concurrency::details::_Context::_Oversubscribe(true);
        }

        ~_Task_generator_oversubscriber()
        {
            Concurrency::details::_Context::_Oversubscribe(false);
        }
    };
#endif

    //
    // Functor returns a result - it needs to be wrapped in a task:
    //
    template<typename _ReturnType>
    struct _SelectorTaskGenerator<details::_TypeSelectorNoAsync, _ReturnType>
    {
#if _MSC_VER >= 1800

#pragma warning(push)
#pragma warning(disable: 4702)
        template<typename _Function>
        static task<_ReturnType> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                HRESULT hr = _Func(_pRet);
                retVal = _pRet;
                return hr;
            }, _taskOptinos);
        }
#pragma warning(pop)

        template<typename _Function>
        static task<_ReturnType> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                HRESULT hr = _Func(_Cts.get_token(), _pRet);
                retVal = _pRet;
                return hr;
            }, _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                HRESULT hr = _Func(_Progress, _pRet);
                retVal = _pRet;
                return hr;
            }, _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                HRESULT hr = _Func(_Progress, _Cts.get_token(), _pRet);
                retVal = _pRet;
                return hr;
            }, _taskOptinos);
        }
#else
        template<typename _Function>
        static task<_ReturnType> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                HRESULT hr = _Func(_pRet);
                retVal = _pRet;
                return hr;
            }, _Cts.get_token());
        }

        template<typename _Function>
        static task<_ReturnType> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                HRESULT hr = _Func(_Cts.get_token(), _pRet);
                retVal = _pRet;
                return hr;
            }, _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                HRESULT hr = _Func(_Progress, _pRet);
                retVal = _pRet;
                return hr;
            }, _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<_ReturnType> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return task<_ReturnType>([=](_ReturnType* retVal) -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                HRESULT hr = _Func(_Progress, _Cts.get_token(), _pRet);
                retVal = _pRet;
                return hr;
            }, _Cts.get_token());
        }
#endif
    };

    template<>
    struct _SelectorTaskGenerator<details::_TypeSelectorNoAsync, void>
    {
#if _MSC_VER >= 1800
        template<typename _Function>
        static task<void> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>([=]() -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                return _Func();
            }, _taskOptinos);
        }

        template<typename _Function>
        static task<void> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>([=]() -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                return _Func(_Cts.get_token());
            }, _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>([=]() -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                return _Func(_Progress);
            }, _taskOptinos);
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            task_options _taskOptinos(_Cts.get_token());
            details::_get_internal_task_options(_taskOptinos)._set_creation_callstack(_callstack);
            return task<void>([=]() -> HRESULT {
                Concurrency::details::_Task_generator_oversubscriber_t _Oversubscriber;
                (_Oversubscriber);
                return _Func(_Progress, _Cts.get_token());
            }, _taskOptinos);
        }
#else
        template<typename _Function>
        static task<void> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>([=]() -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                return _Func();
            }, _Cts.get_token());
        }

        template<typename _Function>
        static task<void> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>([=]() -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                return _Func(_Cts.get_token());
            }, _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>([=]() -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                return _Func(_Progress);
            }, _Cts.get_token());
        }

        template<typename _Function, typename _ProgressObject>
        static task<void> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts)
        {
            return task<void>([=]() -> HRESULT {
                _Task_generator_oversubscriber _Oversubscriber;
                return _Func(_Progress, _Cts.get_token());
            }, _Cts.get_token());
        }
#endif
    };

    //
    // Functor returns a task - the task can directly be returned:
    //
    template<typename _ReturnType>
    struct _SelectorTaskGenerator<details::_TypeSelectorAsyncTask, _ReturnType>
    {
        template<typename _Function>
#if _MSC_VER >= 1800
        static task<_ReturnType> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
#else
        static task<_ReturnType> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
#endif
        {
            task<_ReturnType> _task;
            _Func(&_task);
            return _task;
        }

        template<typename _Function>
#if _MSC_VER >= 1800
        static task<_ReturnType> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
#else
        static task<_ReturnType> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
#endif
        {
            task<_ReturnType> _task;
            _Func(_Cts.get_token(), &_task);
            return _task;
        }

        template<typename _Function, typename _ProgressObject>
#if _MSC_VER >= 1800
        static task<_ReturnType> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
#else
        static task<_ReturnType> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
#endif
        {
            task<_ReturnType> _task;
            _Func(_Progress, &_task);
            return _task;
        }

        template<typename _Function, typename _ProgressObject>
#if _MSC_VER >= 1800
        static task<_ReturnType> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
#else
        static task<_ReturnType> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
#endif
        {
            task<_ReturnType> _task;
            _Func(_Progress, _Cts.get_token(), &_task);
            return _task;
        }
    };

    template<>
    struct _SelectorTaskGenerator<details::_TypeSelectorAsyncTask, void>
    {
        template<typename _Function>
#if _MSC_VER >= 1800
        static task<void> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
#else
        static task<void> _GenerateTask_0(const _Function& _Func, Concurrency::cancellation_token_source _Cts)
#endif
        {
            task<void> _task;
            _Func(&_task);
            return _task;
        }

        template<typename _Function>
#if _MSC_VER >= 1800
        static task<void> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
#else
        static task<void> _GenerateTask_1C(const _Function& _Func, Concurrency::cancellation_token_source _Cts)
#endif
        {
            task<void> _task;
            _Func(_Cts.get_token(), &_task);
            return _task;
        }

        template<typename _Function, typename _ProgressObject>
#if _MSC_VER >= 1800
        static task<void> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
#else
        static task<void> _GenerateTask_1P(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts)
#endif
        {
            task<void> _task;
            _Func(_Progress, &_task);
            return _task;
        }

        template<typename _Function, typename _ProgressObject>
#if _MSC_VER >= 1800
        static task<void> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
#else
        static task<void> _GenerateTask_2PC(const _Function& _Func, const _ProgressObject& _Progress, Concurrency::cancellation_token_source _Cts)
#endif
        {
            task<void> _task;
            _Func(_Progress, _Cts.get_token(), &_task);
            return _task;
        }
    };

    template<typename _Generator, bool _TakesToken, bool TakesProgress>
    struct _TaskGenerator
    {
    };

    template<typename _Generator>
    struct _TaskGenerator<_Generator, false, false>
    {
#if _MSC_VER >= 1800
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _callstack))
        {
            (void)_Ptr;
            return _Generator::_GenerateTask_0(_Func, _Cts, _callstack);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet, _callstack))
        {
            return _Generator::_GenerateTask_0(_Func, _Cts, _pRet, _callstack);
        }
#else
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts))
        {
            (void)_Ptr;
            return _Generator::_GenerateTask_0(_Func, _Cts);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet))
        {
            return _Generator::_GenerateTask_0(_Func, _Cts, _pRet);
        }
#endif
    };

    template<typename _Generator>
    struct _TaskGenerator<_Generator, true, false>
    {
#if _MSC_VER >= 1800
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _callstack))
        {
            return _Generator::_GenerateTask_1C(_Func, _Cts, _callstack);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet, _callstack))
        {
            return _Generator::_GenerateTask_1C(_Func, _Cts, _pRet, _callstack);
        }
#else
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts))
        {
            return _Generator::_GenerateTask_1C(_Func, _Cts);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet))
        {
            return _Generator::_GenerateTask_1C(_Func, _Cts, _pRet);
        }
#endif
    };

    template<typename _Generator>
    struct _TaskGenerator<_Generator, false, true>
    {
#if _MSC_VER >= 1800
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _callstack))
        {
            return _Generator::_GenerateTask_1P(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts, _callstack);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet, _callstack))
        {
            return _Generator::_GenerateTask_1P(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts, _pRet, _callstack);
        }
#else
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts))
        {
            return _Generator::_GenerateTask_1P(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet))
        {
            return _Generator::_GenerateTask_1P(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts, _pRet);
        }
#endif
    };

    template<typename _Generator>
    struct _TaskGenerator<_Generator, true, true>
    {
#if _MSC_VER >= 1800
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _callstack))
        {
            return _Generator::_GenerateTask_2PC(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts, _callstack);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet, _callstack))
        {
            return _Generator::_GenerateTask_2PC(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts, _pRet, _callstack);
        }
#else
        template<typename _Function, typename _ClassPtr, typename _ProgressType>
        static auto _GenerateTaskNoRet(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts))
        {
            return _Generator::_GenerateTask_2PC(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts);
        }

        template<typename _Function, typename _ClassPtr, typename _ProgressType, typename _ReturnType>
        static auto _GenerateTask(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
            -> decltype(_Generator::_GenerateTask_0(_Func, _Cts, _pRet))
        {
            return _Generator::_GenerateTask_2PC(_Func, progress_reporter<_ProgressType>::_CreateReporter(_Ptr), _Cts, _pRet);
        }
#endif
    };

    // ***************************************************************************
    // Async Operation Attributes Classes
    //
    // These classes are passed through the hierarchy of async base classes in order to hold multiple attributes of a given async construct in
    // a single container. An attribute class must define:
    //
    // Mandatory:
    // -------------------------
    //
    // _AsyncBaseType           : The Windows Runtime interface which is being implemented.
    // _CompletionDelegateType  : The Windows Runtime completion delegate type for the interface.
    // _ProgressDelegateType    : If _TakesProgress is true, the Windows Runtime progress delegate type for the interface. If it is false, an empty Windows Runtime type.
    // _ReturnType              : The return type of the async construct (void for actions / non-void for operations)
    //
    // _TakesProgress           : An indication as to whether or not
    //
    // _Generate_Task           : A function adapting the user's function into what's necessary to produce the appropriate task
    //
    // Optional:
    // -------------------------
    //

    template<typename _Function, typename _ProgressType, typename _ReturnType, typename _TaskTraits, bool _TakesToken, bool _TakesProgress>
    struct _AsyncAttributes
    {
    };

    template<typename _Function, typename _ProgressType, typename _ReturnType, typename _TaskTraits, bool _TakesToken>
    struct _AsyncAttributes<_Function, _ProgressType, _ReturnType, _TaskTraits, _TakesToken, true>
    {
        typedef typename ABI::Windows::Foundation::IAsyncOperationWithProgress<_ReturnType, _ProgressType> _AsyncBaseType;
        typedef typename ABI::Windows::Foundation::IAsyncOperationProgressHandler<_ReturnType, _ProgressType> _ProgressDelegateType;
        typedef typename ABI::Windows::Foundation::IAsyncOperationWithProgressCompletedHandler<_ReturnType, _ProgressType> _CompletionDelegateType;
        typedef typename _ReturnType _ReturnType;
        typedef typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationWithProgressSelector(stdx::declval<_AsyncBaseType*>()))>::type _ReturnType_abi;
        typedef typename _ProgressType _ProgressType;
        typedef typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationWithProgressProgressSelector(stdx::declval<_AsyncBaseType*>()))>::type _ProgressType_abi;
        typedef typename _TaskTraits::_AsyncKind _AsyncKind;
        typedef typename _SelectorTaskGenerator<_AsyncKind, _ReturnType> _SelectorTaskGenerator;
        typedef typename _TaskGenerator<_SelectorTaskGenerator, _TakesToken, true> _TaskGenerator;

        static const bool _TakesProgress = true;
        static const bool _TakesToken = _TakesToken;

        template<typename _Function, typename _ClassPtr>
#if _MSC_VER >= 1800
        static task<typename _TaskTraits::_TaskRetType> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType_abi, _ReturnType>(_Func, _Ptr, _Cts, _pRet, _callstack);
        }
#else
        static task<typename _TaskTraits::_TaskRetType> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType_abi, _ReturnType>(_Func, _Ptr, _Cts, _pRet);
        }
#endif
    };

    template<typename _Function, typename _ProgressType, typename _ReturnType, typename _TaskTraits, bool _TakesToken>
    struct _AsyncAttributes<_Function, _ProgressType, _ReturnType, _TaskTraits, _TakesToken, false>
    {
        typedef typename ABI::Windows::Foundation::IAsyncOperation<_ReturnType> _AsyncBaseType;
        typedef _Zip _ProgressDelegateType;
        typedef typename ABI::Windows::Foundation::IAsyncOperationCompletedHandler<_ReturnType> _CompletionDelegateType;
        typedef typename _ReturnType _ReturnType;
        typedef typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncOperationSelector(stdx::declval<_AsyncBaseType*>()))>::type _ReturnType_abi;
        typedef typename _TaskTraits::_AsyncKind _AsyncKind;
        typedef typename _SelectorTaskGenerator<_AsyncKind, _ReturnType> _SelectorTaskGenerator;
        typedef typename _TaskGenerator<_SelectorTaskGenerator, _TakesToken, false> _TaskGenerator;

        static const bool _TakesProgress = false;
        static const bool _TakesToken = _TakesToken;

        template<typename _Function, typename _ClassPtr>
#if _MSC_VER >= 1800
        static task<typename _TaskTraits::_TaskRetType> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType, _ReturnType>(_Func, _Ptr, _Cts, _pRet, _callstack);
        }
#else
        static task<typename _TaskTraits::_TaskRetType> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType, _ReturnType>(_Func, _Ptr, _Cts, _pRet);
        }
#endif
    };

    template<typename _Function, typename _ProgressType, typename _TaskTraits, bool _TakesToken>
    struct _AsyncAttributes<_Function, _ProgressType, void, _TaskTraits, _TakesToken, true>
    {
        typedef typename ABI::Windows::Foundation::IAsyncActionWithProgress<_ProgressType> _AsyncBaseType;
        typedef typename ABI::Windows::Foundation::IAsyncActionProgressHandler<_ProgressType> _ProgressDelegateType;
        typedef typename ABI::Windows::Foundation::IAsyncActionWithProgressCompletedHandler<_ProgressType> _CompletionDelegateType;
        typedef void _ReturnType;
        typedef void _ReturnType_abi;
        typedef typename _ProgressType _ProgressType;
        typedef typename ABI::Windows::Foundation::Internal::GetAbiType<decltype(_UnwrapAsyncActionWithProgressSelector(stdx::declval<_AsyncBaseType*>()))>::type _ProgressType_abi;
        typedef typename _TaskTraits::_AsyncKind _AsyncKind;
        typedef typename _SelectorTaskGenerator<_AsyncKind, _ReturnType> _SelectorTaskGenerator;
        typedef typename _TaskGenerator<_SelectorTaskGenerator, _TakesToken, true> _TaskGenerator;

        static const bool _TakesProgress = true;
        static const bool _TakesToken = _TakesToken;

#if _MSC_VER >= 1800
        template<typename _Function, typename _ClassPtr>
        static task<void> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            return _TaskGenerator::_GenerateTaskNoRet<_Function, _ClassPtr, _ProgressType_abi>(_Func, _Ptr, _Cts, _callstack);
        }
        template<typename _Function, typename _ClassPtr>
        static task<task<void>> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType_abi>(_Func, _Ptr, _Cts, _pRet, _callstack);
        }
#else
        template<typename _Function, typename _ClassPtr>
        static task<void> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts)
        {
            return _TaskGenerator::_GenerateTaskNoRet<_Function, _ClassPtr, _ProgressType_abi>(_Func, _Ptr, _Cts);
        }
        template<typename _Function, typename _ClassPtr>
        static task<task<void>> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType_abi>(_Func, _Ptr, _Cts, _pRet);
        }
#endif
    };

    template<typename _Function, typename _ProgressType, typename _TaskTraits, bool _TakesToken>
    struct _AsyncAttributes<_Function, _ProgressType, void, _TaskTraits, _TakesToken, false>
    {
        typedef typename ABI::Windows::Foundation::IAsyncAction _AsyncBaseType;
        typedef _Zip _ProgressDelegateType;
        typedef typename ABI::Windows::Foundation::IAsyncActionCompletedHandler _CompletionDelegateType;
        typedef void _ReturnType;
        typedef void _ReturnType_abi;
        typedef typename _TaskTraits::_AsyncKind _AsyncKind;
        typedef typename _SelectorTaskGenerator<_AsyncKind, _ReturnType> _SelectorTaskGenerator;
        typedef typename _TaskGenerator<_SelectorTaskGenerator, _TakesToken, false> _TaskGenerator;

        static const bool _TakesProgress = false;
        static const bool _TakesToken = _TakesToken;

#if _MSC_VER >= 1800
        template<typename _Function, typename _ClassPtr>
        static task<void> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, const _TaskCreationCallstack & _callstack)
        {
            return _TaskGenerator::_GenerateTaskNoRet<_Function, _ClassPtr, _ProgressType>(_Func, _Ptr, _Cts, _callstack);
        }
        template<typename _Function, typename _ClassPtr>
        static task<task<void>> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet, const _TaskCreationCallstack & _callstack)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType>(_Func, _Ptr, _Cts, _pRet, _callstack);
        }
#else
        template<typename _Function, typename _ClassPtr>
        static task<void> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts)
        {
            return _TaskGenerator::_GenerateTaskNoRet<_Function, _ClassPtr, _ProgressType>(_Func, _Ptr, _Cts);
        }
        template<typename _Function, typename _ClassPtr>
        static task<task<void>> _Generate_Task(const _Function& _Func, _ClassPtr _Ptr, Concurrency::cancellation_token_source _Cts, _ReturnType* _pRet)
        {
            return _TaskGenerator::_GenerateTask<_Function, _ClassPtr, _ProgressType>(_Func, _Ptr, _Cts, _pRet);
        }
#endif
    };

    template<typename _Function>
    struct _AsyncLambdaTypeTraits
    {
        typedef typename _Unhat<typename _CAFunctorOptions<_Function>::_ReturnType>::_Value _ReturnType;
        typedef typename _FunctorTypeTraits<_Function>::_Argument1Type _Argument1Type;
        typedef typename _CAFunctorOptions<_Function>::_ProgressType _ProgressType;

        static const bool _TakesProgress = _CAFunctorOptions<_Function>::_TakesProgress;
        static const bool _TakesToken = _CAFunctorOptions<_Function>::_TakesToken;

        typedef typename _TaskTypeTraits<_ReturnType> _TaskTraits;
        typedef typename _AsyncAttributes<_Function, _ProgressType, typename _TaskTraits::_TaskRetType, _TaskTraits, _TakesToken, _TakesProgress> _AsyncAttributes;
    };
    // ***************************************************************************
    // AsyncInfo (and completion) Layer:
    //
#ifndef RUNTIMECLASS_Concurrency_winrt_details__AsyncInfoBase_DEFINED
#define RUNTIMECLASS_Concurrency_winrt_details__AsyncInfoBase_DEFINED
    extern const __declspec(selectany) WCHAR RuntimeClass_Concurrency_winrt_details__AsyncInfoBase[] = L"Concurrency_winrt.details._AsyncInfoBase";
#endif

    //
    // Internal base class implementation for async operations (based on internal Windows representation for ABI level async operations)
    //
    template < typename _Attributes, _AsyncResultType resultType = SingleResult >
    class _AsyncInfoBase abstract : public Microsoft::WRL::RuntimeClass<
        Microsoft::WRL::RuntimeClassFlags< Microsoft::WRL::RuntimeClassType::WinRt>, Microsoft::WRL::Implements<typename _Attributes::_AsyncBaseType, ABI::Windows::Foundation::IAsyncInfo>>
    {
        InspectableClass(RuntimeClass_Concurrency_winrt_details__AsyncInfoBase, BaseTrust)
    public:
        _AsyncInfoBase() :
            _M_currentStatus(_AsyncStatusInternal::_AsyncCreated),
            _M_errorCode(S_OK),
            _M_completeDelegate(nullptr),
            _M_CompleteDelegateAssigned(0),
            _M_CallbackMade(0)
        {
#if _MSC_VER < 1800
            _M_id = Concurrency::details::_GetNextAsyncId();
#else
            _M_id = Concurrency::details::platform::GetNextAsyncId();
#endif
        }
    public:
        virtual STDMETHODIMP GetResults(typename _Attributes::_ReturnType_abi* results)
        {
            (void)results;
            return E_UNEXPECTED;
        }

        virtual STDMETHODIMP get_Id(unsigned int* id)
        {
            HRESULT hr = _CheckValidStateForAsyncInfoCall();
            if (FAILED(hr)) return hr;
            if (!id) return E_POINTER;
            *id = _M_id;
            return S_OK;
        }

        virtual STDMETHODIMP put_Id(unsigned int id)
        {
            HRESULT hr = _CheckValidStateForAsyncInfoCall();
            if (FAILED(hr)) return hr;

            if (id == 0)
            {
                return E_INVALIDARG;
            }
            else if (_M_currentStatus != _AsyncStatusInternal::_AsyncCreated)
            {
                return E_ILLEGAL_METHOD_CALL;
            }

            _M_id = id;
            return S_OK;
        }
        virtual STDMETHODIMP get_Status(ABI::Windows::Foundation::AsyncStatus* status)
        {
            HRESULT hr = _CheckValidStateForAsyncInfoCall();
            if (FAILED(hr)) return hr;
            if (!status) return E_POINTER;

            _AsyncStatusInternal _Current = _M_currentStatus;
            //
            // Map our internal cancel pending to cancelled. This way "pending cancelled" looks to the outside as "cancelled" but
            // can still transition to "completed" if the operation completes without acknowledging the cancellation request
            //
            switch (_Current)
            {
            case _AsyncCancelPending:
                _Current = _AsyncCanceled;
                break;
            case _AsyncCreated:
                _Current = _AsyncStarted;
                break;
            default:
                break;
            }

            *status = static_cast<ABI::Windows::Foundation::AsyncStatus>(_Current);
            return S_OK;
        }

        virtual STDMETHODIMP get_ErrorCode(HRESULT* errorCode)
        {
            HRESULT hr = _CheckValidStateForAsyncInfoCall();
            if (FAILED(hr)) return hr;
            if (!hr) return hr;
            *errorCode = _M_errorCode;
            return S_OK;
        }

        virtual STDMETHODIMP get_Progress(typename _Attributes::_ProgressDelegateType** _ProgressHandler)
        {
            return _GetOnProgress(_ProgressHandler);
        }

        virtual STDMETHODIMP put_Progress(typename _Attributes::_ProgressDelegateType* _ProgressHandler)
        {
            return _PutOnProgress(_ProgressHandler);
        }

        virtual STDMETHODIMP Cancel()
        {
            if (_TransitionToState(_AsyncCancelPending))
            {
                _OnCancel();
            }
            return S_OK;
        }

        virtual STDMETHODIMP Close()
        {
            if (_TransitionToState(_AsyncClosed))
            {
                _OnClose();
            }
            else
            {
                if (_M_currentStatus != _AsyncClosed) // Closed => Closed transition is just ignored
                {
                    return E_ILLEGAL_STATE_CHANGE;
                }
            }
            return S_OK;
        }

        virtual STDMETHODIMP get_Completed(typename _Attributes::_CompletionDelegateType** _CompleteHandler)
        {
            _CheckValidStateForDelegateCall();
            if (!_CompleteHandler) return E_POINTER;
            *_CompleteHandler = _M_completeDelegate.Get();
            return S_OK;
        }

        virtual STDMETHODIMP put_Completed(typename _Attributes::_CompletionDelegateType* _CompleteHandler)
        {
            _CheckValidStateForDelegateCall();
            // this delegate property is "write once"
            if (InterlockedIncrement(&_M_CompleteDelegateAssigned) == 1)
            {
                _M_completeDelegateContext = _ContextCallback::_CaptureCurrent();
                _M_completeDelegate = _CompleteHandler;
                // Guarantee that the write of _M_completeDelegate is ordered with respect to the read of state below
                // as perceived from _FireCompletion on another thread.
                MemoryBarrier();
                if (_IsTerminalState())
                {
                    _FireCompletion();
                }
            }
            else
            {
                return E_ILLEGAL_DELEGATE_ASSIGNMENT;
            }
            return S_OK;
        }

    protected:
        // _Start - this is not externally visible since async operations "hot start" before returning to the caller
        STDMETHODIMP _Start()
        {
            if (_TransitionToState(_AsyncStarted))
            {
                _OnStart();
            }
            else
            {
                return E_ILLEGAL_STATE_CHANGE;
            }
            return S_OK;
        }

        HRESULT _FireCompletion()
        {
            HRESULT hr = S_OK;
            _TryTransitionToCompleted();

            // we guarantee that completion can only ever be fired once
            if (_M_completeDelegate != nullptr && InterlockedIncrement(&_M_CallbackMade) == 1)
            {
                hr = _M_completeDelegateContext._CallInContext([=]() -> HRESULT {
                    ABI::Windows::Foundation::AsyncStatus status;
                    HRESULT hr;
                    if (SUCCEEDED(hr = this->get_Status(&status)))
                        _M_completeDelegate->Invoke((_Attributes::_AsyncBaseType*)this, status);
                    _M_completeDelegate = nullptr;
                    return hr;
                });
            }
            return hr;
        }

        virtual STDMETHODIMP _GetOnProgress(typename _Attributes::_ProgressDelegateType** _ProgressHandler)
        {
            (void)_ProgressHandler;
            return E_UNEXPECTED;
        }

        virtual STDMETHODIMP _PutOnProgress(typename _Attributes::_ProgressDelegateType* _ProgressHandler)
        {
            (void)_ProgressHandler;
            return E_UNEXPECTED;
        }


        bool _TryTransitionToCompleted()
        {
            return _TransitionToState(_AsyncStatusInternal::_AsyncCompleted);
        }

        bool _TryTransitionToCancelled()
        {
            return _TransitionToState(_AsyncStatusInternal::_AsyncCanceled);
        }

        bool _TryTransitionToError(const HRESULT error)
        {
            _InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&_M_errorCode), error, S_OK);
            return _TransitionToState(_AsyncStatusInternal::_AsyncError);
        }

        // This method checks to see if the delegate properties can be
        // modified in the current state and generates the appropriate
        // error hr in the case of violation.
        inline HRESULT _CheckValidStateForDelegateCall()
        {
            if (_M_currentStatus == _AsyncClosed)
            {
                return E_ILLEGAL_METHOD_CALL;
            }
            return S_OK;
        }

        // This method checks to see if results can be collected in the
        // current state and generates the appropriate error hr in
        // the case of a violation.
        inline HRESULT _CheckValidStateForResultsCall()
        {
            _AsyncStatusInternal _Current = _M_currentStatus;

            if (_Current == _AsyncError)
            {
                return _M_errorCode;
            }
#pragma warning(push)
#pragma warning(disable: 4127) // Conditional expression is constant
            // single result illegal before transition to Completed or Cancelled state
            if (resultType == SingleResult)
#pragma warning(pop)
            {
                if (_Current != _AsyncCompleted)
                {
                    return E_ILLEGAL_METHOD_CALL;
                }
            }
            // multiple results can be called after Start has been called and before/after Completed
            else if (_Current != _AsyncStarted &&
                _Current != _AsyncCancelPending &&
                _Current != _AsyncCanceled &&
                _Current != _AsyncCompleted)
            {
                return E_ILLEGAL_METHOD_CALL;
            }
            return S_OK;
        }

        // This method can be called by derived classes periodically to determine
        // whether the asynchronous operation should continue processing or should
        // be halted.
        inline bool _ContinueAsyncOperation()
        {
            return _M_currentStatus == _AsyncStarted;
        }

        // These two methods are used to allow the async worker implementation do work on
        // state transitions. No real "work" should be done in these methods. In other words
        // they should not block for a long time on UI timescales.
        virtual void _OnStart() = 0;
        virtual void _OnClose() = 0;
        virtual void _OnCancel() = 0;

    private:

        // This method is used to check if calls to the AsyncInfo properties
        // (id, status, errorcode) are legal in the current state. It also
        // generates the appropriate error hr to return in the case of an
        // illegal call.
        inline HRESULT _CheckValidStateForAsyncInfoCall()
        {
            _AsyncStatusInternal _Current = _M_currentStatus;
            if (_Current == _AsyncClosed)
            {
                return E_ILLEGAL_METHOD_CALL;
            }
            else if (_Current == _AsyncCreated)
            {
                return E_ASYNC_OPERATION_NOT_STARTED;
            }
            return S_OK;
        }

        inline bool _TransitionToState(const _AsyncStatusInternal _NewState)
        {
            _AsyncStatusInternal _Current = _M_currentStatus;

            // This enforces the valid state transitions of the asynchronous worker object
            // state machine.
            switch (_NewState)
            {
            case _AsyncStatusInternal::_AsyncStarted:
                if (_Current != _AsyncCreated)
                {
                    return false;
                }
                break;
            case _AsyncStatusInternal::_AsyncCompleted:
                if (_Current != _AsyncStarted && _Current != _AsyncCancelPending)
                {
                    return false;
                }
                break;
            case _AsyncStatusInternal::_AsyncCancelPending:
                if (_Current != _AsyncStarted)
                {
                    return false;
                }
                break;
            case _AsyncStatusInternal::_AsyncCanceled:
                if (_Current != _AsyncStarted && _Current != _AsyncCancelPending)
                {
                    return false;
                }
                break;
            case _AsyncStatusInternal::_AsyncError:
                if (_Current != _AsyncStarted && _Current != _AsyncCancelPending)
                {
                    return false;
                }
                break;
            case _AsyncStatusInternal::_AsyncClosed:
                if (!_IsTerminalState(_Current))
                {
                    return false;
                }
                break;
            default:
                return false;
                break;
            }

            // attempt the transition to the new state
            // Note: if currentStatus_ == _Current, then there was no intervening write
            // by the async work object and the swap succeeded.
            _AsyncStatusInternal _RetState = static_cast<_AsyncStatusInternal>(
                _InterlockedCompareExchange(reinterpret_cast<volatile LONG*>(&_M_currentStatus),
                _NewState,
                static_cast<LONG>(_Current)));

            // ICE returns the former state, if the returned state and the
            // state we captured at the beginning of this method are the same,
            // the swap succeeded.
            return (_RetState == _Current);
        }

        inline bool _IsTerminalState()
        {
            return _IsTerminalState(_M_currentStatus);
        }

        inline bool _IsTerminalState(_AsyncStatusInternal status)
        {
            return (status == _AsyncError ||
                status == _AsyncCanceled ||
                status == _AsyncCompleted ||
                status == _AsyncClosed);
        }

    private:

        _ContextCallback        _M_completeDelegateContext;
        Microsoft::WRL::ComPtr<typename _Attributes::_CompletionDelegateType>  _M_completeDelegate; //ComPtr cannot be volatile as it does not have volatile accessors
        _AsyncStatusInternal volatile                   _M_currentStatus;
        HRESULT volatile                                _M_errorCode;
        unsigned int                                    _M_id;
        long volatile                                   _M_CompleteDelegateAssigned;
        long volatile                                   _M_CallbackMade;
    };

    // ***************************************************************************
    // Progress Layer (optional):
    //

    template< typename _Attributes, bool _HasProgress, _AsyncResultType _ResultType = SingleResult >
    class _AsyncProgressBase abstract : public _AsyncInfoBase<_Attributes, _ResultType>
    {
    };

    template< typename _Attributes, _AsyncResultType _ResultType>
    class _AsyncProgressBase<_Attributes, true, _ResultType> abstract : public _AsyncInfoBase<_Attributes, _ResultType>
    {
    public:

        _AsyncProgressBase() : _AsyncInfoBase<_Attributes, _ResultType>(),
            _M_progressDelegate(nullptr)
        {
        }

        virtual STDMETHODIMP _GetOnProgress(typename _Attributes::_ProgressDelegateType** _ProgressHandler) override
        {
            HRESULT hr = _CheckValidStateForDelegateCall();
            if (FAILED(hr)) return hr;
            *_ProgressHandler = _M_progressDelegate;
            return S_OK;
        }

        virtual STDMETHODIMP _PutOnProgress(typename _Attributes::_ProgressDelegateType* _ProgressHandler) override
        {
            HRESULT hr = _CheckValidStateForDelegateCall();
            if (FAILED(hr)) return hr;
            _M_progressDelegate = _ProgressHandler;
            _M_progressDelegateContext = _ContextCallback::_CaptureCurrent();
            return S_OK;
        }

    public:

        void _FireProgress(const typename _Attributes::_ProgressType_abi& _ProgressValue)
        {
            if (_M_progressDelegate != nullptr)
            {
                _M_progressDelegateContext._CallInContext([=]() -> HRESULT {
                    _M_progressDelegate->Invoke((_Attributes::_AsyncBaseType*)this, _ProgressValue);
                    return S_OK;
                });
            }
        }

    private:

        _ContextCallback _M_progressDelegateContext;
        typename _Attributes::_ProgressDelegateType* _M_progressDelegate;
    };

    template<typename _Attributes, _AsyncResultType _ResultType = SingleResult>
    class _AsyncBaseProgressLayer abstract : public _AsyncProgressBase<_Attributes, _Attributes::_TakesProgress, _ResultType>
    {
    };

    // ***************************************************************************
    // Task Adaptation Layer:
    //

    //
    // _AsyncTaskThunkBase provides a bridge between IAsync<Action/Operation> and task.
    //
    template<typename _Attributes, typename _ReturnType>
    class _AsyncTaskThunkBase abstract : public _AsyncBaseProgressLayer<_Attributes>
    {
    public:

        //AsyncAction*
        virtual STDMETHODIMP GetResults()
        {
            HRESULT hr = _CheckValidStateForResultsCall();
            if (FAILED(hr)) return hr;
            _M_task.get();
            return S_OK;
        }
    public:
        typedef task<_ReturnType> _TaskType;

        _AsyncTaskThunkBase(const _TaskType& _Task)
            : _M_task(_Task)
        {
        }

        _AsyncTaskThunkBase()
        {
        }
#if _MSC_VER < 1800
        void _SetTaskCreationAddressHint(void* _SourceAddressHint)
        {
            if (!(std::is_same<_Attributes::_AsyncKind, _TypeSelectorAsyncTask>::value))
            {
                // Overwrite the creation address with the return address of create_async unless the
                // lambda returned a task. If the create async lambda returns a task, that task is reused and
                // we want to preserve its creation address hint.
                _M_task._SetTaskCreationAddressHint(_SourceAddressHint);
            }
        }
#endif
    protected:
        virtual void _OnStart() override
        {
            _M_task.then([=](_TaskType _Antecedent) -> HRESULT {
                try
                {
                    _Antecedent.get();
                }
                catch (Concurrency::task_canceled&)
                {
                    _TryTransitionToCancelled();
                }
                catch (IRestrictedErrorInfo*& _Ex)
                {
                    HRESULT hr;
                    HRESULT _hr;
                    hr = _Ex->GetErrorDetails(NULL, &_hr, NULL, NULL);
                    if (SUCCEEDED(hr)) hr = _hr;
                    _TryTransitionToError(hr);
                }
                catch (...)
                {
                    _TryTransitionToError(E_FAIL);
                }
                return _FireCompletion();
            });
        }

    protected:
        _TaskType _M_task;
        Concurrency::cancellation_token_source _M_cts;
    };

    template<typename _Attributes, typename _ReturnType, typename _Return>
    class _AsyncTaskReturn abstract : public _AsyncTaskThunkBase<_Attributes, _Return>
    {
    public:
        //AsyncOperation*
        virtual STDMETHODIMP GetResults(_ReturnType* results)
        {
            HRESULT hr = _CheckValidStateForResultsCall();
            if (FAILED(hr)) return hr;
            _M_task.get();
            *results = _M_results;
            return S_OK;
        }
        template <typename _Function>
#if _MSC_VER >= 1800
        void DoCreateTask(_Function _func, const _TaskCreationCallstack & _callstack)
        {
            _M_task = _Attributes::_Generate_Task(_func, this, _M_cts, &_M_results, _callstack);
        }
#else
        void DoCreateTask(_Function _func)
        {
            _M_task = _Attributes::_Generate_Task(_func, this, _M_cts, &_M_results);
        }
#endif
    protected:
        _ReturnType _M_results;
    };

    template<typename _Attributes, typename _ReturnType>
    class _AsyncTaskReturn<_Attributes, _ReturnType, void> abstract : public _AsyncTaskThunkBase<_Attributes, void>
    {
    public:
        template <typename _Function>
#if _MSC_VER >= 1800
        void DoCreateTask(_Function _func, const _TaskCreationCallstack & _callstack)
        {
            _M_task = _Attributes::_Generate_Task(_func, this, _M_cts, _callstack);
        }
#else
        void DoCreateTask(_Function _func)
        {
            _M_task = _Attributes::_Generate_Task(_func, this, _M_cts);
        }
#endif
    };

    template<typename _Attributes>
    class _AsyncTaskReturn<_Attributes, void, task<void>> abstract : public _AsyncTaskThunkBase<_Attributes, task<void>>
    {
    public:
        template <typename _Function>
#if _MSC_VER >= 1800
        void DoCreateTask(_Function _func, const _TaskCreationCallstack & _callstack)
        {
            _M_task = _Attributes::_Generate_Task(_func, this, _M_cts, &_M_results, _callstack);
        }
#else
        void DoCreateTask(_Function _func)
        {
            _M_task = _Attributes::_Generate_Task(_func, this, _M_cts, &_M_results);
        }
#endif
    protected:
        task<void> _M_results;
    };

    template<typename _Attributes>
    class _AsyncTaskThunk : public _AsyncTaskReturn<_Attributes, typename _Attributes::_ReturnType_abi, typename _Attributes::_ReturnType>
    {
    public:

        _AsyncTaskThunk(const _TaskType& _Task) :
            _AsyncTaskThunkBase(_Task)
        {
        }

        _AsyncTaskThunk()
        {
        }

    protected:

        virtual void _OnClose() override
        {
        }

        virtual void _OnCancel() override
        {
            _M_cts.cancel();
        }
    };

    // ***************************************************************************
    // Async Creation Layer:
    //
    template<typename _Function>
    class _AsyncTaskGeneratorThunk : public _AsyncTaskThunk<typename _AsyncLambdaTypeTraits<_Function>::_AsyncAttributes>
    {
    public:

        typedef typename _AsyncLambdaTypeTraits<_Function>::_AsyncAttributes _Attributes;
        typedef typename _AsyncTaskThunk<_Attributes> _Base;
        typedef typename _Attributes::_AsyncBaseType _AsyncBaseType;

#if _MSC_VER >= 1800
        _AsyncTaskGeneratorThunk(const _Function& _Func, const _TaskCreationCallstack &_callstack) : _M_func(_Func), _M_creationCallstack(_callstack)
#else
        _AsyncTaskGeneratorThunk(const _Function& _Func) : _M_func(_Func)
#endif
        {
            // Virtual call here is safe as the class is declared 'sealed'
            _Start();
        }

    protected:

        //
        // The only thing we must do different from the base class is we must spin the hot task on transition from Created->Started. Otherwise,
        // let the base thunk handle everything.
        //

        virtual void _OnStart() override
        {
            //
            // Call the appropriate task generator to actually produce a task of the expected type. This might adapt the user lambda for progress reports,
            // wrap the return result in a task, or allow for direct return of a task depending on the form of the lambda.
            //
#if _MSC_VER >= 1800
            DoCreateTask<_Function>(_M_func, _M_creationCallstack);
#else
            DoCreateTask<_Function>(_M_func);
#endif
            _Base::_OnStart();
        }

        virtual void _OnCancel() override
        {
            _Base::_OnCancel();
        }

    private:
#if _MSC_VER >= 1800
        _TaskCreationCallstack _M_creationCallstack;
#endif
        _Function _M_func;
    };
} // namespace details

/// <summary>
///     Creates a Windows Runtime asynchronous construct based on a user supplied lambda or function object. The return type of <c>create_async</c> is
///     one of either <c>IAsyncAction^</c>, <c>IAsyncActionWithProgress&lt;TProgress&gt;^</c>, <c>IAsyncOperation&lt;TResult&gt;^</c>, or
///     <c>IAsyncOperationWithProgress&lt;TResult, TProgress&gt;^</c> based on the signature of the lambda passed to the method.
/// </summary>
/// <param name="_Func">
///     The lambda or function object from which to create a Windows Runtime asynchronous construct.
/// </param>
/// <returns>
///     An asynchronous construct represented by an IAsyncAction^, IAsyncActionWithProgress&lt;TProgress&gt;^, IAsyncOperation&lt;TResult&gt;^, or an
///     IAsyncOperationWithProgress&lt;TResult, TProgress&gt;^. The interface returned depends on the signature of the lambda passed into the function.
/// </returns>
/// <remarks>
///     The return type of the lambda determines whether the construct is an action or an operation.
///     <para>Lambdas that return void cause the creation of actions. Lambdas that return a result of type <c>TResult</c> cause the creation of
///     operations of TResult.</para>
///     <para>The lambda may also return a <c>task&lt;TResult&gt;</c> which encapsulates the aysnchronous work within itself or is the continuation of
///     a chain of tasks that represent the asynchronous work. In this case, the lambda itself is executed inline, since the tasks are the ones that
///     execute asynchronously, and the return type of the lambda is unwrapped to produce the asynchronous construct returned by <c>create_async</c>.
///     This implies that a lambda that returns a task&lt;void&gt; will cause the creation of actions, and a lambda that returns a task&lt;TResult&gt; will
///     cause the creation of operations of TResult.</para>
///     <para>The lambda may take either zero, one or two arguments. The valid arguments are <c>progress_reporter&lt;TProgress&gt;</c> and
///     <c>cancellation_token</c>, in that order if both are used. A lambda without arguments causes the creation of an asynchronous construct without
///     the capability for progress reporting. A lambda that takes a progress_reporter&lt;TProgress&gt; will cause <c>create_async</c> to return an asynchronous
///     construct which reports progress of type TProgress each time the <c>report</c> method of the progress_reporter object is called. A lambda that
///     takes a cancellation_token may use that token to check for cancellation, or pass it to tasks that it creates so that cancellation of the
///     asynchronous construct causes cancellation of those tasks.</para>
///     <para>If the body of the lambda or function object returns a result (and not a task&lt;TResult&gt;), the lamdba will be executed
///     asynchronously within the process MTA in the context of a task the Runtime implicitly creates for it. The <c>IAsyncInfo::Cancel</c> method will
///     cause cancellation of the implicit task.</para>
///     <para>If the body of the lambda returns a task, the lamba executes inline, and by declaring the lambda to take an argument of type
///     <c>cancellation_token</c> you can trigger cancellation of any tasks you create within the lambda by passing that token in when you create them.
///     You may also use the <c>register_callback</c> method on the token to cause the Runtime to invoke a callback when you call <c>IAsyncInfo::Cancel</c> on
///     the async operation or action produced..</para>
///     <para>This function is only available to Windows Store apps.</para>
/// </remarks>
/// <seealso cref="task Class"/>
/// <seealso cref="progress_reporter Class"/>
/// <seealso cref="cancelation_token Class"/>
/**/
template<typename _ReturnType, typename _Function>
__declspec(noinline) // Ask for no inlining so that the _ReturnAddress intrinsic gives us the expected result
details::_AsyncTaskGeneratorThunk<_Function>* create_async(const _Function& _Func)
{
    static_assert(std::is_same<decltype(details::_IsValidCreateAsync<_ReturnType>(_Func, 0, 0, 0, 0, 0, 0, 0, 0)), std::true_type>::value,
        "argument to create_async must be a callable object taking zero, one, two or three arguments");
#if _MSC_VER >= 1800
    Microsoft::WRL::ComPtr<details::_AsyncTaskGeneratorThunk<_Function>> _AsyncInfo = Microsoft::WRL::Make<details::_AsyncTaskGeneratorThunk<_Function>>(_Func, _CAPTURE_CALLSTACK());
#else
    Microsoft::WRL::ComPtr<details::_AsyncTaskGeneratorThunk<_Function>> _AsyncInfo = Microsoft::WRL::Make<details::_AsyncTaskGeneratorThunk<_Function>>(_Func);
    _AsyncInfo->_SetTaskCreationAddressHint(_ReturnAddress());
#endif
    return _AsyncInfo.Detach();
}

namespace details
{
#if _MSC_VER < 1800
    // Internal API which retrieves the next async id.
    _CRTIMP2 unsigned int __cdecl _GetNextAsyncId();
#endif
    // Helper struct for when_all operators to know when tasks have completed
    template<typename _Type>
    struct _RunAllParam
    {
        _RunAllParam() : _M_completeCount(0), _M_numTasks(0)
        {
        }

        void _Resize(size_t _Len, bool _SkipVector = false)
        {
            _M_numTasks = _Len;
            if (!_SkipVector)
#if _MSC_VER >= 1800
            {
                _M_vector._Result.resize(_Len);
            }
#else
                _M_vector.resize(_Len);
            _M_contexts.resize(_Len);
#endif
        }

        task_completion_event<_Unit_type>       _M_completed;
        atomic_size_t                           _M_completeCount;
#if _MSC_VER >= 1800
        _ResultHolder<std::vector<_Type> >      _M_vector;
        _ResultHolder<_Type>                    _M_mergeVal;
#else
        std::vector<_Type>                      _M_vector;
        std::vector<_ContextCallback>           _M_contexts;
        _Type                                   _M_mergeVal;
#endif
        size_t                                  _M_numTasks;
    };

#if _MSC_VER >= 1800
    template<typename _Type>
    struct _RunAllParam<std::vector<_Type> >
    {
        _RunAllParam() : _M_completeCount(0), _M_numTasks(0)
        {
        }

        void _Resize(size_t _Len, bool _SkipVector = false)
        {
            _M_numTasks = _Len;

            if (!_SkipVector)
            {
                _M_vector.resize(_Len);
            }
        }

        task_completion_event<_Unit_type>       _M_completed;
        std::vector<_ResultHolder<std::vector<_Type> > >  _M_vector;
        atomic_size_t     _M_completeCount;
        size_t                                  _M_numTasks;
    };
#endif

    // Helper struct specialization for void
    template<>
#if _MSC_VER >= 1800
    struct _RunAllParam<_Unit_type>
#else
    struct _RunAllParam<void>
#endif
    {
        _RunAllParam() : _M_completeCount(0), _M_numTasks(0)
        {
        }

        void _Resize(size_t _Len)
        {
            _M_numTasks = _Len;
        }

        task_completion_event<_Unit_type> _M_completed;
        atomic_size_t _M_completeCount;
        size_t _M_numTasks;
    };

    inline void _JoinAllTokens_Add(const Concurrency::cancellation_token_source& _MergedSrc, Concurrency::details::_CancellationTokenState *_PJoinedTokenState)
    {
        if (_PJoinedTokenState != nullptr && _PJoinedTokenState != Concurrency::details::_CancellationTokenState::_None())
        {
            Concurrency::cancellation_token _T = Concurrency::cancellation_token::_FromImpl(_PJoinedTokenState);
            _T.register_callback([=](){
                _MergedSrc.cancel();
            });
        }
    }

    template<typename _ElementType, typename _Function, typename _TaskType>
    void _WhenAllContinuationWrapper(_RunAllParam<_ElementType>* _PParam, _Function _Func, task<_TaskType>& _Task)
    {
        if (_Task._GetImpl()->_IsCompleted())
        {
            _Func();
#if _MSC_VER >= 1800
            if (Concurrency::details::atomic_increment(_PParam->_M_completeCount) == _PParam->_M_numTasks)
#else
            if (_InterlockedIncrementSizeT(&_PParam->_M_completeCount) == _PParam->_M_numTasks)
#endif
            {
                // Inline execute its direct continuation, the _ReturnTask
                _PParam->_M_completed.set(_Unit_type());
                // It's safe to delete it since all usage of _PParam in _ReturnTask has been finished.
                delete _PParam;
            }
        }
        else
        {
            _CONCRT_ASSERT(_Task._GetImpl()->_IsCanceled());
            if (_Task._GetImpl()->_HasUserException())
            {
                // _Cancel will return false if the TCE is already canceled with or without exception
                _PParam->_M_completed._Cancel(_Task._GetImpl()->_GetExceptionHolder());
            }
            else
            {
                _PParam->_M_completed._Cancel();
            }
#if _MSC_VER >= 1800
            if (Concurrency::details::atomic_increment(_PParam->_M_completeCount) == _PParam->_M_numTasks)
#else
            if (_InterlockedIncrementSizeT(&_PParam->_M_completeCount) == _PParam->_M_numTasks)
#endif
            {
                delete _PParam;
            }
        }
    }

    template<typename _ElementType, typename _Iterator>
    struct _WhenAllImpl
    {
#if _MSC_VER >= 1800
        static task<std::vector<_ElementType>> _Perform(const task_options& _TaskOptions, _Iterator _Begin, _Iterator _End)
#else
        static task<std::vector<_ElementType>> _Perform(Concurrency::details::_CancellationTokenState *_PTokenState, _Iterator _Begin, _Iterator _End)
#endif
        {
#if _MSC_VER >= 1800
            Concurrency::details::_CancellationTokenState *_PTokenState = _TaskOptions.has_cancellation_token() ? _TaskOptions.get_cancellation_token()._GetImplValue() : nullptr;
#endif
            auto _PParam = new _RunAllParam<_ElementType>();
            Concurrency::cancellation_token_source _MergedSource;

            // Step1: Create task completion event.
#if _MSC_VER >= 1800
            task_options _Options(_TaskOptions);
            _Options.set_cancellation_token(_MergedSource.get_token());
            task<_Unit_type> _All_tasks_completed(_PParam->_M_completed, _Options);
#else
            task<_Unit_type> _All_tasks_completed(_PParam->_M_completed, _MergedSource.get_token());
#endif
            // The return task must be created before step 3 to enforce inline execution.
            auto _ReturnTask = _All_tasks_completed._Then([=](_Unit_type, std::vector<_ElementType>* retVal) -> HRESULT {
#if _MSC_VER >= 1800
                * retVal = _PParam->_M_vector.Get();
#else
                auto _Result = _PParam->_M_vector; // copy by value

                size_t _Index = 0;
                for (auto _It = _Result.begin(); _It != _Result.end(); ++_It)
                {
                    *_It = _ResultContext<_ElementType>::_GetValue(*_It, _PParam->_M_contexts[_Index++], false);
                }
                *retVal = _Result;
#endif
                return S_OK;
#if _MSC_VER >= 1800
            }, nullptr);
#else
            }, nullptr, true);
#endif
            // Step2: Combine and check tokens, and count elements in range.
            if (_PTokenState)
            {
                details::_JoinAllTokens_Add(_MergedSource, _PTokenState);
                _PParam->_Resize(static_cast<size_t>(std::distance(_Begin, _End)));
            }
            else
            {
                size_t _TaskNum = 0;
                for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
                {
                    _TaskNum++;
                    details::_JoinAllTokens_Add(_MergedSource, _PTask->_GetImpl()->_M_pTokenState);
                }
                _PParam->_Resize(_TaskNum);
            }

            // Step3: Check states of previous tasks.
            if (_Begin == _End)
            {
                _PParam->_M_completed.set(_Unit_type());
                delete _PParam;
            }
            else
            {
                size_t _Index = 0;
                for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
                {
                    if (_PTask->is_apartment_aware())
                    {
                        _ReturnTask._SetAsync();
                    }

                    _PTask->_Then([_PParam, _Index](task<_ElementType> _ResultTask) -> HRESULT {

#if _MSC_VER >= 1800
                        //  Dev10 compiler bug
                        typedef _ElementType _ElementTypeDev10;
                        auto _PParamCopy = _PParam;
                        auto _IndexCopy = _Index;
                        auto _Func = [_PParamCopy, _IndexCopy, &_ResultTask](){
                            _PParamCopy->_M_vector._Result[_IndexCopy] = _ResultTask._GetImpl()->_GetResult();
                        };
#else
                        auto _Func = [_PParam, _Index, &_ResultTask](){
                            _PParam->_M_vector[_Index] = _ResultTask._GetImpl()->_GetResult();
                            _PParam->_M_contexts[_Index] = _ResultContext<_ElementType>::_GetContext(false);
                        };
#endif
                        _WhenAllContinuationWrapper(_PParam, _Func, _ResultTask);
                        return S_OK;
#if _MSC_VER >= 1800
                    }, Concurrency::details::_CancellationTokenState::_None());
#else
                    }, Concurrency::details::_CancellationTokenState::_None(), false);
#endif

                    _Index++;
                }
            }

            return _ReturnTask;
        }
    };

    template<typename _ElementType, typename _Iterator>
    struct _WhenAllImpl<std::vector<_ElementType>, _Iterator>
    {
#if _MSC_VER >= 1800
        static task<std::vector<_ElementType>> _Perform(const task_options& _TaskOptions, _Iterator _Begin, _Iterator _End)
#else
        static task<std::vector<_ElementType>> _Perform(Concurrency::details::_CancellationTokenState *_PTokenState, _Iterator _Begin, _Iterator _End)
#endif
        {
#if _MSC_VER >= 1800
            Concurrency::details::_CancellationTokenState *_PTokenState = _TaskOptions.has_cancellation_token() ? _TaskOptions.get_cancellation_token()._GetImplValue() : nullptr;
#endif
            auto _PParam = new _RunAllParam<std::vector<_ElementType>>();
            Concurrency::cancellation_token_source _MergedSource;

            // Step1: Create task completion event.
#if _MSC_VER >= 1800
            task_options _Options(_TaskOptions);
            _Options.set_cancellation_token(_MergedSource.get_token());
            task<_Unit_type> _All_tasks_completed(_PParam->_M_completed, _Options);
#else
            task<_Unit_type> _All_tasks_completed(_PParam->_M_completed, _MergedSource.get_token());
#endif
            // The return task must be created before step 3 to enforce inline execution.
            auto _ReturnTask = _All_tasks_completed._Then([=](_Unit_type, std::vector<_ElementType>* retVal) -> HRESULT {
                _CONCRT_ASSERT(_PParam->_M_completeCount == _PParam->_M_numTasks);
                std::vector<_ElementType> _Result;
                for (size_t _I = 0; _I < _PParam->_M_numTasks; _I++)
                {
#if _MSC_VER >= 1800
                    const std::vector<_ElementType>& _Vec = _PParam->_M_vector[_I].Get();
#else
                    std::vector<_ElementType>& _Vec = _PParam->_M_vector[_I];

                    for (auto _It = _Vec.begin(); _It != _Vec.end(); ++_It)
                    {
                        *_It = _ResultContext<_ElementType>::_GetValue(*_It, _PParam->_M_contexts[_I], false);
                    }
#endif
                    _Result.insert(_Result.end(), _Vec.begin(), _Vec.end());
                }
                *retVal = _Result;
                return S_OK;
#if _MSC_VER >= 1800
            }, nullptr);
#else
            }, nullptr, true);
#endif

            // Step2: Combine and check tokens, and count elements in range.
            if (_PTokenState)
            {
                details::_JoinAllTokens_Add(_MergedSource, _PTokenState);
                _PParam->_Resize(static_cast<size_t>(std::distance(_Begin, _End)));
            }
            else
            {
                size_t _TaskNum = 0;
                for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
                {
                    _TaskNum++;
                    details::_JoinAllTokens_Add(_MergedSource, _PTask->_GetImpl()->_M_pTokenState);
                }
                _PParam->_Resize(_TaskNum);
            }

            // Step3: Check states of previous tasks.
            if (_Begin == _End)
            {
                _PParam->_M_completed.set(_Unit_type());
                delete _PParam;
            }
            else
            {
                size_t _Index = 0;
                for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
                {
                    if (_PTask->is_apartment_aware())
                    {
                        _ReturnTask._SetAsync();
                    }

                    _PTask->_Then([_PParam, _Index](task<std::vector<_ElementType>> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
                        //  Dev10 compiler bug
                        typedef _ElementType _ElementTypeDev10;
                        auto _PParamCopy = _PParam;
                        auto _IndexCopy = _Index;
                        auto _Func = [_PParamCopy, _IndexCopy, &_ResultTask]() {
                            _PParamCopy->_M_vector[_IndexCopy].Set(_ResultTask._GetImpl()->_GetResult());
                        };
#else
                        auto _Func = [_PParam, _Index, &_ResultTask]() {
                            _PParam->_M_vector[_Index] = _ResultTask._GetImpl()->_GetResult();
                            _PParam->_M_contexts[_Index] = _ResultContext<_ElementType>::_GetContext(false);
                        };
#endif
                        _WhenAllContinuationWrapper(_PParam, _Func, _ResultTask);
                        return S_OK;
#if _MSC_VER >= 1800
                    }, Concurrency::details::_CancellationTokenState::_None());
#else
                    }, Concurrency::details::_CancellationTokenState::_None(), false);
#endif

                    _Index++;
                }
            }

            return  _ReturnTask;
        }
    };

    template<typename _Iterator>
    struct _WhenAllImpl<void, _Iterator>
    {
#if _MSC_VER >= 1800
        static task<void> _Perform(const task_options& _TaskOptions, _Iterator _Begin, _Iterator _End)
#else
        static task<void> _Perform(Concurrency::details::_CancellationTokenState *_PTokenState, _Iterator _Begin, _Iterator _End)
#endif
        {
#if _MSC_VER >= 1800
            Concurrency::details::_CancellationTokenState *_PTokenState = _TaskOptions.has_cancellation_token() ? _TaskOptions.get_cancellation_token()._GetImplValue() : nullptr;
#endif
            auto _PParam = new _RunAllParam<_Unit_type>();
            Concurrency::cancellation_token_source _MergedSource;

            // Step1: Create task completion event.
#if _MSC_VER >= 1800
            task_options _Options(_TaskOptions);
            _Options.set_cancellation_token(_MergedSource.get_token());
            task<_Unit_type> _All_tasks_completed(_PParam->_M_completed, _Options);
#else
            task<_Unit_type> _All_tasks_completed(_PParam->_M_completed, _MergedSource.get_token());
#endif
            // The return task must be created before step 3 to enforce inline execution.
            auto _ReturnTask = _All_tasks_completed._Then([=](_Unit_type) -> HRESULT { return S_OK;
#if _MSC_VER >= 1800
            }, nullptr);
#else
            }, nullptr, false);
#endif

            // Step2: Combine and check tokens, and count elements in range.
            if (_PTokenState)
            {
                details::_JoinAllTokens_Add(_MergedSource, _PTokenState);
                _PParam->_Resize(static_cast<size_t>(std::distance(_Begin, _End)));
            }
            else
            {
                size_t _TaskNum = 0;
                for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
                {
                    _TaskNum++;
                    details::_JoinAllTokens_Add(_MergedSource, _PTask->_GetImpl()->_M_pTokenState);
                }
                _PParam->_Resize(_TaskNum);
            }

            // Step3: Check states of previous tasks.
            if (_Begin == _End)
            {
                _PParam->_M_completed.set(_Unit_type());
                delete _PParam;
            }
            else
            {
                for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
                {
                    if (_PTask->is_apartment_aware())
                    {
                        _ReturnTask._SetAsync();
                    }

                    _PTask->_Then([_PParam](task<void> _ResultTask) -> HRESULT {

                        auto _Func = []() -> HRESULT { return S_OK;  };
                        _WhenAllContinuationWrapper(_PParam, _Func, _ResultTask);
                        return S_OK;
#if _MSC_VER >= 1800
                    }, Concurrency::details::_CancellationTokenState::_None());
#else
                    }, Concurrency::details::_CancellationTokenState::_None(), false);
#endif
                }
            }

            return _ReturnTask;
        }
    };

    template<typename _ReturnType>
    task<std::vector<_ReturnType>> _WhenAllVectorAndValue(const task<std::vector<_ReturnType>>& _VectorTask, const task<_ReturnType>& _ValueTask,
        bool _OutputVectorFirst)
    {
        auto _PParam = new _RunAllParam<_ReturnType>();
        Concurrency::cancellation_token_source _MergedSource;

        // Step1: Create task completion event.
        task<_Unit_type> _All_tasks_completed(_PParam->_M_completed, _MergedSource.get_token());
        // The return task must be created before step 3 to enforce inline execution.
        auto _ReturnTask = _All_tasks_completed._Then([=](_Unit_type, std::vector<_ReturnType>* retVal) -> HRESULT {
            _CONCRT_ASSERT(_PParam->_M_completeCount == 2);
#if _MSC_VER >= 1800
            auto _Result = _PParam->_M_vector.Get(); // copy by value
            auto _mergeVal = _PParam->_M_mergeVal.Get();
#else
            auto _Result = _PParam->_M_vector; // copy by value
            for (auto _It = _Result.begin(); _It != _Result.end(); ++_It)
            {
                *_It = _ResultContext<_ReturnType>::_GetValue(*_It, _PParam->_M_contexts[0], false);
            }
#endif

            if (_OutputVectorFirst == true)
            {
#if _MSC_VER >= 1800
                _Result.push_back(_mergeVal);
#else
                _Result.push_back(_ResultContext<_ReturnType>::_GetValue(_PParam->_M_mergeVal, _PParam->_M_contexts[1], false));
#endif
            }
            else
            {
#if _MSC_VER >= 1800
                _Result.insert(_Result.begin(), _mergeVal);
#else
                _Result.insert(_Result.begin(), _ResultContext<_ReturnType>::_GetValue(_PParam->_M_mergeVal, _PParam->_M_contexts[1], false));
#endif
            }
            *retVal = _Result;
            return S_OK;
        }, nullptr, true);

        // Step2: Combine and check tokens.
        _JoinAllTokens_Add(_MergedSource, _VectorTask._GetImpl()->_M_pTokenState);
        _JoinAllTokens_Add(_MergedSource, _ValueTask._GetImpl()->_M_pTokenState);

        // Step3: Check states of previous tasks.
        _PParam->_Resize(2, true);

        if (_VectorTask.is_apartment_aware() || _ValueTask.is_apartment_aware())
        {
            _ReturnTask._SetAsync();
        }
        _VectorTask._Then([_PParam](task<std::vector<_ReturnType>> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
            //  Dev10 compiler bug
            typedef _ReturnType _ReturnTypeDev10;
            auto _PParamCopy = _PParam;
            auto _Func = [_PParamCopy, &_ResultTask]() {
                auto _ResultLocal = _ResultTask._GetImpl()->_GetResult();
                _PParamCopy->_M_vector.Set(_ResultLocal);
            };
#else
            auto _Func = [_PParam, &_ResultTask]() {
                _PParam->_M_vector = _ResultTask._GetImpl()->_GetResult();
                _PParam->_M_contexts[0] = _ResultContext<_ReturnType>::_GetContext(false);
            };
#endif

            _WhenAllContinuationWrapper(_PParam, _Func, _ResultTask);
            return S_OK;
#if _MSC_VER >= 1800
        }, _CancellationTokenState::_None());
#else
        }, _CancellationTokenState::_None(), false);
#endif
        _ValueTask._Then([_PParam](task<_ReturnType> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
            //  Dev10 compiler bug
            typedef _ReturnType _ReturnTypeDev10;
            auto _PParamCopy = _PParam;
            auto _Func = [_PParamCopy, &_ResultTask]() {
                auto _ResultLocal = _ResultTask._GetImpl()->_GetResult();
                _PParamCopy->_M_mergeVal.Set(_ResultLocal);
            };
#else
            auto _Func = [_PParam, &_ResultTask]() {
                _PParam->_M_mergeVal = _ResultTask._GetImpl()->_GetResult();
                _PParam->_M_contexts[1] = _ResultContext<_ReturnType>::_GetContext(false);
            };
#endif
            _WhenAllContinuationWrapper(_PParam, _Func, _ResultTask);
            return S_OK;
#if _MSC_VER >= 1800
        }, _CancellationTokenState::_None());
#else
        }, _CancellationTokenState::_None(), false);
#endif

        return _ReturnTask;
    }
} // namespace details

#if _MSC_VER < 1800
/// <summary>
///     Creates a task that will complete successfully when all of the tasks supplied as arguments complete successfully.
/// </summary>
/// <typeparam name="_Iterator">
///     The type of the input iterator.
/// </typeparam>
/// <param name="_Begin">
///     The position of the first element in the range of elements to be combined into the resulting task.
/// </param>
/// <param name="_End">
///     The position of the first element beyond the range of elements to be combined into the resulting task.
/// </param>
/// <returns>
///     A task that completes sucessfully when all of the input tasks have completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;&gt;</c>. If the input tasks are of type <c>void</c> the output
///     task will also be a <c>task&lt;void&gt;</c>.
/// </returns>
/// <remarks>
///     If one of the tasks is canceled or throws an exception, the returned task will complete early, in the canceled state, and the exception,
///     if one is encoutered, will be thrown if you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template <typename _Iterator>
auto when_all(_Iterator _Begin, _Iterator _End)
-> decltype (details::_WhenAllImpl<typename std::iterator_traits<_Iterator>::value_type::result_type, _Iterator>::_Perform(nullptr, _Begin, _End))
{
    typedef typename std::iterator_traits<_Iterator>::value_type::result_type _ElementType;
    return details::_WhenAllImpl<_ElementType, _Iterator>::_Perform(nullptr, _Begin, _End);
}
#endif

/// <summary>
///     Creates a task that will complete successfully when all of the tasks supplied as arguments complete successfully.
/// </summary>
/// <typeparam name="_Iterator">
///     The type of the input iterator.
/// </typeparam>
/// <param name="_Begin">
///     The position of the first element in the range of elements to be combined into the resulting task.
/// </param>
/// <param name="_End">
///     The position of the first element beyond the range of elements to be combined into the resulting task.
/// </param>
/// <param name="_CancellationToken">
///     The cancellation token which controls cancellation of the returned task. If you do not provide a cancellation token, the resulting
///     task will be created with a token that is a combination of all the cancelable tokens (tokens created by methods other than
///     <c>cancellation_token::none()</c>of the tasks supplied.
/// </param>
/// <returns>
///     A task that completes sucessfully when all of the input tasks have completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;&gt;</c>. If the input tasks are of type <c>void</c> the output
///     task will also be a <c>task&lt;void&gt;</c>.
/// </returns>
/// <remarks>
///     If one of the tasks is canceled or throws an exception, the returned task will complete early, in the canceled state, and the exception,
///     if one is encoutered, will be thrown if you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template <typename _Iterator>
#if _MSC_VER >= 1800
auto when_all(_Iterator _Begin, _Iterator _End, const task_options& _TaskOptions = task_options())
-> decltype (details::_WhenAllImpl<typename std::iterator_traits<_Iterator>::value_type::result_type, _Iterator>::_Perform(_TaskOptions, _Begin, _End))
{
    typedef typename std::iterator_traits<_Iterator>::value_type::result_type _ElementType;
    return details::_WhenAllImpl<_ElementType, _Iterator>::_Perform(_TaskOptions, _Begin, _End);
}
#else
auto when_all(_Iterator _Begin, _Iterator _End, Concurrency::cancellation_token _CancellationToken)
-> decltype (details::_WhenAllImpl<typename std::iterator_traits<_Iterator>::value_type::result_type, _Iterator>::_Perform(_CancellationToken._GetImplValue(), _Begin, _End))
{
    typedef typename std::iterator_traits<_Iterator>::value_type::result_type _ElementType;
    return details::_WhenAllImpl<_ElementType, _Iterator>::_Perform(_CancellationToken._GetImplValue(), _Begin, _End);
}
#endif

/// <summary>
///     Creates a task that will complete succesfully when both of the tasks supplied as arguments complete successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes successfully when both of the input tasks have completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;&gt;</c>. If the input tasks are of type <c>void</c> the output
///     task will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA &amp;&amp; taskB &amp;&amp; taskC, which are combined in pairs, the &amp;&amp; operator
///     produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if either one or both of the tasks are of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>.</para>
/// </returns>
/// <remarks>
///     If one of the tasks is canceled or throws an exception, the returned task will complete early, in the canceled state, and the exception,
///     if one is encoutered, will be thrown if you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType>
task<std::vector<_ReturnType>> operator&&(const task<_ReturnType> & _Lhs, const task<_ReturnType> & _Rhs)
{
    task<_ReturnType> _PTasks[2] = { _Lhs, _Rhs };
    return when_all(_PTasks, _PTasks + 2);
}

/// <summary>
///     Creates a task that will complete succesfully when both of the tasks supplied as arguments complete successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes successfully when both of the input tasks have completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;&gt;</c>. If the input tasks are of type <c>void</c> the output
///     task will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA &amp;&amp; taskB &amp;&amp; taskC, which are combined in pairs, the &amp;&amp; operator
///     produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if either one or both of the tasks are of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>.</para>
/// </returns>
/// <remarks>
///     If one of the tasks is canceled or throws an exception, the returned task will complete early, in the canceled state, and the exception,
///     if one is encoutered, will be thrown if you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType>
task<std::vector<_ReturnType>> operator&&(const task<std::vector<_ReturnType>> & _Lhs, const task<_ReturnType> & _Rhs)
{
    return details::_WhenAllVectorAndValue(_Lhs, _Rhs, true);
}

/// <summary>
///     Creates a task that will complete succesfully when both of the tasks supplied as arguments complete successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes successfully when both of the input tasks have completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;&gt;</c>. If the input tasks are of type <c>void</c> the output
///     task will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA &amp;&amp; taskB &amp;&amp; taskC, which are combined in pairs, the &amp;&amp; operator
///     produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if either one or both of the tasks are of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>.</para>
/// </returns>
/// <remarks>
///     If one of the tasks is canceled or throws an exception, the returned task will complete early, in the canceled state, and the exception,
///     if one is encoutered, will be thrown if you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType>
task<std::vector<_ReturnType>> operator&&(const task<_ReturnType> & _Lhs, const task<std::vector<_ReturnType>> & _Rhs)
{
    return details::_WhenAllVectorAndValue(_Rhs, _Lhs, false);
}

/// <summary>
///     Creates a task that will complete succesfully when both of the tasks supplied as arguments complete successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes successfully when both of the input tasks have completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;&gt;</c>. If the input tasks are of type <c>void</c> the output
///     task will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA &amp;&amp; taskB &amp;&amp; taskC, which are combined in pairs, the &amp;&amp; operator
///     produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if either one or both of the tasks are of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>.</para>
/// </returns>
/// <remarks>
///     If one of the tasks is canceled or throws an exception, the returned task will complete early, in the canceled state, and the exception,
///     if one is encoutered, will be thrown if you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType>
task<std::vector<_ReturnType>> operator&&(const task<std::vector<_ReturnType>> & _Lhs, const task<std::vector<_ReturnType>> & _Rhs)
{
    task<std::vector<_ReturnType>> _PTasks[2] = { _Lhs, _Rhs };
    return when_all(_PTasks, _PTasks + 2);
}

/// <summary>
///     Creates a task that will complete succesfully when both of the tasks supplied as arguments complete successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes successfully when both of the input tasks have completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;&gt;</c>. If the input tasks are of type <c>void</c> the output
///     task will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA &amp;&amp; taskB &amp;&amp; taskC, which are combined in pairs, the &amp;&amp; operator
///     produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if either one or both of the tasks are of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>.</para>
/// </returns>
/// <remarks>
///     If one of the tasks is canceled or throws an exception, the returned task will complete early, in the canceled state, and the exception,
///     if one is encoutered, will be thrown if you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
inline task<void> operator&&(const task<void> & _Lhs, const task<void> & _Rhs)
{
    task<void> _PTasks[2] = { _Lhs, _Rhs };
    return when_all(_PTasks, _PTasks + 2);
}

namespace details
{
    // Helper struct for when_any operators to know when tasks have completed
    template <typename _CompletionType>
    struct _RunAnyParam
    {
        _RunAnyParam() : _M_completeCount(0), _M_numTasks(0), _M_exceptionRelatedToken(nullptr), _M_fHasExplicitToken(false)
        {
        }
        ~_RunAnyParam()
        {
            if (Concurrency::details::_CancellationTokenState::_IsValid(_M_exceptionRelatedToken))
                _M_exceptionRelatedToken->_Release();
        }
        task_completion_event<_CompletionType>      _M_Completed;
        Concurrency::cancellation_token_source                   _M_cancellationSource;
        Concurrency::details::_CancellationTokenState*            _M_exceptionRelatedToken;
        atomic_size_t         _M_completeCount;
        size_t                                      _M_numTasks;
        bool                                        _M_fHasExplicitToken;
    };

    template<typename _CompletionType, typename _Function, typename _TaskType>
    void _WhenAnyContinuationWrapper(_RunAnyParam<_CompletionType> * _PParam, const _Function & _Func, task<_TaskType>& _Task)
    {
        bool _IsTokenCancled = !_PParam->_M_fHasExplicitToken && _Task._GetImpl()->_M_pTokenState != Concurrency::details::_CancellationTokenState::_None() && _Task._GetImpl()->_M_pTokenState->_IsCanceled();
        if (_Task._GetImpl()->_IsCompleted() && !_IsTokenCancled)
        {
            _Func();
#if _MSC_VER >= 1800
            if (Concurrency::details::atomic_increment(_PParam->_M_completeCount) == _PParam->_M_numTasks)
#else
            if (_InterlockedIncrementSizeT(&_PParam->_M_completeCount) == _PParam->_M_numTasks)
#endif
            {
                delete _PParam;
            }
        }
        else
        {
            _CONCRT_ASSERT(_Task._GetImpl()->_IsCanceled() || _IsTokenCancled);
            if (_Task._GetImpl()->_HasUserException() && !_IsTokenCancled)
            {
                if (_PParam->_M_Completed._StoreException(_Task._GetImpl()->_GetExceptionHolder()))
                {
                    // This can only enter once.
                    _PParam->_M_exceptionRelatedToken = _Task._GetImpl()->_M_pTokenState;
                    _CONCRT_ASSERT(_PParam->_M_exceptionRelatedToken);
                    // Deref token will be done in the _PParam destructor.
                    if (_PParam->_M_exceptionRelatedToken != Concurrency::details::_CancellationTokenState::_None())
                    {
                        _PParam->_M_exceptionRelatedToken->_Reference();
                    }
                }
            }

#if _MSC_VER >= 1800
            if (Concurrency::details::atomic_increment(_PParam->_M_completeCount) == _PParam->_M_numTasks)
#else
            if (_InterlockedIncrementSizeT(&_PParam->_M_completeCount) == _PParam->_M_numTasks)
#endif
            {
                // If no one has be completed so far, we need to make some final cancellation decision.
                if (!_PParam->_M_Completed._IsTriggered())
                {
                    // If we already explicit token, we can skip the token join part.
                    if (!_PParam->_M_fHasExplicitToken)
                    {
                        if (_PParam->_M_exceptionRelatedToken)
                        {
                            details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, _PParam->_M_exceptionRelatedToken);
                        }
                        else
                        {
                            // If haven't captured any exception token yet, there was no exception for all those tasks,
                            // so just pick a random token (current one) for normal cancellation.
                            details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, _Task._GetImpl()->_M_pTokenState);
                        }
                    }
                    // Do exception cancellation or normal cancellation based on whether it has stored exception.
                    _PParam->_M_Completed._Cancel();
                }
                delete _PParam;
            }
        }
    }

    template<typename _ElementType, typename _Iterator>
    struct _WhenAnyImpl
    {
#if _MSC_VER >= 1800
        static task<std::pair<_ElementType, size_t>> _Perform(const task_options& _TaskOptions, _Iterator _Begin, _Iterator _End)
#else
        static task<std::pair<_ElementType, size_t>> _Perform(Concurrency::details::_CancellationTokenState *_PTokenState, _Iterator _Begin, _Iterator _End)
#endif
        {
            if (_Begin == _End)
            {
                throw Concurrency::invalid_operation("when_any(begin, end) cannot be called on an empty container.");
            }
#if _MSC_VER >= 1800
            Concurrency::details::_CancellationTokenState *_PTokenState = _TaskOptions.has_cancellation_token() ? _TaskOptions.get_cancellation_token()._GetImplValue() : nullptr;
#endif
            auto _PParam = new _RunAnyParam<std::pair<std::pair<_ElementType, size_t>, Concurrency::details::_CancellationTokenState *>>();

            if (_PTokenState)
            {
                details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, _PTokenState);
                _PParam->_M_fHasExplicitToken = true;
            }
#if _MSC_VER >= 1800
            task_options _Options(_TaskOptions);
            _Options.set_cancellation_token(_PParam->_M_cancellationSource.get_token());
            task<std::pair<std::pair<_ElementType, size_t>, Concurrency::details::_CancellationTokenState *>> _Any_tasks_completed(_PParam->_M_Completed, _Options);
#else
            task<std::pair<std::pair<_ElementType, size_t>, Concurrency::details::_CancellationTokenState *>> _Any_tasks_completed(_PParam->_M_Completed, _PParam->_M_cancellationSource.get_token());
            _Any_tasks_completed._GetImpl()->_M_fRuntimeAggregate = true;
#endif
            // Keep a copy ref to the token source
            auto _CancellationSource = _PParam->_M_cancellationSource;

            _PParam->_M_numTasks = static_cast<size_t>(std::distance(_Begin, _End));
            size_t index = 0;
            for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
            {
                if (_PTask->is_apartment_aware())
                {
                    _Any_tasks_completed._SetAsync();
                }

                _PTask->_Then([_PParam, index](task<_ElementType> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
                    auto _PParamCopy = _PParam; // Dev10
                    auto _IndexCopy = index; // Dev10
                    auto _Func = [&_ResultTask, _PParamCopy, _IndexCopy]() {
                        _PParamCopy->_M_Completed.set(std::make_pair(std::make_pair(_ResultTask._GetImpl()->_GetResult(), _IndexCopy), _ResultTask._GetImpl()->_M_pTokenState));
                    };
#else
                    auto _Func = [&_ResultTask, _PParam, index]() {
                        _PParam->_M_Completed.set(std::make_pair(std::make_pair(_ResultTask._GetImpl()->_GetResult(), index), _ResultTask._GetImpl()->_M_pTokenState));
                    };
#endif
                    _WhenAnyContinuationWrapper(_PParam, _Func, _ResultTask);
                    return S_OK;
#if _MSC_VER >= 1800
                }, Concurrency::details::_CancellationTokenState::_None());
#else
                }, Concurrency::details::_CancellationTokenState::_None(), false);
#endif
                index++;
            }

            // All _Any_tasks_completed._SetAsync() must be finished before this return continuation task being created.
            return _Any_tasks_completed._Then([=](std::pair<std::pair<_ElementType, size_t>, Concurrency::details::_CancellationTokenState *> _Result, std::pair<_ElementType, size_t>* retVal) -> HRESULT {
                _CONCRT_ASSERT(_Result.second);
                if (!_PTokenState)
                {
                    details::_JoinAllTokens_Add(_CancellationSource, _Result.second);
                }
                *retVal = _Result.first;
                return S_OK;
#if _MSC_VER >= 1800
            }, nullptr);
#else
            }, nullptr, true);
#endif
        }
    };

    template<typename _Iterator>
    struct _WhenAnyImpl<void, _Iterator>
    {
#if _MSC_VER >= 1800
        static task<size_t> _Perform(const task_options& _TaskOptions, _Iterator _Begin, _Iterator _End)
#else
        static task<size_t> _Perform(Concurrency::details::_CancellationTokenState *_PTokenState, _Iterator _Begin, _Iterator _End)
#endif
        {
            if (_Begin == _End)
            {
                throw Concurrency::invalid_operation("when_any(begin, end) cannot be called on an empty container.");
            }
#if _MSC_VER >= 1800
            Concurrency::details::_CancellationTokenState *_PTokenState = _TaskOptions.has_cancellation_token() ? _TaskOptions.get_cancellation_token()._GetImplValue() : nullptr;
#endif
            auto _PParam = new _RunAnyParam<std::pair<size_t, Concurrency::details::_CancellationTokenState *>>();

            if (_PTokenState)
            {
                details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, _PTokenState);
                _PParam->_M_fHasExplicitToken = true;
            }

#if _MSC_VER >= 1800
            task_options _Options(_TaskOptions);
            _Options.set_cancellation_token(_PParam->_M_cancellationSource.get_token());
            task<std::pair<size_t, _CancellationTokenState *>> _Any_tasks_completed(_PParam->_M_Completed, _Options);
#else
            task<std::pair<size_t, Concurrency::details::_CancellationTokenState *>> _Any_tasks_completed(_PParam->_M_Completed, _PParam->_M_cancellationSource.get_token());
#endif
            // Keep a copy ref to the token source
            auto _CancellationSource = _PParam->_M_cancellationSource;

            _PParam->_M_numTasks = static_cast<size_t>(std::distance(_Begin, _End));
            size_t index = 0;
            for (auto _PTask = _Begin; _PTask != _End; ++_PTask)
            {
                if (_PTask->is_apartment_aware())
                {
                    _Any_tasks_completed._SetAsync();
                }

                _PTask->_Then([_PParam, index](task<void> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
                    auto _PParamCopy = _PParam; // Dev10
                    auto _IndexCopy = index; // Dev10
                    auto _Func = [&_ResultTask, _PParamCopy, _IndexCopy]() {
                        _PParamCopy->_M_Completed.set(std::make_pair(_IndexCopy, _ResultTask._GetImpl()->_M_pTokenState));
                    };
#else
                    auto _Func = [&_ResultTask, _PParam, index]() {
                        _PParam->_M_Completed.set(std::make_pair(index, _ResultTask._GetImpl()->_M_pTokenState));
                    };
#endif
                    _WhenAnyContinuationWrapper(_PParam, _Func, _ResultTask);
                    return S_OK;
#if _MSC_VER >= 1800
                }, Concurrency::details::_CancellationTokenState::_None());
#else
                }, Concurrency::details::_CancellationTokenState::_None(), false);
#endif

                index++;
            }

            // All _Any_tasks_completed._SetAsync() must be finished before this return continuation task being created.
            return _Any_tasks_completed._Then([=](std::pair<size_t, Concurrency::details::_CancellationTokenState *> _Result, size_t* retVal) -> HRESULT {
                _CONCRT_ASSERT(_Result.second);
                if (!_PTokenState)
                {
                    details::_JoinAllTokens_Add(_CancellationSource, _Result.second);
                }
                *retVal = _Result.first;
                return S_OK;
#if _MSC_VER >= 1800
            }, nullptr);
#else
            }, nullptr, false);
#endif
        }
    };
} // namespace details

/// <summary>
///     Creates a task that will complete successfully when any of the tasks supplied as arguments completes successfully.
/// </summary>
/// <typeparam name="_Iterator">
///     The type of the input iterator.
/// </typeparam>
/// <param name="_Begin">
///     The position of the first element in the range of elements to be combined into the resulting task.
/// </param>
/// <param name="_End">
///     The position of the first element beyond the range of elements to be combined into the resulting task.
/// </param>
/// <returns>
///     A task that completes successfully when any one of the input tasks has completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::pair&lt;T, size_t&gt;&gt;></c>, where the first element of the pair is the result
///     of the completing task, and the second element is the index of the task that finished. If the input tasks are of type <c>void</c>
///     the output is a <c>task&lt;size_t&gt;</c>, where the result is the index of the completing task.
/// </returns>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _Iterator>
#if _MSC_VER >= 1800
auto when_any(_Iterator _Begin, _Iterator _End, const task_options& _TaskOptions = task_options())
-> decltype (details::_WhenAnyImpl<typename std::iterator_traits<_Iterator>::value_type::result_type, _Iterator>::_Perform(_TaskOptions, _Begin, _End))
{
    typedef typename std::iterator_traits<_Iterator>::value_type::result_type _ElementType;
    return details::_WhenAnyImpl<_ElementType, _Iterator>::_Perform(_TaskOptions, _Begin, _End);
}
#else
auto when_any(_Iterator _Begin, _Iterator _End)
-> decltype (details::_WhenAnyImpl<typename std::iterator_traits<_Iterator>::value_type::result_type, _Iterator>::_Perform(nullptr, _Begin, _End))
{
    typedef typename std::iterator_traits<_Iterator>::value_type::result_type _ElementType;
    return details::_WhenAnyImpl<_ElementType, _Iterator>::_Perform(nullptr, _Begin, _End);
}
#endif

/// <summary>
///     Creates a task that will complete successfully when any of the tasks supplied as arguments completes successfully.
/// </summary>
/// <typeparam name="_Iterator">
///     The type of the input iterator.
/// </typeparam>
/// <param name="_Begin">
///     The position of the first element in the range of elements to be combined into the resulting task.
/// </param>
/// <param name="_End">
///     The position of the first element beyond the range of elements to be combined into the resulting task.
/// </param>
/// <param name="_CancellationToken">
///     The cancellation token which controls cancellation of the returned task. If you do not provide a cancellation token, the resulting
///     task will receive the cancellation token of the task that causes it to complete.
/// </param>
/// <returns>
///     A task that completes successfully when any one of the input tasks has completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::pair&lt;T, size_t&gt;&gt;></c>, where the first element of the pair is the result
///     of the completing task, and the second element is the index of the task that finished. If the input tasks are of type <c>void</c>
///     the output is a <c>task&lt;size_t&gt;</c>, where the result is the index of the completing task.
/// </returns>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _Iterator>
auto when_any(_Iterator _Begin, _Iterator _End, Concurrency::cancellation_token _CancellationToken)
-> decltype (details::_WhenAnyImpl<typename std::iterator_traits<_Iterator>::value_type::result_type, _Iterator>::_Perform(_CancellationToken._GetImplValue(), _Begin, _End))
{
    typedef typename std::iterator_traits<_Iterator>::value_type::result_type _ElementType;
    return details::_WhenAnyImpl<_ElementType, _Iterator>::_Perform(_CancellationToken._GetImplValue(), _Begin, _End);
}

/// <summary>
///     Creates a task that will complete successfully when either of the tasks supplied as arguments completes successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes sucessfully when either of the input tasks has completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;</c>. If the input tasks are of type <c>void</c> the output task
///     will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA || taskB &amp;&amp; taskC, which are combined in pairs, with &amp;&amp; taking precedence
///     over ||, the operator|| produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if one of the tasks is of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>
///     and the other one is of type <c>task&lt;T&gt;.</para>
/// </returns>
/// <remarks>
///     If both of the tasks are canceled or throw exceptions, the returned task will complete in the canceled state, and one of the exceptions,
///     if any are encountered, will be thrown when you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType>
task<_ReturnType> operator||(const task<_ReturnType> & _Lhs, const task<_ReturnType> & _Rhs)
{
#if _MSC_VER >= 1800
    auto _PParam = new details::_RunAnyParam<std::pair<_ReturnType, size_t>>();

    task<std::pair<_ReturnType, size_t>> _Any_tasks_completed(_PParam->_M_Completed, _PParam->_M_cancellationSource.get_token());
    // Chain the return continuation task here to ensure it will get inline execution when _M_Completed.set is called,
    // So that _PParam can be used before it getting deleted.
    auto _ReturnTask = _Any_tasks_completed._Then([=](std::pair<_ReturnType, size_t> _Ret, _ReturnType* retVal) -> HRESULT {
        _CONCRT_ASSERT(_Ret.second);
        details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, reinterpret_cast<Concurrency::details::_CancellationTokenState *>(_Ret.second));
        *retVal = _Ret.first;
        return S_OK;
    }, nullptr);
#else
    auto _PParam = new details::_RunAnyParam<std::pair<_ReturnType, Concurrency::details::_CancellationTokenState *>>();

    task<std::pair<_ReturnType, Concurrency::details::_CancellationTokenState *>> _Any_tasks_completed(_PParam->_M_Completed, _PParam->_M_cancellationSource.get_token());
    // Chain the return continuation task here to ensure it will get inline execution when _M_Completed.set is called,
    // So that _PParam can be used before it getting deleted.
    auto _ReturnTask = _Any_tasks_completed._Then([=](std::pair<_ReturnType, Concurrency::details::_CancellationTokenState *> _Ret, _ReturnType* retVal) -> HRESULT {
        _CONCRT_ASSERT(_Ret.second);
        details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, _Ret.second);
        *retVal = _Ret.first;
        return S_OK;
    }, nullptr, false);
#endif
    if (_Lhs.is_apartment_aware() || _Rhs.is_apartment_aware())
    {
        _ReturnTask._SetAsync();
    }

    _PParam->_M_numTasks = 2;
    auto _Continuation = [_PParam](task<_ReturnType> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
        //  Dev10 compiler bug
        auto _PParamCopy = _PParam;
        auto _Func = [&_ResultTask, _PParamCopy]() {
            _PParamCopy->_M_Completed.set(std::make_pair(_ResultTask._GetImpl()->_GetResult(), reinterpret_cast<size_t>(_ResultTask._GetImpl()->_M_pTokenState)));
        };
#else
        auto _Func = [&_ResultTask, _PParam]() {
            _PParam->_M_Completed.set(std::make_pair(_ResultTask._GetImpl()->_GetResult(), _ResultTask._GetImpl()->_M_pTokenState));
        };
#endif
        _WhenAnyContinuationWrapper(_PParam, _Func, _ResultTask);
        return S_OK;
    };

#if _MSC_VER >= 1800
    _Lhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None());
    _Rhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None());
#else
    _Lhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None(), false);
    _Rhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None(), false);
#endif
    return _ReturnTask;
}

/// <summary>
///     Creates a task that will complete successfully when any of the tasks supplied as arguments completes successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes sucessfully when either of the input tasks has completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;</c>. If the input tasks are of type <c>void</c> the output task
///     will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA || taskB &amp;&amp; taskC, which are combined in pairs, with &amp;&amp; taking precedence
///     over ||, the operator|| produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if one of the tasks is of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>
///     and the other one is of type <c>task&lt;T&gt;.</para>
/// </returns>
/// <remarks>
///     If both of the tasks are canceled or throw exceptions, the returned task will complete in the canceled state, and one of the exceptions,
///     if any are encountered, will be thrown when you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType>
task<std::vector<_ReturnType>> operator||(const task<std::vector<_ReturnType>> & _Lhs, const task<_ReturnType> & _Rhs)
{
    auto _PParam = new details::_RunAnyParam<std::pair<std::vector<_ReturnType>, Concurrency::details::_CancellationTokenState *>>();

    task<std::pair<std::vector<_ReturnType>, Concurrency::details::_CancellationTokenState *>> _Any_tasks_completed(_PParam->_M_Completed, _PParam->_M_cancellationSource.get_token());
#if _MSC_VER < 1800
    _Any_tasks_completed._GetImpl()->_M_fRuntimeAggregate = true;
#endif
    // Chain the return continuation task here to ensure it will get inline execution when _M_Completed.set is called,
    // So that _PParam can be used before it getting deleted.
    auto _ReturnTask = _Any_tasks_completed._Then([=](std::pair<std::vector<_ReturnType>, Concurrency::details::_CancellationTokenState *> _Ret, std::vector<_ReturnType>* retVal) -> HRESULT {
        _CONCRT_ASSERT(_Ret.second);
        details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, _Ret.second);
        *retVal = _Ret.first;
        return S_OK;
    }, nullptr, true);

    if (_Lhs.is_apartment_aware() || _Rhs.is_apartment_aware())
    {
        _ReturnTask._SetAsync();
    }

    _PParam->_M_numTasks = 2;
    _Lhs._Then([_PParam](task<std::vector<_ReturnType>> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
        //  Dev10 compiler bug
        auto _PParamCopy = _PParam;
        auto _Func = [&_ResultTask, _PParamCopy]() {
            auto _Result = _ResultTask._GetImpl()->_GetResult();
            _PParamCopy->_M_Completed.set(std::make_pair(_Result, _ResultTask._GetImpl()->_M_pTokenState));
        };
#else
        auto _Func = [&_ResultTask, _PParam]() {
            std::vector<_ReturnType> _Result = _ResultTask._GetImpl()->_GetResult();
            _PParam->_M_Completed.set(std::make_pair(_Result, _ResultTask._GetImpl()->_M_pTokenState));
        };
#endif
        _WhenAnyContinuationWrapper(_PParam, _Func, _ResultTask);
        return S_OK;
#if _MSC_VER >= 1800
    }, Concurrency::details::_CancellationTokenState::_None());
#else
    }, Concurrency::details::_CancellationTokenState::_None(), false);
#endif
    _Rhs._Then([_PParam](task<_ReturnType> _ResultTask) -> HRESULT {
#if _MSC_VER >= 1800
        //  Dev10 compiler bug
        typedef _ReturnType _ReturnTypeDev10;
        auto _PParamCopy = _PParam;
        auto _Func = [&_ResultTask, _PParamCopy]() {
            auto _Result = _ResultTask._GetImpl()->_GetResult();

            std::vector<_ReturnTypeDev10> _Vec;
            _Vec.push_back(_Result);
            _PParamCopy->_M_Completed.set(std::make_pair(_Vec, _ResultTask._GetImpl()->_M_pTokenState));
        };
#else
        auto _Func = [&_ResultTask, _PParam]() {
            _ReturnType _Result = _ResultTask._GetImpl()->_GetResult();

            std::vector<_ReturnType> _Vec;
            _Vec.push_back(_Result);
            _PParam->_M_Completed.set(std::make_pair(_Vec, _ResultTask._GetImpl()->_M_pTokenState));
        };
#endif
        _WhenAnyContinuationWrapper(_PParam, _Func, _ResultTask);
        return S_OK;
#if _MSC_VER >= 1800
    }, Concurrency::details::_CancellationTokenState::_None());
#else
    }, Concurrency::details::_CancellationTokenState::_None(), false);
#endif
    return _ReturnTask;
}

/// <summary>
///     Creates a task that will complete successfully when any of the tasks supplied as arguments completes successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes sucessfully when either of the input tasks has completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;</c>. If the input tasks are of type <c>void</c> the output task
///     will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA || taskB &amp;&amp; taskC, which are combined in pairs, with &amp;&amp; taking precedence
///     over ||, the operator|| produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if one of the tasks is of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>
///     and the other one is of type <c>task&lt;T&gt;.</para>
/// </returns>
/// <remarks>
///     If both of the tasks are canceled or throw exceptions, the returned task will complete in the canceled state, and one of the exceptions,
///     if any are encountered, will be thrown when you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
template<typename _ReturnType>
task<std::vector<_ReturnType>> operator||(const task<_ReturnType> & _Lhs, const task<std::vector<_ReturnType>> & _Rhs)
{
    return _Rhs || _Lhs;
}

/// <summary>
///     Creates a task that will complete successfully when any of the tasks supplied as arguments completes successfully.
/// </summary>
/// <typeparam name="_ReturnType">
///     The type of the returned task.
/// </typeparam>
/// <param name="_Lhs">
///     The first task to combine into the resulting task.
/// </param>
/// <param name="_Rhs">
///     The second task to combine into the resulting task.
/// </param>
/// <returns>
///     A task that completes sucessfully when either of the input tasks has completed successfully. If the input tasks are of type <c>T</c>,
///     the output of this function will be a <c>task&lt;std::vector&lt;T&gt;</c>. If the input tasks are of type <c>void</c> the output task
///     will also be a <c>task&lt;void&gt;</c>.
///     <para> To allow for a construct of the sort taskA || taskB &amp;&amp; taskC, which are combined in pairs, with &amp;&amp; taking precedence
///     over ||, the operator|| produces a <c>task&lt;std::vector&lt;T&gt;&gt;</c> if one of the tasks is of type <c>task&lt;std::vector&lt;T&gt;&gt;</c>
///     and the other one is of type <c>task&lt;T&gt;.</para>
/// </returns>
/// <remarks>
///     If both of the tasks are canceled or throw exceptions, the returned task will complete in the canceled state, and one of the exceptions,
///     if any are encountered, will be thrown when you call <c>get()</c> or <c>wait()</c> on that task.
/// </remarks>
/// <seealso cref="Task Parallelism (Concurrency Runtime)"/>
/**/
inline task<void> operator||(const task<void> & _Lhs, const task<void> & _Rhs)
{
    auto _PParam = new details::_RunAnyParam<std::pair<details::_Unit_type, Concurrency::details::_CancellationTokenState *>>();

    task<std::pair<details::_Unit_type, Concurrency::details::_CancellationTokenState *>> _Any_task_completed(_PParam->_M_Completed, _PParam->_M_cancellationSource.get_token());
    // Chain the return continuation task here to ensure it will get inline execution when _M_Completed.set is called,
    // So that _PParam can be used before it getting deleted.
    auto _ReturnTask = _Any_task_completed._Then([=](std::pair<details::_Unit_type, Concurrency::details::_CancellationTokenState *> _Ret) -> HRESULT {
        _CONCRT_ASSERT(_Ret.second);
        details::_JoinAllTokens_Add(_PParam->_M_cancellationSource, _Ret.second);
        return S_OK;
#if _MSC_VER >= 1800
    }, nullptr);
#else
    }, nullptr, false);
#endif

    if (_Lhs.is_apartment_aware() || _Rhs.is_apartment_aware())
    {
        _ReturnTask._SetAsync();
    }

    _PParam->_M_numTasks = 2;
    auto _Continuation = [_PParam](task<void> _ResultTask) mutable -> HRESULT {
        //  Dev10 compiler needs this.
        auto _PParam1 = _PParam;
        auto _Func = [&_ResultTask, _PParam1]() {
            _PParam1->_M_Completed.set(std::make_pair(details::_Unit_type(), _ResultTask._GetImpl()->_M_pTokenState));
        };
        _WhenAnyContinuationWrapper(_PParam, _Func, _ResultTask);
        return S_OK;
    };

#if _MSC_VER >= 1800
    _Lhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None());
    _Rhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None());
#else
    _Lhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None(), false);
    _Rhs._Then(_Continuation, Concurrency::details::_CancellationTokenState::_None(), false);
#endif

    return _ReturnTask;
}

#if _MSC_VER >= 1800
template<typename _Ty>
task<_Ty> task_from_result(_Ty _Param, const task_options& _TaskOptions = task_options())
{
    task_completion_event<_Ty> _Tce;
    _Tce.set(_Param);
    return create_task<_Ty>(_Tce, _TaskOptions);
}

// Work around VS 2010 compiler bug
#if _MSC_VER == 1600
inline task<bool> task_from_result(bool _Param)
{
    task_completion_event<bool> _Tce;
    _Tce.set(_Param);
    return create_task<bool>(_Tce, task_options());
}
#endif
inline task<void> task_from_result(const task_options& _TaskOptions = task_options())
{
    task_completion_event<void> _Tce;
    _Tce.set();
    return create_task<void>(_Tce, _TaskOptions);
}

template<typename _TaskType, typename _ExType>
task<_TaskType> task_from_exception(_ExType _Exception, const task_options& _TaskOptions = task_options())
{
    task_completion_event<_TaskType> _Tce;
    _Tce.set_exception(_Exception);
    return create_task<_TaskType>(_Tce, _TaskOptions);
}

namespace details
{
    /// <summary>
    /// A convenient extension to Concurrency: loop until a condition is no longer met
    /// </summary>
    /// <param name="func">
    ///   A function representing the body of the loop. It will be invoked at least once and
    ///   then repetitively as long as it returns true.
    /// </param>
    inline
    task<bool> do_while(std::function<task<bool>(void)> func)
    {
            task<bool> first = func();
            return first.then([=](bool guard, task<bool>* retVal) -> HRESULT {
                if (guard)
                    *retVal = do_while(func);
                else
                    *retVal = first;
                return S_OK;
            });
    }

} // namespace details
#endif

} // namespace Concurrency_winrt

namespace concurrency_winrt = Concurrency_winrt;

#pragma pop_macro("new")
#pragma warning(pop)
#pragma pack(pop)
#endif

#endif
