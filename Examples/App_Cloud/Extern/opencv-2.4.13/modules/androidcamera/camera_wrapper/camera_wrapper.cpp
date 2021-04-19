#if !defined(ANDROID_r2_2_0) && !defined(ANDROID_r2_3_3) && !defined(ANDROID_r3_0_1) && \
 !defined(ANDROID_r4_0_0) && !defined(ANDROID_r4_0_3) && !defined(ANDROID_r4_1_1) && \
 !defined(ANDROID_r4_2_0) && !defined(ANDROID_r4_3_0) && !defined(ANDROID_r4_4_0)
# error Building camera wrapper for your version of Android is not supported by OpenCV.\
 You need to modify OpenCV sources in order to compile camera wrapper for your version of Android.
#endif

#include <camera/Camera.h>
#include <camera/CameraParameters.h>

#if defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
# include <system/camera.h>
#endif //defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)

#include "camera_wrapper.h"
#include "../include/camera_properties.h"

#if defined(ANDROID_r3_0_1) || defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3) || defined(ANDROID_r4_1_1)
//Include SurfaceTexture.h file with the SurfaceTexture class
# include <gui/SurfaceTexture.h>
# define MAGIC_OPENCV_TEXTURE_ID (0x10)
#elif defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0)
# include <gui/ISurface.h>
# include <gui/BufferQueue.h>
#elif defined(ANDROID_r4_3_0) || defined(ANDROID_r4_4_0)
# include <gui/IGraphicBufferProducer.h>
# include <gui/BufferQueue.h>
# include <ui/GraphicBuffer.h>
#else
# include <surfaceflinger/ISurface.h>
#endif

#include <string>
#include <fstream>

//undef logging macro from /system/core/libcutils/loghack.h
#ifdef LOGD
# undef LOGD
#endif

#ifdef LOGI
# undef LOGI
#endif

#ifdef LOGW
# undef LOGW
#endif

#ifdef LOGE
# undef LOGE
#endif

// LOGGING
#include <android/log.h>
#define CAMERA_LOG_TAG "OpenCV_NativeCamera"
#define LOGD(...) ((void)__android_log_print(ANDROID_LOG_DEBUG, CAMERA_LOG_TAG, __VA_ARGS__))
#define LOGI(...) ((void)__android_log_print(ANDROID_LOG_INFO, CAMERA_LOG_TAG, __VA_ARGS__))
#define LOGW(...) ((void)__android_log_print(ANDROID_LOG_WARN, CAMERA_LOG_TAG, __VA_ARGS__))
#define LOGE(...) ((void)__android_log_print(ANDROID_LOG_ERROR, CAMERA_LOG_TAG, __VA_ARGS__))

#include <dlfcn.h>

using namespace android;

// non-public camera related classes are not binary compatible
// objects of these classes have different sizeof on different platforms
// additional memory tail to all system objects to overcome sizeof issue
#define MAGIC_TAIL 4096


void debugShowFPS();

#if defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0) || defined(ANDROID_r4_3_0)
class ConsumerListenerStub: public BufferQueue::ConsumerListener
{
public:
    virtual void onFrameAvailable()
    {
    }
    virtual void onBuffersReleased()
    {
    }
};
#elif defined(ANDROID_r4_4_0)
class ConsumerListenerStub: public android::BnConsumerListener
{
public:
    virtual void onFrameAvailable()
    {
    }
    virtual void onBuffersReleased()
    {
    }
    virtual ~ConsumerListenerStub()
    {
    }
};
#endif


std::string getProcessName()
{
    std::string result;
    std::ifstream f;

    f.open("/proc/self/cmdline");
    if (f.is_open())
    {
        std::string fullPath;
        std::getline(f, fullPath, '\0');
        if (!fullPath.empty())
        {
            int i = fullPath.size()-1;
            while ((i >= 0) && (fullPath[i] != '/')) i--;
            result = fullPath.substr(i+1, std::string::npos);
        }
    }

    f.close();

    return result;
}

void debugShowFPS()
{
    static int mFrameCount = 0;
    static int mLastFrameCount = 0;
    static nsecs_t mLastFpsTime = systemTime();
    static float mFps = 0;

    mFrameCount++;

    if (( mFrameCount % 30 ) != 0)
        return;

    nsecs_t now = systemTime();
    nsecs_t diff = now - mLastFpsTime;

    if (diff==0)
        return;

    mFps =  ((mFrameCount - mLastFrameCount) * float(s2ns(1))) / diff;
    mLastFpsTime = now;
    mLastFrameCount = mFrameCount;
    LOGI("### Camera FPS ### [%d] Frames, %.2f FPS", mFrameCount, mFps);
}

class CameraHandler: public CameraListener
{
protected:
    int cameraId;
    sp<Camera> camera;
#if defined(ANDROID_r3_0_1) || defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
    sp<SurfaceTexture> surface;
#endif
#if defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0) || defined(ANDROID_r4_3_0) || defined(ANDROID_r4_4_0)
    sp<BufferQueue> queue;
    sp<ConsumerListenerStub> listener;
#endif
    CameraParameters* params;
    CameraCallback cameraCallback;
    void* userData;

    int emptyCameraCallbackReported;

    int width;
    int height;

    static const char* flashModesNames[ANDROID_CAMERA_FLASH_MODES_NUM];
    static const char* focusModesNames[ANDROID_CAMERA_FOCUS_MODES_NUM];
    static const char* whiteBalanceModesNames[ANDROID_CAMERA_WHITE_BALANCE_MODES_NUM];
    static const char* antibandingModesNames[ANDROID_CAMERA_ANTIBANDING_MODES_NUM];

    void doCall(void* buffer, size_t bufferSize)
    {
        if (cameraCallback == 0)
        {
            if (!emptyCameraCallbackReported)
                LOGE("CameraHandler::doCall(void*, size_t): Camera callback is empty!");

            emptyCameraCallbackReported++;
        }
        else
        {
            bool res = (*cameraCallback)(buffer, bufferSize, userData);

            if(!res)
            {
                LOGE("CameraHandler::doCall(void*, size_t): cameraCallback returns false (camera connection will be closed)");
                closeCameraConnect();
            }
        }
    }

    void doCall(const sp<IMemory>& dataPtr)
    {
        if (dataPtr == NULL)
        {
            LOGE("CameraHandler::doCall(const sp<IMemory>&): dataPtr==NULL (no frame to handle)");
            return;
        }

        size_t size = dataPtr->size();
        if (size <= 0)
        {
            LOGE("CameraHandler::doCall(const sp<IMemory>&): IMemory object is of zero size");
            return;
        }

        void* buffer = (void *)dataPtr->pointer();
        if (!buffer)
        {
            LOGE("CameraHandler::doCall(const sp<IMemory>&): Buffer pointer is NULL");
            return;
        }

        doCall(buffer, size);
    }

    virtual void postDataTimestamp(nsecs_t timestamp, int32_t msgType, const sp<IMemory>& dataPtr)
    {
        static uint32_t count = 0;
        count++;

        LOGE("Recording cb: %d %lld %%p Offset:%%d Stride:%%d\n", msgType, timestamp);

        if (dataPtr == NULL)
        {
            LOGE("postDataTimestamp: dataPtr IS ZERO -- returning");
            camera->releaseRecordingFrame(dataPtr);
            LOGE("postDataTimestamp:  camera->releaseRecordingFrame(dataPtr) is done");
            return;
        }

        uint8_t *ptr = (uint8_t*) dataPtr->pointer();
        if (ptr)
            LOGE("VID_CB: 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x 0x%x", ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], ptr[5], ptr[6], ptr[7], ptr[8], ptr[9]);
        else
            LOGE("postDataTimestamp: Ptr is zero");

        camera->releaseRecordingFrame(dataPtr);
    }

    // Split list of floats, returns number of floats found
    static int split_float(const char *str, float* out, char delim, int max_elem_num,
                           char **endptr = NULL)
    {
        // Find the first float.
        char *end = const_cast<char*>(str);
        int elem_num = 0;
        for(; elem_num < max_elem_num; elem_num++ ){
            char* curr_end;
            out[elem_num] = (float)strtof(end, &curr_end);
            // No other numbers found, finish the loop
            if(end == curr_end){
                break;
            }
            if (*curr_end != delim) {
                // When end of string, finish the loop
                if (*curr_end == 0){
                    elem_num++;
                    break;
                }
                else {
                    LOGE("Cannot find delimeter (%c) in str=%s", delim, str);
                    return -1;
                }
            }
            // Skip the delimiter character
            end = curr_end + 1;
        }
        if (endptr)
            *endptr = end;
        return elem_num;
    }

    int is_supported(const char* supp_modes_key, const char* mode)
    {
        const char* supported_modes = params->get(supp_modes_key);
        return (supported_modes && mode && (strstr(supported_modes, mode) > 0));
    }

    float getFocusDistance(int focus_distance_type)
    {
#if !defined(ANDROID_r2_2_0)
        if (focus_distance_type >= 0 && focus_distance_type < 3)
    {
            float focus_distances[3];
            const char* output = params->get(CameraParameters::KEY_FOCUS_DISTANCES);
            int val_num = CameraHandler::split_float(output, focus_distances, ',', 3);
            if(val_num == 3)
        {
                return focus_distances[focus_distance_type];
            }
            else
        {
                LOGE("Invalid focus distances.");
            }
        }
#endif
    return -1;
    }

    static int getModeNum(const char** modes, const int modes_num, const char* mode_name)
    {
        for (int i = 0; i < modes_num; i++){
            if(!strcmp(modes[i],mode_name))
                return i;
        }
        return -1;
    }

public:
    CameraHandler(CameraCallback callback = 0, void* _userData = 0):
        cameraId(0),
        cameraCallback(callback),
        userData(_userData),
        emptyCameraCallbackReported(0)
    {
        LOGD("Instantiated new CameraHandler (%p, %p)", callback, _userData);
        void* params_buffer = operator new(sizeof(CameraParameters) + MAGIC_TAIL);
        params = new(params_buffer) CameraParameters();
    }

    virtual ~CameraHandler()
    {
        if (params)
            params->~CameraParameters();
            operator delete(params);
        LOGD("CameraHandler destructor is called");
    }

    virtual void notify(int32_t msgType, int32_t ext1, int32_t ext2)
    {
        LOGE("CameraHandler::Notify: msgType=%d ext1=%d ext2=%d\n", msgType, ext1, ext2);
#if 0
        if ( msgType & CAMERA_MSG_FOCUS )
            LOGE("CameraHandler::Notify  AutoFocus %s in %llu us\n", (ext1) ? "OK" : "FAIL", timevalDelay(&autofocus_start));

        if ( msgType & CAMERA_MSG_SHUTTER )
            LOGE("CameraHandler::Notify  Shutter done in %llu us\n", timeval_delay(&picture_start));
#endif
    }

    virtual void postData(int32_t msgType, const sp<IMemory>& dataPtr
#if defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3) || defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0) \
 || defined(ANDROID_r4_3_0) || defined(ANDROID_r4_4_0)
                          ,camera_frame_metadata_t*
#endif
                          )
    {
        debugShowFPS();

        if ( msgType & CAMERA_MSG_PREVIEW_FRAME )
        {
            doCall(dataPtr);
            return;
        }

        //if (msgType != CAMERA_MSG_PREVIEW_FRAME)
            //LOGE("CameraHandler::postData  Recieved message %d is not equal to CAMERA_MSG_PREVIEW_FRAME (%d)", (int) msgType, CAMERA_MSG_PREVIEW_FRAME);

        if ( msgType & CAMERA_MSG_RAW_IMAGE )
            LOGE("CameraHandler::postData  Unexpected data format: RAW\n");

        if (msgType & CAMERA_MSG_POSTVIEW_FRAME)
            LOGE("CameraHandler::postData  Unexpected data format: Postview frame\n");

        if (msgType & CAMERA_MSG_COMPRESSED_IMAGE )
            LOGE("CameraHandler::postData  Unexpected data format: JPEG");
    }

    static CameraHandler* initCameraConnect(const CameraCallback& callback, int cameraId, void* userData, CameraParameters* prevCameraParameters);
    void closeCameraConnect();
    double getProperty(int propIdx);
    void setProperty(int propIdx, double value);
    static void applyProperties(CameraHandler** ppcameraHandler);

    std::string cameraPropertySupportedPreviewSizesString;
    std::string cameraPropertyPreviewFormatString;
};

const char* CameraHandler::flashModesNames[ANDROID_CAMERA_FLASH_MODES_NUM] =
{
    CameraParameters::FLASH_MODE_AUTO,
    CameraParameters::FLASH_MODE_OFF,
    CameraParameters::FLASH_MODE_ON,
    CameraParameters::FLASH_MODE_RED_EYE,
    CameraParameters::FLASH_MODE_TORCH
};

const char* CameraHandler::focusModesNames[ANDROID_CAMERA_FOCUS_MODES_NUM] =
{
    CameraParameters::FOCUS_MODE_AUTO,
#if !defined(ANDROID_r2_2_0)
    CameraParameters::FOCUS_MODE_CONTINUOUS_VIDEO,
#else
    CameraParameters::FOCUS_MODE_AUTO,
#endif
    CameraParameters::FOCUS_MODE_EDOF,
    CameraParameters::FOCUS_MODE_FIXED,
    CameraParameters::FOCUS_MODE_INFINITY,
    CameraParameters::FOCUS_MODE_MACRO,
#if !defined(ANDROID_r2_2_0) && !defined(ANDROID_r2_3_3) && !defined(ANDROID_r3_0_1)
    CameraParameters::FOCUS_MODE_CONTINUOUS_PICTURE
#else
    CameraParameters::FOCUS_MODE_AUTO
#endif
};

const char* CameraHandler::whiteBalanceModesNames[ANDROID_CAMERA_WHITE_BALANCE_MODES_NUM] =
{
    CameraParameters::WHITE_BALANCE_AUTO,
    CameraParameters::WHITE_BALANCE_CLOUDY_DAYLIGHT,
    CameraParameters::WHITE_BALANCE_DAYLIGHT,
    CameraParameters::WHITE_BALANCE_FLUORESCENT,
    CameraParameters::WHITE_BALANCE_INCANDESCENT,
    CameraParameters::WHITE_BALANCE_SHADE,
    CameraParameters::WHITE_BALANCE_TWILIGHT
};

const char* CameraHandler::antibandingModesNames[ANDROID_CAMERA_ANTIBANDING_MODES_NUM] =
{
    CameraParameters::ANTIBANDING_50HZ,
    CameraParameters::ANTIBANDING_60HZ,
    CameraParameters::ANTIBANDING_AUTO
};


CameraHandler* CameraHandler::initCameraConnect(const CameraCallback& callback, int cameraId, void* userData, CameraParameters* prevCameraParameters)
{

    typedef sp<Camera> (*Android22ConnectFuncType)();
    typedef sp<Camera> (*Android23ConnectFuncType)(int);
    typedef sp<Camera> (*Android3DConnectFuncType)(int, int);
    typedef sp<Camera> (*Android43ConnectFuncType)(int, const String16&, int);

    const int ANY_CAMERA_INDEX = -1;
    const int BACK_CAMERA_INDEX = 99;
    const int FRONT_CAMERA_INDEX = 98;

    enum {
    CAMERA_SUPPORT_MODE_2D = 0x01, /* Camera Sensor supports 2D mode. */
    CAMERA_SUPPORT_MODE_3D = 0x02, /* Camera Sensor supports 3D mode. */
    CAMERA_SUPPORT_MODE_NONZSL = 0x04, /* Camera Sensor in NON-ZSL mode. */
    CAMERA_SUPPORT_MODE_ZSL = 0x08 /* Camera Sensor supports ZSL mode. */
    };

    // used for Android 4.3
    enum {
        USE_CALLING_UID = -1
    };

    const char Android22ConnectName[] = "_ZN7android6Camera7connectEv";
    const char Android23ConnectName[] = "_ZN7android6Camera7connectEi";
    const char Android3DConnectName[] = "_ZN7android6Camera7connectEii";
    const char Android43ConnectName[] = "_ZN7android6Camera7connectEiRKNS_8String16Ei";

    int localCameraIndex = cameraId;

    if (cameraId == ANY_CAMERA_INDEX)
    {
        localCameraIndex = 0;
    }
#if !defined(ANDROID_r2_2_0)
    else if (cameraId == BACK_CAMERA_INDEX)
    {
        LOGD("Back camera selected");
        for (int i = 0; i < Camera::getNumberOfCameras(); i++)
        {
            CameraInfo info;
            Camera::getCameraInfo(i, &info);
            if (info.facing == CAMERA_FACING_BACK)
            {
                localCameraIndex = i;
                break;
            }
        }
    }
    else if (cameraId == FRONT_CAMERA_INDEX)
    {
        LOGD("Front camera selected");
        for (int i = 0; i < Camera::getNumberOfCameras(); i++)
        {
            CameraInfo info;
            Camera::getCameraInfo(i, &info);
            if (info.facing == CAMERA_FACING_FRONT)
            {
                localCameraIndex = i;
                break;
            }
        }
    }

    if (localCameraIndex == BACK_CAMERA_INDEX)
    {
        LOGE("Back camera not found!");
        return NULL;
    }
    else if (localCameraIndex == FRONT_CAMERA_INDEX)
    {
        LOGE("Front camera not found!");
        return NULL;
    }
#endif

    LOGD("CameraHandler::initCameraConnect(%p, %d, %p, %p)", callback, localCameraIndex, userData, prevCameraParameters);

    sp<Camera> camera = 0;

    void* CameraHALHandle = dlopen("libcamera_client.so", RTLD_LAZY);

    if (!CameraHALHandle)
    {
        LOGE("Cannot link to \"libcamera_client.so\"");
        return NULL;
    }

    // reset errors
    dlerror();

    if (Android22ConnectFuncType Android22Connect = (Android22ConnectFuncType)dlsym(CameraHALHandle, Android22ConnectName))
    {
        LOGD("Connecting to CameraService v 2.2");
        camera = Android22Connect();
    }
    else if (Android23ConnectFuncType Android23Connect = (Android23ConnectFuncType)dlsym(CameraHALHandle, Android23ConnectName))
    {
        LOGD("Connecting to CameraService v 2.3");
        camera = Android23Connect(localCameraIndex);
    }
    else if (Android3DConnectFuncType Android3DConnect = (Android3DConnectFuncType)dlsym(CameraHALHandle, Android3DConnectName))
    {
        LOGD("Connecting to CameraService v 3D");
        camera = Android3DConnect(localCameraIndex, CAMERA_SUPPORT_MODE_2D);
    }
    else if (Android43ConnectFuncType Android43Connect = (Android43ConnectFuncType)dlsym(CameraHALHandle, Android43ConnectName))
    {
        std::string currentProcName = getProcessName();
        LOGD("Current process name for camera init: %s", currentProcName.c_str());
        camera = Android43Connect(localCameraIndex, String16(currentProcName.c_str()), USE_CALLING_UID);
    }
    else
    {
        dlclose(CameraHALHandle);
        LOGE("Cannot connect to CameraService. Connect method was not found!");
        return NULL;
    }

    dlclose(CameraHALHandle);

    if ( 0 == camera.get() )
    {
        LOGE("initCameraConnect: Unable to connect to CameraService\n");
        return 0;
    }

    CameraHandler* handler = new CameraHandler(callback, userData);
    camera->setListener(handler);

    handler->camera = camera;
    handler->cameraId = localCameraIndex;

    if (prevCameraParameters != NULL)
    {
        LOGI("initCameraConnect: Setting paramers from previous camera handler");
        camera->setParameters(prevCameraParameters->flatten());
        handler->params->unflatten(prevCameraParameters->flatten());
    }
    else
    {
        android::String8 params_str = camera->getParameters();
        LOGI("initCameraConnect: [%s]", params_str.string());

        handler->params->unflatten(params_str);

        LOGD("Supported Cameras: %s", handler->params->get("camera-indexes"));
        LOGD("Supported Picture Sizes: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_PICTURE_SIZES));
        LOGD("Supported Picture Formats: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_PICTURE_FORMATS));
        LOGD("Supported Preview Sizes: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_PREVIEW_SIZES));
        LOGD("Supported Preview Formats: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_PREVIEW_FORMATS));
        LOGD("Supported Preview Frame Rates: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_PREVIEW_FRAME_RATES));
        LOGD("Supported Thumbnail Sizes: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_JPEG_THUMBNAIL_SIZES));
        LOGD("Supported Whitebalance Modes: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_WHITE_BALANCE));
        LOGD("Supported Effects: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_EFFECTS));
        LOGD("Supported Scene Modes: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_SCENE_MODES));
        LOGD("Supported Focus Modes: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_FOCUS_MODES));
        LOGD("Supported Antibanding Options: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_ANTIBANDING));
        LOGD("Supported Flash Modes: %s", handler->params->get(CameraParameters::KEY_SUPPORTED_FLASH_MODES));

#if !defined(ANDROID_r2_2_0)
        // Set focus mode to continuous-video if supported
        const char* available_focus_modes = handler->params->get(CameraParameters::KEY_SUPPORTED_FOCUS_MODES);
        if (available_focus_modes != 0)
        {
            if (strstr(available_focus_modes, "continuous-video") != NULL)
            {
                handler->params->set(CameraParameters::KEY_FOCUS_MODE, CameraParameters::FOCUS_MODE_CONTINUOUS_VIDEO);

                status_t resParams = handler->camera->setParameters(handler->params->flatten());

                if (resParams != 0)
                {
                    LOGE("initCameraConnect: failed to set autofocus mode to \"continuous-video\"");
                }
                else
                {
                    LOGD("initCameraConnect: autofocus is set to mode \"continuous-video\"");
                }
            }
        }
#endif

        //check if yuv420sp format available. Set this format as preview format.
        const char* available_formats = handler->params->get(CameraParameters::KEY_SUPPORTED_PREVIEW_FORMATS);
        if (available_formats != 0)
        {
            const char* format_to_set = 0;
            const char* pos = available_formats;
            const char* ptr = pos;
            while(true)
            {
                while(*ptr != 0 && *ptr != ',') ++ptr;
                if (ptr != pos)
                {
                    if (0 == strncmp(pos, "yuv420sp", ptr - pos))
                    {
                        format_to_set = "yuv420sp";
                        break;
                    }
                    if (0 == strncmp(pos, "yvu420sp", ptr - pos))
                        format_to_set = "yvu420sp";
                }
                if (*ptr == 0)
                    break;
                pos = ++ptr;
            }

            if (0 != format_to_set)
            {
                handler->params->setPreviewFormat(format_to_set);

                status_t resParams = handler->camera->setParameters(handler->params->flatten());

                if (resParams != 0)
                    LOGE("initCameraConnect: failed to set preview format to %s", format_to_set);
                else
                    LOGD("initCameraConnect: preview format is set to %s", format_to_set);
            }
        }

        handler->params->setPreviewSize(640, 480);
        status_t resParams = handler->camera->setParameters(handler->params->flatten());
        if (resParams != 0)
            LOGE("initCameraConnect: failed to set preview resolution to 640x480");
        else
            LOGD("initCameraConnect: preview format is set to 640x480");
    }

    status_t bufferStatus;
#if defined(ANDROID_r2_2_0)
    bufferStatus = camera->setPreviewDisplay(sp<ISurface>(0 /*new DummySurface*/));
    if (bufferStatus != 0)
        LOGE("initCameraConnect: failed setPreviewDisplay(0) call (status %d); camera might not work correctly on some devices", bufferStatus);
#elif defined(ANDROID_r2_3_3)
    /* Do nothing in case of 2.3 for now */
#elif defined(ANDROID_r3_0_1) || defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
    void* surface_texture_obj = operator new(sizeof(SurfaceTexture) + MAGIC_TAIL);
    handler->surface = new(surface_texture_obj) SurfaceTexture(MAGIC_OPENCV_TEXTURE_ID);
    bufferStatus = camera->setPreviewTexture(handler->surface);
    if (bufferStatus != 0)
        LOGE("initCameraConnect: failed setPreviewTexture call (status %d); camera might not work correctly", bufferStatus);
#elif defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0) || defined(ANDROID_r4_3_0)
    void* buffer_queue_obj = operator new(sizeof(BufferQueue) + MAGIC_TAIL);
    handler->queue = new(buffer_queue_obj) BufferQueue();
    void* consumer_listener_obj = operator new(sizeof(ConsumerListenerStub) + MAGIC_TAIL);
    handler->listener = new(consumer_listener_obj) ConsumerListenerStub();
    handler->queue->consumerConnect(handler->listener);
    bufferStatus = camera->setPreviewTexture(handler->queue);
    if (bufferStatus != 0)
        LOGE("initCameraConnect: failed setPreviewTexture call; camera might not work correctly");
# elif defined(ANDROID_r4_4_0)
    void* buffer_queue_obj = operator new(sizeof(BufferQueue) + MAGIC_TAIL);
    handler->queue = new(buffer_queue_obj) BufferQueue();
    handler->queue->setConsumerUsageBits(GraphicBuffer::USAGE_HW_TEXTURE);
    void* consumer_listener_obj = operator new(sizeof(ConsumerListenerStub) + MAGIC_TAIL);
    handler->listener = new(consumer_listener_obj) ConsumerListenerStub();
    handler->queue->consumerConnect(handler->listener, true);
    bufferStatus = handler->camera->setPreviewTarget(handler->queue);
    if (bufferStatus != 0)
        LOGE("applyProperties: failed setPreviewTexture call; camera might not work correctly");
# endif

#if (defined(ANDROID_r2_2_0) || defined(ANDROID_r2_3_3) || defined(ANDROID_r3_0_1))
# if 1
    ////ATTENTION: switching between two versions: with and without copying memory inside Android OS
    //// see the method  CameraService::Client::copyFrameAndPostCopiedFrame and where it is used
    camera->setPreviewCallbackFlags( FRAME_CALLBACK_FLAG_ENABLE_MASK | FRAME_CALLBACK_FLAG_COPY_OUT_MASK);//with copy
# else
    camera->setPreviewCallbackFlags( FRAME_CALLBACK_FLAG_ENABLE_MASK );//without copy
# endif
#else
    camera->setPreviewCallbackFlags( CAMERA_FRAME_CALLBACK_FLAG_ENABLE_MASK | CAMERA_FRAME_CALLBACK_FLAG_COPY_OUT_MASK);//with copy
#endif //!(defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3))

    LOGD("Starting preview");
    status_t previewStatus = camera->startPreview();

    if (previewStatus != 0)
    {
        LOGE("initCameraConnect: startPreview() fails. Closing camera connection...");
        handler->closeCameraConnect();
        handler = 0;
    }
    else
    {
        LOGD("Preview started successfully");
    }

    return handler;
}

void CameraHandler::closeCameraConnect()
{
    if (camera == NULL)
    {
        LOGI("... camera is already NULL");
        return;
    }

    camera->stopPreview();
#if defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3) || defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0) \
 || defined(ANDROID_r4_3_0) || defined(ANDROID_r4_4_0)
    camera->setPreviewCallbackFlags(CAMERA_FRAME_CALLBACK_FLAG_NOOP);
#endif
    camera->disconnect();
    camera.clear();
    camera=NULL;
    // ATTENTION!!!!!!!!!!!!!!!!!!!!!!!!!!
    // When we set
    //    camera=NULL
    // above, the pointed instance of android::Camera object is destructed,
    // since this member `camera' has type android::sp<Camera> (android smart pointer template class),
    // and this is the only pointer to it.
    //
    // BUT this instance of CameraHandler is set as a listener for that android::Camera object
    // (see the function CameraHandler::initCameraConnect above),
    // so this instance of CameraHandler is pointed from that android::Camera object as
    //     sp<CameraListener>  mListener
    // and there is no other android smart pointers to this.
    //
    // It means, when that instance of the android::Camera object is destructed,
    // it calls destructor for this CameraHandler instance too.
    //
    // So, this line `camera=NULL' causes to the call `delete this'
    // (see destructor of the template class android::sp)
    //
    // So, we must not call `delete this' after the line, since it just has been called indeed
}

double CameraHandler::getProperty(int propIdx)
{
    LOGD("CameraHandler::getProperty(%d)", propIdx);

    switch (propIdx)
    {
    case ANDROID_CAMERA_PROPERTY_FRAMEWIDTH:
    {
        int w,h;
        params->getPreviewSize(&w, &h);
        return w;
    }
    case ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT:
    {
        int w,h;
        params->getPreviewSize(&w, &h);
        return h;
    }
    case ANDROID_CAMERA_PROPERTY_SUPPORTED_PREVIEW_SIZES_STRING:
    {
        cameraPropertySupportedPreviewSizesString = params->get(CameraParameters::KEY_SUPPORTED_PREVIEW_SIZES);
        union {const char* str;double res;} u;
        memset(&u.res, 0, sizeof(u.res));
        u.str = cameraPropertySupportedPreviewSizesString.c_str();
        return u.res;
    }
    case ANDROID_CAMERA_PROPERTY_PREVIEW_FORMAT_STRING:
    {
        const char* fmt = params->get(CameraParameters::KEY_PREVIEW_FORMAT);
        if (fmt == CameraParameters::PIXEL_FORMAT_YUV422SP)
            fmt = "yuv422sp";
        else if (fmt == CameraParameters::PIXEL_FORMAT_YUV420SP)
            fmt = "yuv420sp";
        else if (fmt == CameraParameters::PIXEL_FORMAT_YUV422I)
            fmt = "yuv422i";
        else if (fmt == CameraParameters::PIXEL_FORMAT_RGB565)
            fmt = "rgb565";
        else if (fmt == CameraParameters::PIXEL_FORMAT_JPEG)
            fmt = "jpeg";
        cameraPropertyPreviewFormatString = fmt;

        union {const char* str;double res;} u;
        memset(&u.res, 0, sizeof(u.res));
        u.str = cameraPropertyPreviewFormatString.c_str();
        return u.res;
    }
    case ANDROID_CAMERA_PROPERTY_EXPOSURE:
    {
        int exposure = params->getInt(CameraParameters::KEY_EXPOSURE_COMPENSATION);
        return exposure;
    }
    case ANDROID_CAMERA_PROPERTY_FPS:
    {
        return params->getPreviewFrameRate();
    }
    case ANDROID_CAMERA_PROPERTY_FLASH_MODE:
    {
        int flash_mode = getModeNum(CameraHandler::flashModesNames,
                                    ANDROID_CAMERA_FLASH_MODES_NUM,
                                    params->get(CameraParameters::KEY_FLASH_MODE));
        return flash_mode;
    }
    case ANDROID_CAMERA_PROPERTY_FOCUS_MODE:
    {
        int focus_mode = getModeNum(CameraHandler::focusModesNames,
                                    ANDROID_CAMERA_FOCUS_MODES_NUM,
                                    params->get(CameraParameters::KEY_FOCUS_MODE));
        return focus_mode;
    }
    case ANDROID_CAMERA_PROPERTY_WHITE_BALANCE:
    {
        int white_balance = getModeNum(CameraHandler::whiteBalanceModesNames,
                                       ANDROID_CAMERA_WHITE_BALANCE_MODES_NUM,
                                       params->get(CameraParameters::KEY_WHITE_BALANCE));
        return white_balance;
    }
    case ANDROID_CAMERA_PROPERTY_ANTIBANDING:
    {
        int antibanding = getModeNum(CameraHandler::antibandingModesNames,
                                     ANDROID_CAMERA_ANTIBANDING_MODES_NUM,
                                     params->get(CameraParameters::KEY_ANTIBANDING));
        return antibanding;
    }
    case ANDROID_CAMERA_PROPERTY_FOCAL_LENGTH:
    {
        float focal_length = params->getFloat(CameraParameters::KEY_FOCAL_LENGTH);
        return focal_length;
    }
    case ANDROID_CAMERA_PROPERTY_FOCUS_DISTANCE_NEAR:
    {
        return getFocusDistance(ANDROID_CAMERA_FOCUS_DISTANCE_NEAR_INDEX);
    }
    case ANDROID_CAMERA_PROPERTY_FOCUS_DISTANCE_OPTIMAL:
    {
        return getFocusDistance(ANDROID_CAMERA_FOCUS_DISTANCE_OPTIMAL_INDEX);
    }
    case ANDROID_CAMERA_PROPERTY_FOCUS_DISTANCE_FAR:
    {
        return getFocusDistance(ANDROID_CAMERA_FOCUS_DISTANCE_FAR_INDEX);
    }
#if !defined(ANDROID_r2_2_0) && !defined(ANDROID_r2_3_3) && !defined(ANDROID_r3_0_1)
    case ANDROID_CAMERA_PROPERTY_WHITEBALANCE_LOCK:
    {
        const char* status = params->get(CameraParameters::KEY_AUTO_WHITEBALANCE_LOCK);
        if (status == CameraParameters::TRUE)
            return 1.;
        else
            return 0.;
    }
    case ANDROID_CAMERA_PROPERTY_EXPOSE_LOCK:
    {
        const char* status = params->get(CameraParameters::KEY_AUTO_EXPOSURE_LOCK);
        if (status == CameraParameters::TRUE)
            return 1.;
        else
            return 0.;
    }
#endif
    default:
        LOGW("CameraHandler::getProperty - Unsupported property.");
    };
    return -1;
}

void CameraHandler::setProperty(int propIdx, double value)
{
    LOGD("CameraHandler::setProperty(%d, %f)", propIdx, value);

    android::String8 params_str;
    params_str = camera->getParameters();
    LOGI("Params before set: [%s]", params_str.string());

    switch (propIdx)
    {
    case ANDROID_CAMERA_PROPERTY_FRAMEWIDTH:
    {
        int w,h;
        params->getPreviewSize(&w, &h);
        width = (int)value;
    }
    break;
    case ANDROID_CAMERA_PROPERTY_FRAMEHEIGHT:
    {
        int w,h;
        params->getPreviewSize(&w, &h);
        height = (int)value;
    }
    break;
    case ANDROID_CAMERA_PROPERTY_EXPOSURE:
    {
        int max_exposure = params->getInt("max-exposure-compensation");
        int min_exposure = params->getInt("min-exposure-compensation");
        if(max_exposure && min_exposure)
        {
            int exposure = (int)value;
            if(exposure >= min_exposure && exposure <= max_exposure)
                params->set("exposure-compensation", exposure);
            else
                LOGE("Exposure compensation not in valid range (%i,%i).", min_exposure, max_exposure);
        } else
            LOGE("Exposure compensation adjust is not supported.");

        camera->setParameters(params->flatten());
    }
    break;
    case ANDROID_CAMERA_PROPERTY_FLASH_MODE:
    {
        int new_val = (int)value;
        if(new_val >= 0 && new_val < ANDROID_CAMERA_FLASH_MODES_NUM)
        {
            const char* mode_name = flashModesNames[new_val];
            if(is_supported(CameraParameters::KEY_SUPPORTED_FLASH_MODES, mode_name))
                params->set(CameraParameters::KEY_FLASH_MODE, mode_name);
            else
                LOGE("Flash mode %s is not supported.", mode_name);
        }
        else
            LOGE("Flash mode value not in valid range.");

        camera->setParameters(params->flatten());
    }
    break;
    case ANDROID_CAMERA_PROPERTY_FOCUS_MODE:
    {
        int new_val = (int)value;
        if(new_val >= 0 && new_val < ANDROID_CAMERA_FOCUS_MODES_NUM)
        {
            const char* mode_name = focusModesNames[new_val];
            if(is_supported(CameraParameters::KEY_SUPPORTED_FOCUS_MODES, mode_name))
                params->set(CameraParameters::KEY_FOCUS_MODE, mode_name);
            else
                LOGE("Focus mode %s is not supported.", mode_name);
        }
        else
            LOGE("Focus mode value not in valid range.");

        camera->setParameters(params->flatten());
    }
    break;
    case ANDROID_CAMERA_PROPERTY_WHITE_BALANCE:
    {
        int new_val = (int)value;
        if(new_val >= 0 && new_val < ANDROID_CAMERA_WHITE_BALANCE_MODES_NUM)
        {
            const char* mode_name = whiteBalanceModesNames[new_val];
            if(is_supported(CameraParameters::KEY_SUPPORTED_WHITE_BALANCE, mode_name))
                params->set(CameraParameters::KEY_WHITE_BALANCE, mode_name);
            else
                LOGE("White balance mode %s is not supported.", mode_name);
        }
        else
            LOGE("White balance mode value not in valid range.");

        camera->setParameters(params->flatten());
    }
    break;
    case ANDROID_CAMERA_PROPERTY_ANTIBANDING:
    {
        int new_val = (int)value;
        if(new_val >= 0 && new_val < ANDROID_CAMERA_ANTIBANDING_MODES_NUM)
        {
            const char* mode_name = antibandingModesNames[new_val];
            if(is_supported(CameraParameters::KEY_SUPPORTED_ANTIBANDING, mode_name))
                params->set(CameraParameters::KEY_ANTIBANDING, mode_name);
            else
                LOGE("Antibanding mode %s is not supported.", mode_name);
        }
        else
            LOGE("Antibanding mode value not in valid range.");

        camera->setParameters(params->flatten());
    }
    break;
#if !defined(ANDROID_r2_2_0) && !defined(ANDROID_r2_3_3) && !defined(ANDROID_r3_0_1)
    case ANDROID_CAMERA_PROPERTY_EXPOSE_LOCK:
    {
        if (is_supported(CameraParameters::KEY_AUTO_EXPOSURE_LOCK_SUPPORTED, "true"))
        {
            if (value != 0)
                params->set(CameraParameters::KEY_AUTO_EXPOSURE_LOCK, CameraParameters::TRUE);
            else
                params->set(CameraParameters::KEY_AUTO_EXPOSURE_LOCK, CameraParameters::FALSE);
            LOGE("Expose lock is set");
        }
        else
            LOGE("Expose lock is not supported");

        camera->setParameters(params->flatten());
    }
    break;
    case ANDROID_CAMERA_PROPERTY_WHITEBALANCE_LOCK:
    {
        if (is_supported(CameraParameters::KEY_AUTO_WHITEBALANCE_LOCK_SUPPORTED, "true"))
        {
            if (value != 0)
                params->set(CameraParameters::KEY_AUTO_WHITEBALANCE_LOCK, CameraParameters::TRUE);
            else
                params->set(CameraParameters::KEY_AUTO_WHITEBALANCE_LOCK, CameraParameters::FALSE);
            LOGE("White balance lock is set");
        }
        else
            LOGE("White balance lock is not supported");

        camera->setParameters(params->flatten());
    }
    break;
#endif
    default:
        LOGW("CameraHandler::setProperty - Unsupported property.");
    };

    params_str = camera->getParameters();
    LOGI("Params after set: [%s]", params_str.string());
}

void CameraHandler::applyProperties(CameraHandler** ppcameraHandler)
{
    LOGD("CameraHandler::applyProperties()");

    if (ppcameraHandler == 0)
    {
        LOGE("applyProperties: Passed NULL ppcameraHandler");
        return;
    }

    if (*ppcameraHandler == 0)
    {
        LOGE("applyProperties: Passed NULL *ppcameraHandler");
        return;
    }

    // delayed resolution setup to exclude errors during other parameres setup on the fly
    // without camera restart
    if (((*ppcameraHandler)->width != 0) && ((*ppcameraHandler)->height != 0))
        (*ppcameraHandler)->params->setPreviewSize((*ppcameraHandler)->width, (*ppcameraHandler)->height);

#if defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3) || defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0) \
 || defined(ANDROID_r4_3_0) || defined(ANDROID_r4_4_0)
    CameraHandler* handler=*ppcameraHandler;

    handler->camera->stopPreview();
    handler->camera->setPreviewCallbackFlags(CAMERA_FRAME_CALLBACK_FLAG_NOOP);

    status_t reconnectStatus = handler->camera->reconnect();
    if (reconnectStatus != 0)
    {
        LOGE("applyProperties: failed to reconnect camera (status %d)", reconnectStatus);
        return;
    }

    handler->camera->setParameters((*ppcameraHandler)->params->flatten());

    status_t bufferStatus;
# if defined(ANDROID_r4_0_0) || defined(ANDROID_r4_0_3)
    void* surface_texture_obj = operator new(sizeof(SurfaceTexture) + MAGIC_TAIL);
    handler->surface = new(surface_texture_obj) SurfaceTexture(MAGIC_OPENCV_TEXTURE_ID);
    bufferStatus = handler->camera->setPreviewTexture(handler->surface);
    if (bufferStatus != 0)
        LOGE("applyProperties: failed setPreviewTexture call (status %d); camera might not work correctly", bufferStatus);
# elif defined(ANDROID_r4_1_1) || defined(ANDROID_r4_2_0) || defined(ANDROID_r4_3_0)
    void* buffer_queue_obj = operator new(sizeof(BufferQueue) + MAGIC_TAIL);
    handler->queue = new(buffer_queue_obj) BufferQueue();
    handler->queue->consumerConnect(handler->listener);
    bufferStatus = handler->camera->setPreviewTexture(handler->queue);
    if (bufferStatus != 0)
        LOGE("applyProperties: failed setPreviewTexture call; camera might not work correctly");
# elif defined(ANDROID_r4_4_0)
    void* buffer_queue_obj = operator new(sizeof(BufferQueue) + MAGIC_TAIL);
    handler->queue = new(buffer_queue_obj) BufferQueue();
    handler->queue->setConsumerUsageBits(GraphicBuffer::USAGE_HW_TEXTURE);
    handler->queue->consumerConnect(handler->listener, true);
    bufferStatus = handler->camera->setPreviewTarget(handler->queue);
    if (bufferStatus != 0)
        LOGE("applyProperties: failed setPreviewTexture call; camera might not work correctly");
# endif

    handler->camera->setPreviewCallbackFlags( CAMERA_FRAME_CALLBACK_FLAG_ENABLE_MASK | CAMERA_FRAME_CALLBACK_FLAG_COPY_OUT_MASK);//with copy

    LOGD("Starting preview");
    status_t previewStatus = handler->camera->startPreview();

    if (previewStatus != 0)
    {
        LOGE("initCameraConnect: startPreview() fails. Closing camera connection...");
        handler->closeCameraConnect();
        handler = NULL;
    }
    else
    {
        LOGD("Preview started successfully");
    }
#else
    CameraHandler* previousCameraHandler=*ppcameraHandler;
    CameraCallback cameraCallback=previousCameraHandler->cameraCallback;
    void* userData=previousCameraHandler->userData;
    int cameraId=previousCameraHandler->cameraId;

    LOGD("CameraHandler::applyProperties(): before previousCameraHandler->closeCameraConnect");
    previousCameraHandler->closeCameraConnect();
    LOGD("CameraHandler::applyProperties(): after previousCameraHandler->closeCameraConnect");

    LOGD("CameraHandler::applyProperties(): before initCameraConnect");
    CameraHandler* handler=initCameraConnect(cameraCallback, cameraId, userData, (*ppcameraHandler)->params);
    LOGD("CameraHandler::applyProperties(): after initCameraConnect, handler=0x%x", (int)handler);
    if (handler == NULL) {
        LOGE("ERROR in applyProperties --- cannot reinit camera");
        handler=initCameraConnect(cameraCallback, cameraId, userData, NULL);
        LOGD("CameraHandler::applyProperties(): repeate initCameraConnect after ERROR, handler=0x%x", (int)handler);
        if (handler == NULL) {
            LOGE("ERROR in applyProperties --- cannot reinit camera AGAIN --- cannot do anything else");
        }
    }
    (*ppcameraHandler)=handler;
#endif
}


extern "C" {

void* initCameraConnectC(void* callback, int cameraId, void* userData)
{
    return CameraHandler::initCameraConnect((CameraCallback)callback, cameraId, userData, NULL);
}

void closeCameraConnectC(void** camera)
{
    CameraHandler** cc = (CameraHandler**)camera;
    (*cc)->closeCameraConnect();
    *cc = 0;
}

double getCameraPropertyC(void* camera, int propIdx)
{
    return ((CameraHandler*)camera)->getProperty(propIdx);
}

void setCameraPropertyC(void* camera, int propIdx, double value)
{
    ((CameraHandler*)camera)->setProperty(propIdx,value);
}

void applyCameraPropertiesC(void** camera)
{
    CameraHandler::applyProperties((CameraHandler**)camera);
}

}
