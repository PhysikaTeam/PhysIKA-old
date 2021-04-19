# ----------------------------------------------------------------------------
#  Detect 3rd-party GUI libraries
# ----------------------------------------------------------------------------

#--- Win32 UI ---
ocv_clear_vars(HAVE_WIN32UI)
if(WITH_WIN32UI)
  try_compile(HAVE_WIN32UI
    "${OpenCV_BINARY_DIR}"
    "${OpenCV_SOURCE_DIR}/cmake/checks/win32uitest.cpp"
    CMAKE_FLAGS "-DLINK_LIBRARIES:STRING=user32;gdi32")
endif()

# --- QT4 ---
ocv_clear_vars(HAVE_QT HAVE_QT5)
if(WITH_QT)
  if(NOT CMAKE_VERSION VERSION_LESS 2.8.3 AND NOT WITH_QT EQUAL 4)
    find_package(Qt5Core)
    find_package(Qt5Gui)
    find_package(Qt5Widgets)
    find_package(Qt5Test)
    find_package(Qt5Concurrent)
    if(Qt5Core_FOUND AND Qt5Gui_FOUND AND Qt5Widgets_FOUND AND Qt5Test_FOUND AND Qt5Concurrent_FOUND)
      set(HAVE_QT5 ON)
      set(HAVE_QT  ON)
      find_package(Qt5OpenGL)
      if(Qt5OpenGL_FOUND)
        set(QT_QTOPENGL_FOUND ON)
      endif()
    endif()
  endif()

  if(NOT HAVE_QT)
    find_package(Qt4 REQUIRED QtCore QtGui QtTest)
    if(QT4_FOUND)
      set(HAVE_QT TRUE)
    endif()
  endif()
endif()

# --- GTK ---
ocv_clear_vars(HAVE_GTK HAVE_GTHREAD HAVE_GTKGLEXT)
if(WITH_GTK AND NOT HAVE_QT)
  CHECK_MODULE(gtk+-2.0 HAVE_GTK)
  CHECK_MODULE(gthread-2.0 HAVE_GTHREAD)
  if(WITH_OPENGL)
    CHECK_MODULE(gtkglext-1.0 HAVE_GTKGLEXT)
  endif()
endif()

# --- OpenGl ---
ocv_clear_vars(HAVE_OPENGL HAVE_QT_OPENGL)
if(WITH_OPENGL)
  if(WITH_WIN32UI OR (HAVE_QT AND QT_QTOPENGL_FOUND) OR HAVE_GTKGLEXT)
    find_package (OpenGL QUIET)
    if(OPENGL_FOUND)
      set(HAVE_OPENGL TRUE)
      list(APPEND OPENCV_LINKER_LIBS ${OPENGL_LIBRARIES})
      if(QT_QTOPENGL_FOUND)
        set(HAVE_QT_OPENGL TRUE)
      else()
        ocv_include_directories(${OPENGL_INCLUDE_DIR})
      endif()
    endif()
  endif()
endif(WITH_OPENGL)

# --- Carbon & Cocoa ---
if(APPLE)
  if(WITH_CARBON)
    set(HAVE_CARBON YES)
  elseif(NOT IOS)
    set(HAVE_COCOA YES)
  endif()
endif()
