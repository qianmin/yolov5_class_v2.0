QT       += core gui
QT += concurrent
greaterThan(QT_MAJOR_VERSION, 4): QT += widgets
CONFIG += c++11

SOURCES += \
        main.cpp \
        yolo5.cpp

HEADERS += \
    yolo5.h

# Default rules for deployment.
qnx: target.path = /tmp/$${TARGET}/bin
else: unix:!android: target.path = /opt/$${TARGET}/bin
!isEmpty(target.path): INSTALLS += target

###opencv(MSVC opencv4.15.1)
INCLUDEPATH+= D:\software\opencv_full_gpu_451\build\install\include
LIBS += -LD:\software\opencv_full_gpu_451\build\install\x64\vc16\lib -lopencv_world451d
