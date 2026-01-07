#include "PanoramicCamera.h"

#ifdef DEBUG
#include <iostream>
#endif

PanoramicCamera::PanoramicCamera(const std::string name) : DataRecordCamera<PanoramicCameraData>(name)
{
    // Allocate the SBT record for the associated raygen program
    if constexpr (debug_cameras == true) { std::cout << "Creating 360 camera." << std::endl;}
    // set the start radius of the 360 camera
    setStartRadius(0.0f);
    if constexpr (debug_cameras == true) { std::cout << "My d_pointer is at: " << getRecordPtr() << std::endl; }
}
PanoramicCamera::~PanoramicCamera()
{
    if constexpr (debug_cameras == true) { std::cout << "Destroying 360 camera." << std::endl; }
}

void PanoramicCamera::setStartRadius(float d)
{
    sbtRecord.data.specializedData.startRadius = d;
}
