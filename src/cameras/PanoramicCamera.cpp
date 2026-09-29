#include <iostream>
#include "PanoramicCamera.h"

cray::PanoramicCamera::PanoramicCamera (const std::string name) : cray::DataRecordCamera<PanoramicCameraData>(name)
{
    // Allocate the SBT record for the associated raygen program
    if constexpr (debug_cameras == true) { std::cout << "Creating Panoramic camera." << std::endl;}
    // set the start radius of the 360 camera
    this->setStartRadius (0.0f);
    if constexpr (debug_cameras == true) { std::cout << "My d_pointer is at: " << getRecordPtr() << std::endl; }
}

cray::PanoramicCamera::~PanoramicCamera()
{
    if constexpr (debug_cameras == true) { std::cout << "Destroying Panoramic camera." << std::endl; }
}

void cray::PanoramicCamera::setStartRadius (float d)
{
    sbtRecord.data.specializedData.startRadius = d;
}
