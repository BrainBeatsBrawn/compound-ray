#include "OrthographicCamera.h"

cray::OrthographicCamera::OrthographicCamera (const std::string name) : cray::DataRecordCamera<OrthographicCameraData>(name) {}
cray::OrthographicCamera::~OrthographicCamera() {}

void cray::OrthographicCamera::setXYscale (float x, float y)
{
    specializedData.scale.x = x;
    specializedData.scale.y = y;
}
