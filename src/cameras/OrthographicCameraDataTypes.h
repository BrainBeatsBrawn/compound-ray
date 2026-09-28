#pragma once

#include <vector_types.h>
#include "GenericCameraDataTypes.h"

namespace cray
{
    struct OrthographicCameraData
    {
        float2 scale;

        inline bool operator==(const OrthographicCameraData& other)
        { return (this->scale.x == other.scale.x && this->scale.y == other.scale.y); }
    };
}

// A typedef for a RaygenPosedContainer containing an OrthographicCameraData
typedef cray::RaygenPosedContainer<cray::OrthographicCameraData> OrthographicCameraPosedData;
