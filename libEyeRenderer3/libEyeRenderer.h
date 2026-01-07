#ifndef LIB_EYE_RENDERER_3_H
#define LIB_EYE_RENDERER_3_H
#include <cstddef>
#include <vector_types.h>
#include <vector> // Use std::vector/array to get data out
#include <array>
#include <string>

#include <sutil/Matrix.h>

// A simplified ommatidium object, to make it easier to
// transfer ommatidial information from external API users.
struct OmmatidiumPacket
{
    float posX,posY,posZ;
    float dirX,dirY,dirZ;
    float acceptanceAngle;
    float focalpointOffset;
};

// Non-C global functions

// Gets the current view of the camera as a vector of colour values.
void getCameraData (std::vector<std::array<float, 3>>& cameraData);

// Tell client applications what the name of the compound.eye file path was. For compound
// eyes. Returns empty for non-compound eyes.
std::string getEyeDataPath();

extern "C"
{
    // Allocation functions

    void multicamAlloc();
    void multicamDealloc();

    // Configuration

    // turns on/off the '[PyEye]' debug outputs
    void setVerbosity (bool v);
    // Loads a given gltf file
    void loadGlTFscene (const char* filepath, sutil::Matrix4x4 root_transform);
    // Stops the eyeRenderer in a slightly more elegant way
    void stop();
    // Sets the output buffer to be large enough for a w x h image
    void setRenderSize (int w, int h);
    // Actually renders the frame, returns the time it took to render the frame (in ms)
    double renderFrame();
    // Save a copy of the render frame (non-compound cameras only)
    void saveFrameAs (const char* ppmFilename);
    // Retrieves a pointer to the frame data
    unsigned char* getFramePointer();
    // Set if the non compound eye pipeline is required
    void setRequireNoncompoundPipeline (const bool require_ncp);

    // Camera control

    size_t getCameraCount();
    void nextCamera();
    void previousCamera();
    size_t getCurrentCameraIndex();
    const char* getCurrentCameraName();
    void gotoCamera (int index);
    bool gotoCameraByName (char* name);
    void setCameraPosition (float x, float y, float z);
    void getCameraPosition (float& x, float& y, float& z);
    void setCameraLocalSpace (float lxx, float lxy, float lxz,
                              float lyx, float lyy, float lyz,
                              float lzx, float lzy, float lzz);
    // Rotate current camera
    void rotateCameraAround (float angle, float axisX, float axisY, float axisZ);
    // Rotate current camera
    void rotateCameraLocallyAround (float angle, float axisX, float axisY, float axisZ);
    // Translate current camera
    void translateCamera (float x, float y, float z);
    // Translate current camera
    void translateCameraLocally (float x, float y, float z);
    // Translate ALL cameras
    void translateCamerasLocally (float x, float y, float z);
    // Rotate ALL cameras
    void rotateCamerasAround (float angle, float axisX, float axisY, float axisZ);
    // Rotate ALL cameras
    void rotateCamerasLocallyAround (float angle, float axisX, float axisY, float axisZ);
    // Reset camera position/rotation
    void resetCameraPose();
    // Rotates the camera around rot[X,Y,Z] around world axes and then sets translation to pos[X,Y,Z]
    void setCameraPose (float posX, float posY, float posZ, float rotX, float rotY, float rotZ);
    // Set the camera position using a transform matrix that defines the camera's local coordinate
    // axes in the world space
    void setCameraPoseMatrix (const sutil::Matrix4x4& camera_localspace);
    // Compound-specific

    bool isCompoundEyeActive();
    // Changes the current eye samples per ommatidium. WARNING: This resets the random seed
    // values. A render must be called to regenerate them, this will take significantly longer than
    // a frame render.
    void setCurrentEyeSamplesPerOmmatidium (int s);
    // Returns the current eye samples per ommatidium
    int  getCurrentEyeSamplesPerOmmatidium();
    // Changes the current eye samples per ommatidium. WARNING: This resets the random seed
    // values. A render must be called to regenerate them, this will take significantly longer than
    // a frame render.
    void changeCurrentEyeSamplesPerOmmatidiumBy (int s);
    // Returns the number of ommatidia in this eye
    size_t getCurrentEyeOmmatidialCount();
    // Sets the ommatidia for the current eye
    void setOmmatidia (OmmatidiumPacket* omms, size_t count);
    const char* getCurrentEyeDataPath();
    // Sets the compound projection shader the current eye is using
    void setCurrentEyeShaderName (char* name);

    // Scene manipulation

    // tests whether a point is within a named piece of hit geometry
    bool isInsideHitGeometry (float x, float y, float z, char* name);
    // Returns the maximal bounds of a geometry element, specified by name.
    float3 getGeometryMaxBounds (char* name);
    // Returns the minimal bounds of a geometry element, specified by name.
    float3 getGeometryMinBounds (char* name);
}

#endif
