#include <iostream>

#include <glad/glad.h>
#include <cuda_runtime.h>
#include "GLDisplay.h"

#include <sutil/CUDAOutputBuffer.h>

#include <sutil/sutil.h>
#include "libEyeRenderer.h"
#include <GLFW/glfw3.h>

#include <sutil/vec_math.h>
#include "BasicController.h"

// This subproject loads in libEyeRenderer and uses it to render a given scene.
// Basic controls are offered.
// It also stands as an example of how to interface the the rendering library.

bool dirtyUI = true; // a flag to keep track of if the UI has changed in any way
BasicController controller;

// An initial width and height
constexpr int32_t win_width = 400;
constexpr int32_t win_height = 400;

// These are from libEyeRenderer and are runtime-mutable
extern int32_t width;
extern int32_t height;

// Was in CUDAOutputBuffer
GLuint _pbo = 0;

// Was in CUDAOutputBuffer
template <typename PIXEL_FORMAT>
GLuint getPBO (sutil::CUDAOutputBuffer<PIXEL_FORMAT>* buf_ptr)
{
    if (_pbo == 0u) { GL_CHECK (glGenBuffers (1, &_pbo)); }

    const size_t buffer_size = buf_ptr->area() * sizeof(PIXEL_FORMAT);

    const auto typ = buf_ptr->getType();
    if (typ == sutil::CUDAOutputBufferType::CUDA_DEVICE) {
        // We need a host buffer to act as a way-station
        if (buf_ptr->m_host_pixels.empty()) { buf_ptr->m_host_pixels.resize (buf_ptr->area()); }

        buf_ptr->makeCurrent();
        CUDA_CHECK(cudaMemcpy (static_cast<void*>(buf_ptr->m_host_pixels.data()),
                               buf_ptr->m_device_pixels,
                               buffer_size,
                               cudaMemcpyDeviceToHost));

        GL_CHECK (glBindBuffer (GL_ARRAY_BUFFER, _pbo));
        GL_CHECK (glBufferData (GL_ARRAY_BUFFER,
                                buffer_size,
                                static_cast<void*>(buf_ptr->m_host_pixels.data()),
                                GL_STREAM_DRAW));
        GL_CHECK (glBindBuffer (GL_ARRAY_BUFFER, 0));

    } else if (typ == sutil::CUDAOutputBufferType::GL_INTEROP || typ == sutil::CUDAOutputBufferType::CUDA_P2P) {
        throw sutil::Exception("Unsupported");

    } else { // typ == CUDAOutputBufferType::ZERO_COPY
        GL_CHECK (glBindBuffer (GL_ARRAY_BUFFER, _pbo));
        GL_CHECK (glBufferData (GL_ARRAY_BUFFER,
                                buffer_size,
                                static_cast<void*>(buf_ptr->m_host_zcopy_pixels),
                                GL_STREAM_DRAW));
        GL_CHECK (glBindBuffer (GL_ARRAY_BUFFER, 0));
    }

    return _pbo;
}

void initGL()
{
    int glad_gl_version = gladLoadGL();
    if (!glad_gl_version) { throw sutil::Exception ("Failed to initialize GL functions"); }
    GL_CHECK (glClearColor (0.212f, 0.271f, 0.31f, 1.0f));
    GL_CHECK (glClear (GL_COLOR_BUFFER_BIT));
}

static void errorCallback (int error, const char* description)
{
    std::cerr << "GLFW Error " << error << ": " << description << std::endl;
}

GLFWwindow* initGLFW (const char* window_title, int width, int height, bool visible = true)
{
    glfwSetErrorCallback (errorCallback);
    if(!glfwInit()) { throw sutil::Exception ("Failed to initialize GLFW"); }

    glfwWindowHint (GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint (GLFW_CONTEXT_VERSION_MINOR, 1);
    glfwWindowHint (GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE); // Removes functions deprecated in version <4.1
    glfwWindowHint (GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint (GLFW_VISIBLE, (visible ? GLFW_TRUE : GLFW_FALSE));

    GLFWwindow* window = glfwCreateWindow (width, height, window_title, nullptr, nullptr);
    if( !window ) { throw sutil::Exception ("Failed to create GLFW window"); }
    glfwMakeContextCurrent (window);
    glfwSwapInterval (0);  // No vsync

    return window;
}

GLFWwindow* initWindow (const char* window_title, int width, int height)
{
    GLFWwindow* window = initGLFW (window_title, width, height);
    glfwMakeContextCurrent (window);
    initGL();
    return window;
}

// Global pointers to resources
GLFWwindow* window = initWindow ("CompoundRay Example GUI", win_width, win_height);
extern sutil::CUDAOutputBuffer<uchar4>* outputBuffer;
sutil::GLDisplay* gl_display = nullptr;

static void keyCallback( GLFWwindow* window, int32_t key, int32_t /*scancode*/, int32_t action, int32_t /*mods*/)
{
    // Handle keypresses
    if(action == GLFW_PRESS) {
        if (key == GLFW_KEY_ESCAPE || key == GLFW_KEY_Q) {
            // Control keypresses
            glfwSetWindowShouldClose (window, true);
        } else {
            //// Movement keypresses

            // Camera changing
            if (key == GLFW_KEY_N) {
                std::cout << "switch to next camera...";
                nextCamera();
                std::cout << " " << getCurrentCameraName()
                          << (isCompoundEyeActive() ? " which is compound\n" : " which is isn't compound\n");
            } else if (key == GLFW_KEY_B) {
                std::cout << "switch to prev camera...";
                previousCamera();
                std::cout << " " << getCurrentCameraName()
                          << (isCompoundEyeActive() ? " which is is compound\n" : " which is isn't compound\n");
            } else if (key == GLFW_KEY_PAGE_UP) {
                int csamp = getCurrentEyeSamplesPerOmmatidium();
                if (csamp < 32000) {
                    changeCurrentEyeSamplesPerOmmatidiumBy(csamp); // double
                } else {
                    // else graphics memory use will get very large
                    std::cout << "max allowed samples\n";
                }
            } else if (key == GLFW_KEY_PAGE_DOWN) {
                int csamp = getCurrentEyeSamplesPerOmmatidium();
                changeCurrentEyeSamplesPerOmmatidiumBy(-(csamp/2)); // halve
            } else if (key == GLFW_KEY_C) {
                saveFrameAs("output.ppm");
            }
            dirtyUI = true;
        }
    }
    // Camera movement (mark the UI dirty if the controller has moved
    dirtyUI |= controller.ingestKeyAction(key, action);
}

static void windowSizeCallback (GLFWwindow* _window, int32_t res_x, int32_t res_y)
{
    setRenderSize (res_x, res_y);
    dirtyUI = true;
}

void printHelp()
{
  std::cout << "USAGE:\nnewGuiEyeRenderer -f <path to gltf scene>" << std::endl << std::endl;
  std::cout << "\t-h\tDisplay this help information." << std::endl;
  std::cout << "\t-f\tPath to a gltf scene file (absolute or relative to current working directory, e.g. './natural-standin-sky.gltf')." << std::endl;
}

void displayFrame()
{
    int fb_res_x = 0; // The display's resolution (could be HDPI res)
    int fb_res_y = 0;
    glfwGetFramebufferSize (window, &fb_res_x, &fb_res_y);

    // If compound eye active, copy data into outputBuffer.

    if (outputBuffer != nullptr) {
        gl_display->display (outputBuffer->width(), outputBuffer->height(),
                             fb_res_x, fb_res_y,
                             getPBO<uchar4> (outputBuffer));
    }
    glfwSwapBuffers (window);
}

int main (int argc, char* argv[])
{
    std::cout << "Running eye Renderer GUI...\n";

    // Initialize the rendering size:
    width = win_width;
    height = win_height;
    // Allocates a scene, launch params and output buffer in libEyeRenderer
    multicamAlloc();

    // We DO require the noncompound OptiX pipeline in this program:
    setRequireNoncompoundPipeline (true);

    gl_display = new sutil::GLDisplay();

    // Parse Inputs
    std::string path = "";
    for (int i=0; i<argc; i++) {
        std::string arg = std::string(argv[i]);
        if (arg == "-h") {
            printHelp();
            return 0;
        } else if(arg == "-f") {
            i++;
            path = std::string(argv[i]);
        }
    }

    if(path == "") {
        printHelp();
        return 1;
    }

    // Attach callbacks
    glfwSetKeyCallback (window, keyCallback);
    glfwSetWindowSizeCallback (window, windowSizeCallback);

    int rtn = 0;
    try {
        // Turn off verbose logging
        setVerbosity(false);

        // Load the file
        std::cout << "Loading file \"" << path << "\"..." << std::endl;
        loadGlTFscene(path.c_str(), sutil::Matrix4x4::identity());

        std::cout << "Initial camera is " << getCurrentCameraName()
                  << (isCompoundEyeActive() ? " which is compound\n" : " which isn't compound\n");

        // The main loop
        do {
            glfwPollEvents(); // Check if anything's happened, user-input-wise.

            if (controller.isActivelyMoving()) {
                float3 t = controller.getMovementVector();// Local translation
                translateCamerasLocally(t.x, t.y, t.z);
                float va = controller.getVerticalRotationAngle();
                float vh = controller.getHorizontalRotationAngle();
                rotateCamerasLocallyAround (va, 1.0f, 0.0f, 0.0f);
                rotateCamerasAround (vh, 0.0f, 1.0f, 0.0f);
                dirtyUI = true;
            }

            // Render and display the frame if anything's changed (movement or window resize etc)
            // also re-render the frame if the current camera is a compound eye in order to get a
            // better feeling of the stochastic spread encountered.
            if (dirtyUI || isCompoundEyeActive()) {
                renderFrame();
                displayFrame();
                dirtyUI = false; // Comment this out to force constant re-rendering
            }

        } while (!glfwWindowShouldClose (window));
        stop();

    } catch (std::exception& e) {
        std::cerr << "Caught exception: " << e.what() << "\n";
        rtn = 1;
    }

    delete gl_display;
    multicamDealloc();

    return rtn;
}
