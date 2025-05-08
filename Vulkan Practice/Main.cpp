// GLFW will include its own definitions and automatically load the Vulkan header with it
#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

/* The stdexcept and iostream headers are included for reporting and propagating errors. The cstdlib header provides the
EXIT_SUCCESS and EXIT_FAILURE macros */
#include <iostream>
#include <stdexcept>
#include <cstdlib>

#include <vector>

const uint32_t WIDTH = 800;
const uint32_t HEIGHT = 600;

class HelloTriangleApplication 
{
public:
    void run() 
    {
        initWindow();
        initVulkan();
        mainLoop();
        cleanup();
    }

private:

    GLFWwindow* window;
    VkInstance instance;

    void createInstance()
    {
        /* To create an instance we'll first have to fill in a struct with some information about our application. This data
        is technically optional, but it may provide some useful information to the driver in order to optimize our specific
        application. This struct is called VkApplicationInfo */
        VkApplicationInfo appInfo{};

        // Many structs in Vulkan requires explicitly specifying the type in the sType member
        appInfo.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
        appInfo.pApplicationName = "Hello Triangle";
        appInfo.applicationVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.pEngineName = "No Engine";
        appInfo.engineVersion = VK_MAKE_VERSION(1, 0, 0);
        appInfo.apiVersion = VK_API_VERSION_1_0;

        /* Fill in one more struct to provide sufficient information for creating an instance. This next struct is not
        optional and tells the Vulkan driver which global extensions and validation layers we want to use. Global here means
        that they apply to the entire program and not a specific device */
        VkInstanceCreateInfo createInfo{};
        createInfo.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
        createInfo.pApplicationInfo = &appInfo;

        // GLFW has a handy built-in function that returns any extensions it needs to do that which can be passed to the struct
        uint32_t glfwExtensionCount = 0;
        const char** glfwExtensions;

        glfwExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);

        // Determine the global validation layers to enable
        createInfo.enabledExtensionCount = glfwExtensionCount;
        createInfo.ppEnabledExtensionNames = glfwExtensions;
        createInfo.enabledLayerCount = 0;

        // We've specified everything Vulkan needs to create an instance and we can finally issue the vkCreateInstance call
        VkResult result = vkCreateInstance(&createInfo, nullptr, &instance);

        /* Check if the instance was created successfully, we don't need to store the result and can just use a check for the
        success value */
        if (vkCreateInstance(&createInfo, nullptr, &instance) != VK_SUCCESS) {
            throw std::runtime_error("failed to create instance!");
        }

        // To allocate an array to hold the extension details we first need to know how many there are
        uint32_t extensionCount = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);

        // Allocate an array to hold the extension details
        std::vector<VkExtensionProperties> extensions(extensionCount);

        // Finally we can query the extension details
        vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, extensions.data());

        std::cout << "available extensions:\n";

        /* Each VkExtensionProperties struct contains the name and version of an extension. We can list them with a simple
        for loop (\t is a tab for indentation) */
        for (const auto& extension : extensions) 
        {
            std::cout << '\t' << extension.extensionName << '\n';
        }
    }

    void initWindow() 
    {
        glfwInit(); //  Initialize the GLFW library

        // Because GLFW was originally designed to create an OpenGL context, tell it not to create an OpenGL context
        glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

        // Disable window resizing
        glfwWindowHint(GLFW_RESIZABLE, GLFW_FALSE);

        /* The first three parameters specify the width, height and title of the window. The fourth parameter allows you to
        optionally specify a monitor to open the window on and the last parameter is only relevant to OpenGL */
        window = glfwCreateWindow(WIDTH, HEIGHT, "Vulkan", nullptr, nullptr);
    }

    void initVulkan() 
    {
        createInstance();
    }

    void mainLoop() 
    {
        // To keep the application running until either an error occurs or the window is closed, we need to add an event loop
        while (!glfwWindowShouldClose(window)) 
        {
            // It loops and checks for events like pressing the X button until the window has been closed by the user
            glfwPollEvents();
        }
    }

    void cleanup() 
    {
        // The VkInstance should be only destroyed right before the program exits by calling vkDestroyInstance

        /* The allocation and deallocation functions in Vulkan have an optional allocator callback that we'll ignore by
        passing nullptr to it */
        vkDestroyInstance(instance, nullptr);

        glfwDestroyWindow(window);

        glfwTerminate();
    }
};

int main() 
{
    HelloTriangleApplication app;

    try 
    {
        app.run();
    }

    catch (const std::exception& e) 
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}