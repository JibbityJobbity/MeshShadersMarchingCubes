#include "stdafx.h"

#include "VulkanRenderer.h"

int main() {
    if (enableValidationLayers)
        std::cerr << "==========VALIDATION LAYERS ARE ENABLED=========="<< std::endl
                  << "BE SURE TO DISABLE FOR GATHERING PERFORMANCE DATA" << std::endl;
    VulkanRenderer renderer;

    try {
        renderer.run();
    }
    catch (const std::exception& e) {
#ifdef _WIN32
        std::string message(e.what());
        std::wstring wMessage;
        wMessage.resize(message.length());
        for (size_t i = 0; i < message.length(); i++) {
            wMessage[i] = message[i];
        }
        int mBox = MessageBox(
            NULL,
            (LPCWSTR)wMessage.c_str(),
            (LPCWSTR)L"Renderer Error",
            MB_ICONERROR | MB_DEFBUTTON1 | MB_OK
        );
#endif
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}