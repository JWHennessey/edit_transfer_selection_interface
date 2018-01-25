#include <iostream>

//#include <glog/logging.h>
#include "EditTransferScreen.h"

int main(int argc, char** argv)
{
   // google::InitGoogleLogging(argv[0]);

    try {
        nanogui::init();

        {
            nanogui::ref<EditTransferScreen> app =
                new EditTransferScreen({1200, 640}, "Edit Transfer");
            app->drawAll();
            app->setVisible(true);
            nanogui::mainloop();
        }

        nanogui::shutdown();
    }
    catch (const std::runtime_error& e) {
        std::string error_msg =
            std::string("Caught a fatal error: ") + std::string(e.what());
        std::cerr << error_msg << endl;
        return -1;
    }

    return 0;
}
