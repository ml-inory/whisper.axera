#include <stdio.h>
#include "cmdline.hpp"


int main(int argc, char** argv) {
    cmdline::parser cmd;
    cmd.add<std::string>("model", 'm', "axmodel", true, "");
    cmd.add<std::string>("wav", 'w', "wav file", true, "");
    cmd.parse_check(argc, argv);

    // 0. get app args, can be removed from user's app
    auto model_file = cmd.get<std::string>("model");
    auto wav_file = cmd.get<std::string>("wav");

    printf("model_file: %s\nwav_file: %s\n", model_file.c_str(), wav_file.c_str());

    return 0;
}
    