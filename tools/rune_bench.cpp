#include "detect/detector.hpp"

#include <opencv2/videoio.hpp>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "usage: rune_video_test <engine> <video> [max_frames] [score_threshold] [keypoint_threshold]\n";
        return 2;
    }
    const int max_frames = argc >= 4 ? std::stoi(argv[3]) : 0;
    rmcs::RuneDetector detector;
    detector.config.engine_path = argv[1];
    if (argc >= 5) detector.config.score_threshold = std::stof(argv[4]);
    if (argc >= 6) detector.config.keypoint_threshold = std::stof(argv[5]);
    if (!detector.initialize()) {
        std::cerr << "failed to initialize TensorRT engine: " << argv[1] << '\n';
        return 3;
    }
    cv::VideoCapture capture(argv[2]);
    if (!capture.isOpened()) {
        std::cerr << "failed to open video: " << argv[2] << '\n';
        return 4;
    }

    int frames = 0, detected_frames = 0;
    std::size_t targets = 0;
    double inference_ms = 0.0;
    cv::Mat frame;
    while ((max_frames <= 0 || frames < max_frames) && capture.read(frame)) {
        const auto begin = std::chrono::steady_clock::now();
        const auto elements = detector.detect(frame);
        const auto end = std::chrono::steady_clock::now();
        inference_ms += std::chrono::duration<double, std::milli>(end - begin).count();
        if (!elements.bullseyes.empty()) ++detected_frames;
        targets += elements.bullseyes.size();
        ++frames;
    }
    const double average = frames ? inference_ms / frames : 0.0;
    std::cout << std::fixed << std::setprecision(2)
              << "frames=" << frames << '\n'
              << "detected_frames=" << detected_frames << '\n'
              << "targets=" << targets << '\n'
              << "average_ms=" << average << '\n'
              << "pipeline_fps=" << (average > 0.0 ? 1000.0 / average : 0.0) << '\n';
    return frames > 0 ? 0 : 5;
}
