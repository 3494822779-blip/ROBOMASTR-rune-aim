#include "detect/detector.hpp"

#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
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
    int red_frames = 0, red_detected = 0, blue_frames = 0, blue_detected = 0;
    std::size_t targets = 0;
    double inference_ms = 0.0;
    cv::Mat frame;
    while ((max_frames <= 0 || frames < max_frames) && capture.read(frame)) {
        const auto begin = std::chrono::steady_clock::now();
        const auto elements = detector.detect(frame);
        const auto end = std::chrono::steady_clock::now();
        inference_ms += std::chrono::duration<double, std::milli>(end - begin).count();
        if (!elements.bullseyes.empty()) ++detected_frames;
        cv::Mat hsv;
        cv::cvtColor(frame, hsv, cv::COLOR_BGR2HSV);
        cv::Mat red_low, red_high, blue;
        cv::inRange(hsv, cv::Scalar(0, 90, 80), cv::Scalar(15, 255, 255), red_low);
        cv::inRange(hsv, cv::Scalar(165, 90, 80), cv::Scalar(179, 255, 255), red_high);
        cv::inRange(hsv, cv::Scalar(90, 90, 80), cv::Scalar(140, 255, 255), blue);
        const int red_pixels = cv::countNonZero(red_low) + cv::countNonZero(red_high);
        const int blue_pixels = cv::countNonZero(blue);
        if (red_pixels > blue_pixels && red_pixels > 50) {
            ++red_frames;
            if (!elements.bullseyes.empty()) ++red_detected;
        } else if (blue_pixels > red_pixels && blue_pixels > 50) {
            ++blue_frames;
            if (!elements.bullseyes.empty()) ++blue_detected;
        }
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
    std::cout << "red_detected=" << red_detected << '/' << red_frames << '\n'
              << "blue_detected=" << blue_detected << '/' << blue_frames << '\n';
    return frames > 0 ? 0 : 5;
}
