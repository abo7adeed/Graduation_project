#include <iostream>
#include <fstream>
#include <cstdint>
#include <vector>
#include <string>
#include <tensorflow/lite/interpreter.h>
#include <tensorflow/lite/model.h>
#include <tensorflow/lite/kernels/register.h>
#include <tensorflow/lite/string_util.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>
#include <filesystem>

using namespace std;
using namespace tflite;
using namespace cv;

constexpr int kImageWidth = 128;
constexpr int kImageHeight = 128;
constexpr int kImageChannels = 3;

int main(int argc, char* argv[]) {

    std::string model_file = "E:/Graduation_project/TASKs/1.MOVENET/Git/tflite/open_close_camera/assets/model.tflite";
    std::string output_folder = "E:/Graduation_project/TASKs/1.MOVENET/Git/tflite/open_close_haarcascade_image_folder/output_images";  // Set the path to the output folder

    // Load the TFLite model
    auto model = tflite::FlatBufferModel::BuildFromFile(model_file.c_str());
    if (!model) {
        throw std::runtime_error("Failed to load TFLite model");
        return 1;
    }

    // Load the Haarcascade classifiers for face and eyes
    cv::CascadeClassifier face_cascade;
    cv::CascadeClassifier leye_cascade;
    cv::CascadeClassifier reye_cascade;

    if (!face_cascade.load("E:/Graduation_project/TASKs/1.MOVENET/Git/tflite/open_close_haarcascade/cascade_model/haarcascade_frontalface_alt.xml") ||
        !leye_cascade.load("E:/Graduation_project/TASKs/1.MOVENET/Git/tflite/open_close_haarcascade/cascade_model/haarcascade_lefteye_2splits.xml") ||
        !reye_cascade.load("E:/Graduation_project/TASKs/1.MOVENET/Git/tflite/open_close_haarcascade/cascade_model/haarcascade_righteye_2splits.xml")) {
        std::cerr << "Failed to load Haarcascade classifiers!" << std::endl;
        return 1;
    }

    // Build the interpreter
    tflite::ops::builtin::BuiltinOpResolver resolver;
    tflite::InterpreterBuilder builder(*model, resolver);
    std::unique_ptr<tflite::Interpreter> interpreter;
    if (builder(&interpreter) != kTfLiteOk) {
        std::cerr << "Failed to build interpreter!" << std::endl;
        return 1;
    }

    // Allocate tensors
    interpreter->AllocateTensors();

    // Check the input tensor format
    auto input_tensor = interpreter->tensor(interpreter->inputs()[0]);
    input_tensor->dims->data[0] = 1;
    input_tensor->dims->data[1] = kImageHeight;
    input_tensor->dims->data[2] = kImageWidth;
    input_tensor->dims->data[3] = kImageChannels;

    cout << "Input tensor shape: " << input_tensor->dims->data[0] << " "
         << input_tensor->dims->data[1] << " " << input_tensor->dims->data[2] << " "
         << input_tensor->dims->data[3] << endl;
    cout << "Input tensor data type: " << TfLiteTypeGetName(input_tensor->type) << endl;

    if (argc > 1) {
        std::string input_folder = argv[1];

        // Check if the path is a directory
        if (!std::filesystem::is_directory(input_folder)) {
            std::cerr << "Provided path is not a directory." << std::endl;
            return 1;
        }

        // Create the output folder if it doesn't exist
        std::filesystem::create_directory(output_folder);

        // Get a list of image files in the specified folder
        std::vector<std::string> image_files;
        for (const auto& entry : std::filesystem::directory_iterator(input_folder)) {
            if (entry.is_regular_file() && (entry.path().extension() == ".jpg" || entry.path().extension() == ".png")) {
                image_files.push_back(entry.path().string());
            }
        }

        // Declare output_path outside the if block
        std::filesystem::path output_path;

        // Process each image in the folder
        for (const auto& image_path : image_files) {
            cv::Mat frame = cv::imread(image_path, cv::IMREAD_COLOR);

            std::cout << "Processing image: " << image_path << std::endl;
            if (frame.empty()) {
                std::cerr << "Failed to read image: " << image_path << std::endl;
                continue;  // Move to the next image if loading fails
            }

            // Convert frame to grayscale for face and eye detection
            cv::Mat gray_frame;
            cv::cvtColor(frame, gray_frame, cv::COLOR_BGR2GRAY);

            // Detect faces
            std::vector<cv::Rect> faces;
            face_cascade.detectMultiScale(gray_frame, faces, 1.3, 5, 0, cv::Size(30, 30));

            if (faces.empty()) {
                std::cout << "No faces detected." << std::endl;
            }

            for (const auto& face : faces) {
                std::cout << "Face detected. Coordinates: "
                          << "x=" << face.x << ", y=" << face.y
                          << ", width=" << face.width << ", height=" << face.height << std::endl;

                cv::rectangle(frame, face, cv::Scalar(100, 100, 100), 1);

                // Extract regions of interest (ROI) for left and right eyes within the detected face
                Rect roi_l_rect(face.x + face.width / 8, face.y + face.height / 4, face.width / 4, face.height / 4);
                Rect roi_r_rect(face.x + (face.width / 2) + face.width / 8, face.y + face.height / 4, face.width / 4, face.height / 4);

                cv::Mat roi_l_eye = gray_frame(roi_l_rect);
                cv::Mat roi_r_eye = gray_frame(roi_r_rect);

                // Resize the eyes to match the model input size
                cv::resize(roi_l_eye, roi_l_eye, cv::Size(kImageWidth, kImageHeight));
                cv::resize(roi_r_eye, roi_r_eye, cv::Size(kImageWidth, kImageHeight));

                // Convert the eyes to the required format for TensorFlow Lite
                cv::cvtColor(roi_l_eye, roi_l_eye, cv::COLOR_GRAY2RGB);
                cv::cvtColor(roi_r_eye, roi_r_eye, cv::COLOR_GRAY2RGB);

                cv::Mat input_image_float_l, input_image_float_r;
                roi_l_eye.convertTo(input_image_float_l, CV_32FC3); // Convert to float32
                roi_r_eye.convertTo(input_image_float_r, CV_32FC3); // Convert to float32

                input_image_float_l /= 255.0f; // Normalize pixel values to [0, 1]
                input_image_float_r /= 255.0f; // Normalize pixel values to [0, 1]

                // Copy input data into the input tensor buffer for left eye
                std::memcpy(interpreter->typed_input_tensor<float>(0), input_image_float_l.data,
                            input_image_float_l.total() * input_image_float_l.elemSize());

                // Run inference for left eye
                std::cout << "Running inference for left eye." << std::endl;
                interpreter->Invoke();

                // Get output tensor data for left eye
                const auto* output_l = interpreter->tensor(interpreter->outputs()[0]);
                const float* output_data_l = output_l->data.f;

                // Copy input data into the input tensor buffer for right eye
                std::memcpy(interpreter->typed_input_tensor<float>(1), input_image_float_r.data,
                            input_image_float_r.total() * input_image_float_r.elemSize());

                // Run inference for right eye
                std::cout << "Running inference for right eye." << std::endl;
                interpreter->Invoke();

                // Get output tensor data for right eye
                const auto* output_r = interpreter->tensor(interpreter->outputs()[0]);  // Assuming only one output tensor
                const float* output_data_r = output_r->data.f;

                // Perform classification based on output data for left eye
                int predicted_class_l = (output_data_l[0] > output_data_l[1]) ? 0 : 1;

                // Perform classification based on output data for right eye
                int predicted_class_r = (output_data_r[0] > output_data_r[1]) ? 0 : 1;

                // Display the result on the image
                cv::Scalar text_color_l = (predicted_class_l == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
                cv::Scalar text_color_r = (predicted_class_r == 0) ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);

                std::string text_l = (predicted_class_l == 0) ? "Left Eye: Open" : "Left Eye: Close";
                std::string text_r = (predicted_class_r == 0) ? "Right Eye: Open" : "Right Eye: Close";

                cv::putText(frame, text_l, cv::Point(10, frame.rows - 20), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color_l, 1, cv::LINE_AA);
                cv::putText(frame, text_r, cv::Point(10, frame.rows - 40), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, text_color_r, 1, cv::LINE_AA);

                // Save the processed image to the output folder
                output_path = std::filesystem::path(output_folder) / std::filesystem::path(image_path).filename();
                cv::imwrite(output_path.string(), frame);
            }
        }
    } else {
        std::cerr << "Please provide the path to the input image folder as a command-line argument." << std::endl;
        return 1;
    }

    cv::destroyAllWindows();
    interpreter.reset();
    model.reset();
    return 0;
}
