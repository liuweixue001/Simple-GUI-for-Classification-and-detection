#include "main.h"


//调用模型
string detect_simple_model(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net simple_model;
    simple_model.load_param("classed_sim.param");
    simple_model.load_model("classed_sim.bin");
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
    //按照pytorch样例进行标准化
    const float mean_vals[3] = { 0.485f, 0.456f, 0.406f };
    const float std_vals[3] = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
    const float norm_255[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f };
    in.substract_mean_normalize(0, norm_255);
    in.substract_mean_normalize(mean_vals, std_vals);
    ncnn::Extractor ex = simple_model.create_extractor();
    ex.input("input.1", in);
    ncnn::Mat out;
    ex.extract("25", out);
    std::string result = (out[0] > out[1]) ? "right" : "wrong";
    return result;
}
#include "pch.h"