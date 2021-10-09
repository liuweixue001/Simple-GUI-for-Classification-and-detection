#include "qfile.h"
#include "qfiledialog.h"
#include "qdebug.h"
#include "qimage.h"
#include "iostream"
#include "opencv2/imgproc/types_c.h"


#include "layer.h"
#include "net.h"
#include "iostream"
#include "ctime"
#if defined(USE_NCNN_SIMPLEOCV)
#include "simpleocv.h"
#else
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#endif
#include <float.h>
#include <stdio.h>
#include <vector>


cv::VideoCapture capture;


// ������ʶ��
struct Object
{
    // ���ο�
    cv::Rect_<float> rect;
    // ��ǩ
    int label;
    // ���Ŷ�
    float prob;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    // �����������ο��ص��������¾��ο�
    cv::Rect_<float> inter = a.rect & b.rect;
    // ���ؾ��ο����
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    // �������
    int i = left;
    // �Ҳ�����
    int j = right;
    // �м�����
    float p = faceobjects[(left + right) / 2].prob;
    // ʵ�ֿ�������
    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    // �ж��Ƿ�Ϊ�գ��ǿ��򷵻ؿ�
    if (faceobjects.empty())
        return;
    // ��������
    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    // ���size��ֻ��ͨ��push_back���Ԫ��
    picked.clear();
    // ����Ŀ��������
    const int n = faceobjects.size();
    // �����n��size�ĸ�������
    std::vector<float> areas(n);
    // ��������¼ÿ��Ŀ������
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }
    // ��������Ŀ������ݸ�ֵ���ṹ��
    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        // �����������ص�������������㽻�����
        // ����IOU����nms��ֵ�Աȣ���С����ֵ����������������ֵ�򽫼�¼��Ӧ����
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    // ǿ��ת������
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}

int detect_yolov5(const cv::Mat& bgr, std::vector<Object>& objects)
{
    ncnn::Net yolov5;
    yolov5.opt.num_threads = 8;
    yolov5.opt.use_int8_inference = true;


    yolov5.load_param("last1.param");
    yolov5.load_model("last1.bin");
    const int target_size = 448;
    const float prob_threshold = 0.8f;
    const float nms_threshold = 0.45f;
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }
    // ͼƬ�ߴ�ת��
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);


    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);
    // ���ع�һ��
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    in_pad.substract_mean_normalize(0, norm_vals);
    // ����ģ�ͼ����
    ncnn::Extractor ex = yolov5.create_extractor();
    // ����ͼƬ
    ex.input("images", in_pad);
    // �������
    std::vector<Object> proposals;


    {
        // �����������
        ncnn::Mat out;
        ex.extract("output", out);
        // ����anchors�ߴ�
        ncnn::Mat anchors(6);

        anchors[0] = 10.f;
        anchors[1] = 13.f;
        anchors[2] = 16.f;
        anchors[3] = 30.f;
        anchors[4] = 33.f;
        anchors[5] = 23.f;
        // �����������ṹ��
        std::vector<Object> objects8;
        // ���ɼ���
        generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects8);
        // �������
        proposals.insert(proposals.end(), objects8.begin(), objects8.end());
    }
    // �г߶�anchors�������ͬ��
    // stride 32
    {
        ncnn::Mat out;
        // ��������֣������ģ���޸�
        ex.extract("237", out);

        ncnn::Mat anchors(6);
        anchors[0] = 30.f;
        anchors[1] = 61.f;
        anchors[2] = 62.f;
        anchors[3] = 45.f;
        anchors[4] = 59.f;
        anchors[5] = 119.f;

        std::vector<Object> objects16;
        generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects16);

        proposals.insert(proposals.end(), objects16.begin(), objects16.end());
    }
    // ��߶�anchors�������ͬ��
    // stride 64
    {
        ncnn::Mat out;
        // ��������֣������ģ���޸�
        ex.extract("257", out);

        ncnn::Mat anchors(6);

        anchors[0] = 116.f;
        anchors[1] = 90.f;
        anchors[2] = 156.f;
        anchors[3] = 198.f;
        anchors[4] = 373.f;
        anchors[5] = 326.f;
        std::vector<Object> objects32;
        generate_proposals(anchors, 64, in_pad, out, prob_threshold, objects32);

        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
    }
    // �Լ��������Ŷ�����
    // sort all proposals by score from highest to lowest
    qsort_descent_inplace(proposals);
    // ͨ���Ǽ���ֵ����ɸѡ����
    // apply nms with nms_threshold
    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);
    // ��¼��������
    int count = picked.size();
    // �����תΪ���������ĳߴ�
    objects.resize(count);
    for (int i = 0; i < count; i++)
    {
        // �������õ����������Ӧ��λ����Ϣ
        objects[i] = proposals[picked[i]];
        // ����������������ƫ��
        // adjust offset to original unpadded
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;
        // clip
        x0 = (std::max)((std::min)(x0, (float)(img_w - 1)), 0.f);
        y0 = (std::max)((std::min)(y0, (float)(img_h - 1)), 0.f);
        x1 = (std::max)((std::min)(x1, (float)(img_w - 1)), 0.f);
        y1 = (std::max)((std::min)(y1, (float)(img_h - 1)), 0.f);
        // ȷ������λ��
        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;
    }

    return 0;
}

cv::Mat draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects, int num)
{

    // ����Ŀ������
    static const char* class_names[] = { "winding" };
    // ����ͼƬ
    cv::Mat image = bgr.clone();
    // ���Ƽ���
    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];
        // ������λ��
        fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
            obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
        // ����
        cv::rectangle(image, obj.rect, cv::Scalar(0, 255, 0));
        // д��label
        char text[256];
        sprintf_s(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        // д��ͼƬ
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        // ȷ�����ο�λ��
        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;
        // ����
        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
            cv::Scalar(255, 255, 255), -1);
        // д��
        cv::putText(image, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
    }
    cv::waitKey(num);
    return image;
}

// ��������
std::string detect_simple_model(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net simple_model;
    simple_model.load_param("simple_model.param");
    simple_model.load_model("simple_model.bin");
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
    //����pytorch�������б�׼��
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


// �������
std::string detect_over_model(const cv::Mat& bgr, std::vector<float>& cls_scores)
{
    ncnn::Net simple_model;
    simple_model.load_param("classed_sim1.param");
    simple_model.load_model("classed_sim1.bin");
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, bgr.cols, bgr.rows, 224, 224);
    //����pytorch�������б�׼��
    const float mean_vals[3] = { 0.485f, 0.456f, 0.406f };
    const float std_vals[3] = { 1 / 0.229f, 1 / 0.224f, 1 / 0.225f };
    const float norm_255[3] = { 1 / 255.0f, 1 / 255.0f, 1 / 255.0f };
    in.substract_mean_normalize(0, norm_255);
    in.substract_mean_normalize(mean_vals, std_vals);
    ncnn::Extractor ex = simple_model.create_extractor();
    ex.input("input.1", in);
    ncnn::Mat out;
    ex.extract("15", out);
    std::string result = (out[0] > out[1]) ? "right" : "wrong";
    return result;
}


// Qt����
qfile::qfile(QWidget* parent)
    : QWidget(parent)
    , ui(new Ui::qfileClass)
{
    //cv::Mat frame_detcetion, frame_class, frame_over;
    open_video = false;
    ui->setupUi(this);


    //��ͣ
    connect(ui->pushButton_2, &QPushButton::clicked, [=]() {
        num = 0;
        play = false;
        if (!open_video)
        {
            ui->label_2->setText("please select a video for detection");
        }
        });


    //����
    connect(ui->pushButton_3, &QPushButton::clicked, [=]() {
        num = 1;
        play = true;
        if (!open_video)
        {
            ui->label_2->setText("please select a video for detection");
            play = false;
        }
        while (play)
        {
            switch (ui->comBox->currentIndex()) {
            case 0: {
                ui->label_2->setText("Is is detecting for Z75");
                capture >> frame_detcetion;
                std::vector<Object> objects;
                detect_yolov5(frame_detcetion, objects);
                cv::Mat frame_detcetion1;
                frame_detcetion1 = draw_objects(frame_detcetion, objects, num);
                cv::waitKey(num);
                int height = ui->label->size().height();
                int width = ui->label->size().width();
                cv::resize(frame_detcetion1, frame_detcetion1, cv::Size(width, height));
                cv::cvtColor(frame_detcetion1, frame_detcetion1, CV_BGR2RGB);
                QImage Qtemp = QImage((const unsigned char*)(frame_detcetion1.data), frame_detcetion1.cols, frame_detcetion1.rows,
                    frame_detcetion1.step, QImage::Format_RGB888);
                ui->label->setPixmap(QPixmap::fromImage(Qtemp));
                break;}
            case 1: {
                ui->label_2->setText("Is is detcting luansheng for Z75");
                capture >> frame_class;
                cv::Mat frame_class1;
                frame_class1 = frame_class.clone();
                std::vector<float> cls_scores;
                std::string result = detect_simple_model(frame_class, cls_scores);
                if (result == "right")
                    cv::putText(frame_class1, result, cv::Point(559, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 4, 2);
                else
                    cv::putText(frame_class1, result, cv::Point(559, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 4, 2);
                cv::waitKey(num);
                int height = ui->label->size().height();
                int width = ui->label->size().width();
                cv::resize(frame_class1, frame_class1, cv::Size(width, height));
                cv::cvtColor(frame_class1, frame_class1, CV_BGR2RGB);
                QImage Qtemp = QImage((const unsigned char*)(frame_class1.data), frame_class1.cols, frame_class1.rows,
                    frame_class1.step, QImage::Format_RGB888);
                ui->label->setPixmap(QPixmap::fromImage(Qtemp));
                break;}
            case 2: {
                ui->label_2->setText("Is is detecting over for Z110");
                capture >> frame_over;
                std::vector<float> cls_scores;
                std::string result = detect_over_model(frame_over, cls_scores);
                if (result == "right") {
                    cv::putText(frame_over, "Show result: ", cv::Point(359, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 4, 2);
                    cv::putText(frame_over, "Normal", cv::Point(559, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 255, 0), 4, 2);
                }
                else {
                    cv::putText(frame_over, "Show result: ", cv::Point(359, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 4, 2);
                    cv::putText(frame_over, "Excessive", cv::Point(559, 50), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0, 0, 255), 4, 1);
                }
                cv::waitKey(num);
                int height = ui->label->size().height();
                int width = ui->label->size().width();
                cv::resize(frame_over, frame_over, cv::Size(width, height));
                cv::cvtColor(frame_over, frame_over, CV_BGR2RGB);
                QImage Qtemp = QImage((const unsigned char*)(frame_over.data), frame_over.cols, frame_over.rows,
                    frame_over.step, QImage::Format_RGB888);
                ui->label->setPixmap(QPixmap::fromImage(Qtemp));
                break;}
            }
        }
        });


    //��ȡ��Ƶ
    connect(ui->pushButton, &QPushButton::clicked, [=]() {
        filepath = QFileDialog::getOpenFileName(this, "open the file", "I:/test");
        ui->lineEdit->setText(filepath);
        std::string path = filepath.toStdString();
        capture = cv::VideoCapture(path);
        if (!capture.isOpened())
        {
            qDebug() << "Read video Failed !";
        }
        open_video = true;
        });
}


