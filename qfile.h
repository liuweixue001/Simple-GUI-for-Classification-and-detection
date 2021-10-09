#pragma once

#include <QtWidgets/QWidget>
#include "ui_qfile.h"
#include "opencv2/opencv.hpp"
class qfile : public QWidget
{
    Q_OBJECT

public:
    qfile(QWidget *parent = Q_NULLPTR);
    QString filepath;
    void zanting();
    void bofang();
    int num;
    QImage Qtemp;
    bool play;
    cv::VideoCapture capture;
    bool open_video;
    cv::Mat frame_detcetion, frame_class, frame_over;
private:
    Ui::qfileClass *ui;
};
