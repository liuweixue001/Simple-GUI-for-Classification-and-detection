#include "qfile.h"
#include <QtWidgets/QApplication>
int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    qfile w;
    w.show();
    return a.exec();
}
