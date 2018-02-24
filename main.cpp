#include "zoezentracker.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    ZoezenTracker w;
    w.show();

    return a.exec();
}
