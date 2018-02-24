#ifndef POSERECORDERDLG_H
#define POSERECORDERDLG_H

#include <QDialog>
#include "MarkerDetector.h"

namespace Ui {
class Poserecorderdlg;
}

class Poserecorderdlg : public QDialog
{
    Q_OBJECT

public:
    explicit Poserecorderdlg(QWidget *parent = 0);
    ~Poserecorderdlg();

    void saveMarkerInfo(QString filename);

	void recordMarkerInfo(vector<DetectedMarker> Maker_Detected);

	
	private slots:
    void on_pushButton_save_clicked();

	void on_pushButton_record_clicked();



private:
    Ui::Poserecorderdlg *ui;

	bool recordFlag;

	QStringList markerInfoList;
};

#endif // POSERECORDERDLG_H
