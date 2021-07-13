#ifndef PSTATUSBAR_H
#define PSTATUSBAR_H

#include <QStatusBar>

class QProgressBar;
class QLabel;

namespace PhysIKA {
class PStatusBar : public QStatusBar
{
public:
    PStatusBar(QWidget* parent = Q_NULLPTR);

private:
    QProgressBar* m_progressBar;
    QLabel*       m_progressLabel;
};
}  // namespace PhysIKA

#endif  // PSTATUSBAR_H
