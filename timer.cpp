#include <opencv2/opencv.hpp>

static double timecnt;

void tick(void)
{
    timecnt = (double)cvGetTickCount();
}

int tock(void)
{
    double t = (double)cvGetTickCount() - timecnt;
    return cvRound(t/((double)cvGetTickFrequency()*1000.0));
}
