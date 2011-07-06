#include "trainer.h"

typedef struct {
    float confidence;
    int iNearest, nearest;
    int recognizeTime;
} rec_result;

rec_result recognizeFromImage(cv::Mat camImg, Trainer *trainer);
