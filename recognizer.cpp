#include <assert.h>
#include <sys/stat.h>
#include <sys/types.h>

#include "timer.h"
#include "recognizer.h"

int findNearestNeighbor(cv::Mat projectedTestFace, float *pConfidence, Trainer *trainer);

rec_result recognizeFromImage(cv::Mat camImg, Trainer *trainer)
{
    cv::Mat projectedTestFace;
    cv::Mat greyImg;
    cv::Mat sizedImg;
    cv::Mat equalizedImg;
    rec_result result;

    tick();

    // Make sure the image is greyscale, since the Eigenfaces is only done on greyscale image.
    if (camImg.channels() > 1)
        cv::cvtColor(camImg, greyImg, CV_BGR2GRAY);
    else
        greyImg = camImg;

    // Make sure the image is the same dimensions as the training images.
    cv::resize(greyImg, sizedImg, trainer->faceSize);
    // Give the image a standard brightness and contrast, in case it was too dark or low contrast.
    cv::equalizeHist(sizedImg, equalizedImg);
    GaussianBlur( equalizedImg, equalizedImg, cv::Size(7,7), 3 );

    // project the test image onto the PCA subspace
    // XXX need to resize
    projectedTestFace = trainer->pca->project(equalizedImg.reshape(0, 1));

    // Check which person it is most likely to be.
    result.iNearest = findNearestNeighbor(projectedTestFace, &result.confidence, trainer);
    result.nearest  = trainer->personNumTruthMat.at<uint16_t>(result.iNearest);

    result.recognizeTime = tock();
    return result;
}

// Find the most likely person based on a detection. Returns the index, and stores the confidence value into pConfidence.
int findNearestNeighbor(cv::Mat projectedTestFace, float *pConfidence, Trainer *trainer)
{
    //double leastDistSq = 1e12;
    double leastDistSq = DBL_MAX;
    double totDistSq = 0;
    int i, iTrain, iNearest = 0;
    const float *facedata = (const float *)projectedTestFace.data;

    for(iTrain=0; iTrain<trainer->nFaces; iTrain++) {
        double distSq=0;
        const float *trainFaceData = (const float *)trainer->projectedTrainFaceMat.row(iTrain).data;

        for(i=0; i<trainer->nEigens; i++) {
            float d_i = facedata[i] - trainFaceData[i];
#define USE_MAHALANOBIS_DISTANCE
#ifdef USE_MAHALANOBIS_DISTANCE
            distSq += d_i*d_i / ((float *)trainer->pca->eigenvalues.data)[i];  // Mahalanobis distance (might give better results than Eucalidean distance)
#else
            distSq += d_i*d_i; // Euclidean distance.
#endif
        }

        totDistSq += distSq;
        if(distSq < leastDistSq) {
            leastDistSq = distSq;
            iNearest = iTrain;
        }
    }

    // Return the confidence level based on the Euclidean distance,
    // so that similar images should give a confidence between 0.5 to 1.0,
    // and very different images should give a confidence between 0.0 to 0.5.
    double avgDist = sqrt(totDistSq/(double)(trainer->nFaces));
    *pConfidence = 1.0f - sqrt(leastDistSq)/avgDist;

    // Return the found index.
    return iNearest;
}
