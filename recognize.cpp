#include <getopt.h>
#include <stdlib.h>
#include <vector>
#include <errno.h>

#include <opencv2/opencv.hpp>

#include "timer.h"
#include "recognizer.h"

void drawRectangle(cv::Mat img, cv::Rect faceRect)
{
    // Show the detected face region.
    cv::Point tl = cv::Point(faceRect.x, faceRect.y);
    cv::Point br = cv::Point(faceRect.x + faceRect.width-1, faceRect.y + faceRect.height-1);
    cv::rectangle(img, tl, br, cv::Scalar(0,255,0));
}

void perf(cv::VideoCapture cam, cv::CascadeClassifier detector, int gui)
{
    cv::Mat camImg;
    cv::Mat shownImg;
    std::vector<cv::Rect> objects;

    int scale_ms = 0;
    int noscale_ms = 0;
    if (gui)
        cv::namedWindow("Input", CV_WINDOW_AUTOSIZE);
    for (int i = 0; i < 20; i++) {
        int ms;
        cam >> camImg;
#if 0
        tick();
        detector.detectMultiScale(camImg, objects, 1.2f, 2, 0, cv::Size(20, 20));
        ms = tock();
        printf("[Face Detection took %d ms and found %zu objects]\n",
               ms, objects.size());
        noscale_ms += ms;
#endif

        tick();
        detector.detectMultiScale(camImg, objects, 1.2f, 2, detector.SCALE_IMAGE, cv::Size(20, 20));
        ms = tock();
        printf("[Face Detection took %d ms and found %zu objects]\n",
               ms, objects.size());
        if (gui) {
            shownImg = camImg.clone();
            for (std::vector<cv::Rect>::iterator i=objects.begin(); i != objects.end(); i++) {
                drawRectangle(shownImg, *i);
            }
            cv::imshow("Input", shownImg);
            cvWaitKey(10);
        }
        scale_ms += ms;

    }
    printf("Average time (scale, noscale): (%d ms, %d ms)\n", scale_ms/20, noscale_ms/20);
}

void recognizeFromCam(cv::VideoCapture cam, cv::CascadeClassifier detector, Trainer &trainer)
{
    cv::Mat camImg;
    cv::Mat faceImg;
    cv::Mat shownImg;
    rec_result result;
    cv::Rect faceRect;
    std::vector<cv::Rect> objects;

    // Create a GUI window for the user to see the camera image.
    cv::namedWindow("Input", CV_WINDOW_AUTOSIZE);
    while (1) {
        // Get the camera frame
        cam >> camImg;
        shownImg = camImg.clone();

        tick();
        detector.detectMultiScale(camImg, objects, 1.2f, 2, detector.SCALE_IMAGE, cv::Size(20, 20));
        printf("[Face Detection took %d ms and found %zu objects]\n",
                tock(), objects.size());
        if (objects.size()) {
            char *name;
            faceRect = objects[0];

            // Crop out the face image ROI
            faceImg = cv::Mat(camImg, faceRect);

            result = recognizeFromImage(faceImg, &trainer);
            name = trainer.get_name(result.nearest);
            // Show the data on the screen.
            printf("[Face Recognition took %d ms]\n", result.recognizeTime);
            printf("Most likely person in camera: '%s' (confidence=%f.\n",
                   name, result.confidence);

            // Show the detected face region.
            drawRectangle(shownImg, faceRect);
            // Show the name of the recognized person, overlayed on the image below their face.
            const int font = CV_FONT_HERSHEY_PLAIN;
            cv::Scalar textColor(0,255,255);	// light blue text
            char text[256];
            snprintf(text, sizeof(text)-1, "Name: '%s'", name);
            cv::putText(shownImg, text, cv::Point(faceRect.x, faceRect.y + faceRect.height + 15), font, 1, textColor);
            snprintf(text, sizeof(text)-1, "Confidence: %f", result.confidence);
            cv::putText(shownImg, text, cvPoint(faceRect.x, faceRect.y + faceRect.height + 30), font, 1, textColor);
            free(name);
        } else {
            printf("No face found\n");
        }

        // Display the image.
        cv::imshow("Input", shownImg);
        // Give some time for OpenCV to draw the GUI and check if the user has pressed something in the GUI window.
        if(cvWaitKey(10) != -1) {
            break;	// Stop processing input.
        }
    }
}

int verify_cb(int index, const char *filename, void *data)
{
    IplImage *img;
    Trainer *trainer = (Trainer *)data;
    rec_result result;

    img = cvLoadImage(filename, CV_LOAD_IMAGE_GRAYSCALE);
    if (!img) {
        fprintf(stderr, "Unable to load image %s\n", filename);
        return -1;
    }
    result = recognizeFromImage(img, trainer);
    printf("Verifying %s: Expect %d, Got %d [%s]\n", filename, index, result.nearest,
           (index == result.nearest) ? "Success" : "Failure");

    cvReleaseImage(&img);
    return 0;
}

void verify_training_images(Trainer *trainer)
{
    trainer->get_pictures(*verify_cb, trainer);
}

void usage(const char *prog)
{
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "Recognize mode\n");
    fprintf(stderr, "%s [--trainfile file] [--haarfile file]\n", prog);
    fprintf(stderr, "Train mode\n");
    fprintf(stderr, "%s [--trainfile file] [--picsfile file] train\n", prog);
    exit(0);
}

int main(int argc, char *argv[])
{
    int c;
    int option_index;

    const char *haarfile = "data/haarcascades/haarcascade_frontalface_alt.xml";
    const char *trainfile = "facedata.xml";
    const char *picsfile = "faces.txt";
    const char *camsrc = "0";
    const char *dbname = "test.db";

    static struct option long_options[] = {
        {"haarfile", required_argument, NULL, 'h'},
        {"trainfile", required_argument, NULL, 't'},
        {"picsfile", required_argument, NULL, 'p'},
        {"videosrc", required_argument, NULL, 'v'},
    };
    while (1) {
        c = getopt_long(argc, argv, "h:t:p:v:", long_options, &option_index);
        if (c == -1) break;

        switch (c) {
            case 'h':
                printf("Haarfile = %s\n", optarg);
                haarfile = optarg;
                break;
            case 't':
                printf("Trainfile = %s\n", optarg);
                trainfile = optarg;
                break;
            case 'p':
                printf("Picsfile = %s\n", optarg);
                picsfile = optarg;
                break;
            case 'v':
                printf("Videosrc = %s\n", optarg);
                camsrc = optarg;
                break;
            case '?':
                usage(argv[0]);
                break;
            default:
                abort();
        }

    }

    Trainer t(dbname);

    if (optind < argc && strcmp(argv[optind], "train") == 0) {
        printf("Training...\n");
        if(t.loadDbFromList(picsfile) <= 0)
            exit(1);
        if(t.learn())
            exit(1);
        printf("Training complete.  Saving...\n");
        t.storeEigenfaceImages();
        t.storeTrainingData(trainfile);
    } else if (optind < argc && strcmp(argv[optind], "verify") == 0) {
        if (t.loadTrainingData(trainfile))
            exit(1);
        verify_training_images(&t);
    } else {
        cv::VideoCapture c;
        if (camsrc[0] >= '0' && camsrc[0] <= '1') {
            int cam_num = strtol(camsrc, NULL, 10);
            c.open(cam_num);
        } else {
            c.open(camsrc);
        }
        cv::CascadeClassifier d(haarfile);
        if (!c.isOpened()) {
            printf("Failed to open video source %s\n", camsrc);
            exit(1);
        }
        if (d.empty()) {
            printf("Failed to load cascade file\n");
            exit(1);
        }
        if (optind < argc && strcmp(argv[optind], "perf") == 0) {
            perf(c, d, 1);
        } else {
            if (t.loadTrainingData(trainfile))
                exit(1);
            recognizeFromCam(c, d, t);
        }
    }
}
