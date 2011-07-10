#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <sstream>
#include <dirent.h>
#include <vector>
#include <stdio.h>

using namespace std;
using namespace cv;

void detectAndDraw( Mat& img,
                   CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
                   double scale, int sImg);


const String cascadeName = "data/haarcascades/haarcascade_frontalface_alt.xml";
const String nestedCascadeName = "data/haarcascades/haarcascade_eye_tree_eyeglasses.xml";
String captureDir = "data/captured/";
int captures = 0;

int main( int argc, const char** argv )
{
    int sImg = 0;
    string imgPath;
    vector<string> files;
    if(argc > 1){
        sImg = 1;
        imgPath = argv[1];
        DIR *dirp;
        struct dirent *entry;

        if((dirp = opendir(imgPath.c_str()))){
            while((entry = readdir(dirp))){
                if(entry->d_name[0] != '.'){
                    files.push_back(string(entry->d_name));
                    cerr << "Found file: " << string(entry->d_name) << endl;
                }
            }
            closedir(dirp);
        }
    }
    CvCapture* capture = 0;
    Mat frame, frameCopy, image;
    String inputName;
    CascadeClassifier cascade, nestedCascade;
    double scale = 1;
    captureDir += "image";

    if( !cascade.load( cascadeName ) )
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return -1;
    }

    if(!sImg)
	    capture = cvCaptureFromCAM(0);
    cvNamedWindow( "result", 1 );

    if( capture || sImg )
    {

        for(;;)
        {
            IplImage* iplImg;
            if(!sImg){
                iplImg = cvQueryFrame( capture );
                frame = iplImg;
                
                if( frame.empty() )
                    break;
                if( iplImg->origin == IPL_ORIGIN_TL )
                    frame.copyTo( frameCopy );
                else
                    flip( frame, frameCopy, 0 );
            }

            else{
                string cImg = imgPath;
                cImg += files[captures].c_str();
                frameCopy = cvLoadImage(cImg.c_str());
            }
            
            detectAndDraw( frameCopy, cascade, nestedCascade, scale, sImg );
            if(captures >= 10)
                break;
        }
        if(!sImg)
            cvReleaseCapture( &capture );
    }else{
		waitKey(0);
    }
    cvDestroyWindow("result");

    return 0;
}

void detectAndDraw( Mat& img,
                   CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
                   double scale, int sImg)
{
    int i = 0, saveFace = 0;
    double t = 0;
    vector<Rect> faces;
    Mat gray, smallImg( cvRound (img.rows/scale), cvRound(img.cols/scale), CV_8UC1 );

    cvtColor( img, gray, CV_BGR2GRAY );
    resize( gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR );
    equalizeHist( smallImg, smallImg );
    Mat eqImg(smallImg);

    t = (double)cvGetTickCount();
    cascade.detectMultiScale( smallImg, faces,
        1.1, 2, 0
        |CV_HAAR_SCALE_IMAGE
        ,
        Size(30, 30) );
    t = (double)cvGetTickCount() - t;
    printf( "detection time = %g ms\n", t/((double)cvGetTickFrequency()*1000.) );
    int key = waitKey(10);
    if(key >= 0)
        saveFace = 1;
    for( vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++ )
    {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        CvPoint pt1, pt2;
        pt1.x = r->x*scale;
        pt1.y = r->y*scale;
        pt2.x = pt1.x + r->width*scale;
        pt2.y = pt1.y + r->height*scale;
        if(saveFace||sImg){
            captures++;
            Mat faceRect = smallImg(*r);
            Mat resFace( 100, 100, CV_8UC1 );
            resize(faceRect, resFace, resFace.size(), 0, 0, INTER_LINEAR);
            GaussianBlur( resFace, resFace, Size(7,7), 3 );
            imshow("result",resFace);
            String imageFName = captureDir;
            stringstream out;
            out << captures;
            imageFName += out.str();
            imageFName += ".pgm";
            cerr << "Saving image " << imageFName << endl;
            waitKey(0);
            imwrite(imageFName, resFace);
            saveFace = 0;
        }else{
            rectangle( eqImg, pt1, pt2, CV_RGB(255,255,255), 2, 8, 0);
        }
    }
    cv::imshow( "result", eqImg );
}
