#include <opencv2/opencv.hpp>
#include <sqlite3.h>

typedef int(*picture_cb)(int index, const char *filename, void *data);

class Trainer {
    public:
        Trainer(const char *dbfile);
        ~Trainer();
        int learn(void);
        void storeEigenfaceImages(void);
        int loadDbFromList(const char *filename);
        int loadTrainingData(const char *filename);
        int storeTrainingData(const char *filename);
        char *get_name(int index);
        int get_pictures(picture_cb cb, void *data);
        int add_training_face(const char *name, const cv::Mat &img);

        int nEigens, nFaces;
        cv::Mat personNumTruthMat; // 1d array mapping picture indexes to person numbers
        cv::Mat projectedTrainFaceMat; // projected training faces
        cv::Size faceSize;
        cv::PCA *pca;
    private:
        const char *dbname;
        sqlite3 *db;
        int nPersons;
        std::vector<cv::Mat> faceImages;

        int loadImagesFromDb(void);
        void doPCA(void);

        int opendb(void);
        int set_sync(void);
        int set_async(void);
        int get_picture_count(void);
        int add_person(const char *name);
        int db_add_picture(int index, const char *filename);
        int db_add_picture(const char *name, const char *filename);
        int get_person_index(const char *name);
        int create_tables(void);
        int check_table_init(void);
};
