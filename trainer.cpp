#include "trainer.h"

#define ERROR_CHECK(x, err) if (x != SQLITE_OK) { \
    fputs(err, stderr); \
    sqlite3_free(err); \
    return ret; \
}

#define RET_CHECK(ret) if (ret != SQLITE_OK) { \
    fputs(sqlite3_errmsg(db), stderr); \
    return ret; \
}

#define RET_CHECK_NULL(ret) if (ret != SQLITE_OK) { \
    fputs(sqlite3_errmsg(db), stderr); \
    exit(1); \
    return NULL; \
}

Trainer::Trainer(const char  *dbfile) : dbname(dbfile)
{
    pca = NULL;
    int ret = opendb();
    if (ret != 0) {
        throw(ret);
    }
}

Trainer::~Trainer()
{

    if (db)
        sqlite3_close(db);
    if (pca)
        delete pca;
}

// Train from the data in the given text file, and store the trained data into the file
int Trainer::learn(void)
{
    int i, ret;

    // load training data
    ret = loadImagesFromDb();
    if (ret) {
        printf("Failed to load images from database\n");
        return -1;
    }
    printf("Got %d training images.\n", nFaces);
    if(nFaces < 2) { fprintf(stderr,
                "Need 2 or more training faces\n"
                "Database contains only %d\n", nFaces);
        return -1;
    }

    // do PCA on the training faces
    doPCA();

    // project the training images onto the PCA subspace
    projectedTrainFaceMat.create(nFaces, nEigens, CV_32FC1);
    printf("Projecting %d training faces\n", nFaces);
    for(i=0; i<nFaces; i++) {
        cv::Mat row = projectedTrainFaceMat.row(i);
        pca->project(faceImages[i].reshape(0, 1)).copyTo(row);
    }
    return 0;
}

// Read the names & image filenames of people from a text file, and load all those images listed.
int Trainer::loadDbFromList(const char *filename) {
    FILE * imgListFile = 0;
    int count = 0;

    set_async();

    printf("Loading the training images in '%s'\n", filename);
    // open the input file
    if(!(imgListFile = fopen(filename, "r"))) {
        fprintf(stderr, "Can\'t open file %s\n", filename);
        return -1;
    }

    // store the face images in an array
    while(1) {
        char linebuf[256];
        char *personName, *imgFilename;

        // read person number (beginning with 1), their name and the image filename.
        if (fgets(linebuf, sizeof(linebuf), imgListFile) == NULL)
            break;
        personName = linebuf;
        strsep(&personName, ",");
        imgFilename = personName;
        strsep(&imgFilename, ",");
        int len = strlen(imgFilename);
        if (imgFilename[len-1] == '\n')
            imgFilename[len-1] = '\0';
        if (personName && imgFilename) {
            db_add_picture(personName, imgFilename);
            count++;
        }
    }
    set_sync();

    fclose(imgListFile);
    printf("Loaded %d filenames from %s\n", count, filename);
    return count;
}

// Read the names & image filenames of people from a text file, and load all those images listed.
int Trainer::loadImagesFromDb(void)
{
    sqlite3_stmt *pstmt;
    const char *filename;
    int personNumber, i=0;

    faceImages.clear();
    // count the number of faces
    nFaces = get_picture_count();
    // allocate the person number matrix
    personNumTruthMat.create(1, nFaces, CV_16UC1);

    int ret = sqlite3_prepare_v2(db, "SELECT pid,path FROM pictures;",
                                 -1, &pstmt, NULL);
    RET_CHECK_NULL(ret);
    while (sqlite3_step(pstmt) == SQLITE_ROW && i < nFaces) {
        personNumber = sqlite3_column_int(pstmt, 0);
        filename = (const char *) sqlite3_column_text(pstmt, 1);

        personNumTruthMat.at<uint16_t>(i) = personNumber;

        // load the face image
        faceImages.push_back(cv::imread(filename, CV_LOAD_IMAGE_GRAYSCALE));
        if(!faceImages[i].data) {
            fprintf(stderr, "Can\'t load image from '%s'\n", filename);
            return -1;
        }
        i++;
    }
    return 0;
}

// Do the Principal Component Analysis, finding the average image
// and the eigenfaces that represent any image in the given dataset.
void Trainer::doPCA(void)
{
    int i;

    cv::Mat data;

    // Save the face size
    faceSize = faceImages[0].size();

    // set the number of eigenvalues to use
    nEigens = nFaces-1;

    // allocate the eigenvetor images
    data.create(nFaces, faceSize.area(), CV_32FC1);
    for (i = 0; i < nFaces; i++) {
        cv::Mat row = data.row(i);
        faceImages[i].reshape(0, 1).convertTo(row, CV_32FC1);
    }

    // compute average image, eigenvalues, and eigenvectors
    printf("Calculating eigenvectors for %d images\n", nFaces);
    pca = new cv::PCA(data, cv::Mat(), CV_PCA_DATA_AS_ROW, nEigens);

    //printf("Normalizing eigen values XXX not implemented\n");
    //cvNormalize(eigenValMat, eigenValMat, 1, 0, CV_L1, 0);
}

// Save all the eigenvectors as images, so that they can be checked.
void Trainer::storeEigenfaceImages(void)
{
    // Store the average image to a file
    printf("Saving the image of the average face as 'out_averageImage.bmp'.\n");
    cv::imwrite("out_averageImage.bmp", pca->mean.reshape(0, faceSize.height));
    // Create a large image made of many eigenface images.
    // Must also convert each eigenface image to a normal 8-bit UCHAR image instead of a 32-bit float image.
    printf("Saving the %d eigenvector images as 'out_eigenfaces.bmp'\n", nEigens);
    if (nEigens > 0) {
        // Put all the eigenfaces next to each other.
        int i;
        int COLUMNS = 8;	// Put upto 8 images on a row.
        int nCols = min(nEigens, COLUMNS);
        int nRows = 1 + (nEigens / COLUMNS);	// Put the rest on new rows.
        cv::Size size(nCols * faceSize.width, nRows * faceSize.height);

        cv::Mat bigImg(size, CV_8UC1);
        bigImg = cv::Scalar(255);
        for (i=0; i<nEigens; i++) {
            // Get the eigenface image.
            cv::Mat norm;
            cv::Mat im = pca->eigenvectors.row(i);
            im = im.reshape(0, faceSize.height);
            normalize(im, norm, 0, 255, cv::NORM_MINMAX);
            // Paste it into the correct position.
            int x = faceSize.width * (i % COLUMNS);
            int y = faceSize.height * (i / COLUMNS);
            cv::Mat roi(bigImg, cv::Rect(x, y, faceSize.width, faceSize.height));
            norm.convertTo(roi,CV_8UC1);
        }
        cv::imwrite("out_eigenfaces.bmp", bigImg);
    }
}

// Open the training data from the file
int Trainer::loadTrainingData(const char *filename)
{
    cv::FileStorage fs;

    // create a file-storage interface
    if (!fs.open(filename, cv::FileStorage::READ)) {
        printf("Can't open training xml file '%s'.\n", filename);
        return -1;
    }

    pca = new cv::PCA();
    fs["eigenvals"] >> pca->eigenvalues;
    fs["eigenvects"] >> pca->eigenvectors;
    fs["mean"] >> pca->mean;
    fs["projectedTrainFaceMat"] >> projectedTrainFaceMat;
    fs["personNumTruthMat"] >> personNumTruthMat;
    fs["nEigens"] >> nEigens;
    fs["nFaces"] >> nFaces;
    fs["faceSizeW"] >> faceSize.width;
    fs["faceSizeH"] >> faceSize.height;
    //for(int i=0; i<nEigens; i++) {
    //    char varname[200];
    //    sprintf( varname, "eigenVect_%d", i );
    //    eigenVectArr[i] = (IplImage *)cvReadByName(fileStorage, 0, varname, 0);
    //}

    // release the file-storage interface
    fs.release();

    printf("Training data loaded (%d training images):\n", nFaces);
    return 0;
}

// Save the training data to the file
int Trainer::storeTrainingData(const char *filename)
{
    cv::FileStorage fs;

    // create a file-storage interface
    if (!fs.open(filename, cv::FileStorage::WRITE)) {
        printf("Can't open training xml file '%s'.\n", filename);
        return -1;
    }

    fs << "eigenvals" << pca->eigenvalues;
    fs << "eigenvects" << pca->eigenvectors;
    fs << "mean" << pca->mean;
    fs << "projectedTrainFaceMat" << projectedTrainFaceMat;
    fs << "personNumTruthMat" << personNumTruthMat;
    fs << "nEigens" << nEigens;
    fs << "nFaces" << nFaces;
    fs << "faceSizeW" << faceSize.width;
    fs << "faceSizeH" << faceSize.height;
    //for(int i=0; i<nEigens; i++) {
        //char varname[200];
        //sprintf( varname, "eigenVect_%d", i );
        //cvWrite(fileStorage, varname, eigenVectArr[i], cvAttrList(0,0));
    //}

    // release the file-storage interface
    fs.release();
    return 0;
}

// Returns the name corresponding to the person index.  String must be freed.
char *Trainer::get_name(int index)
{
    sqlite3_stmt *pstmt;
    char *name;
    int ret = sqlite3_prepare_v2(db, "SELECT name from names WHERE id = ?;",
                                 -1, &pstmt, NULL);
    RET_CHECK_NULL(ret);
    ret = sqlite3_bind_int(pstmt, 1, index);
    RET_CHECK_NULL(ret);
    if (sqlite3_step(pstmt) == SQLITE_ROW) {
        name = strdup((const char *)sqlite3_column_text(pstmt, 0));
    } else {
        return NULL;
    }
    ret = sqlite3_finalize(pstmt);
    RET_CHECK_NULL(ret);
    return name;
}

int Trainer::get_pictures(picture_cb cb, void *data)
{
    sqlite3_stmt *pstmt;
    int ret = sqlite3_prepare_v2(db, "SELECT pid, path FROM pictures;",
                                 -1, &pstmt, NULL);
    RET_CHECK(ret);
    const char *filename;
    int index;
    while (sqlite3_step(pstmt) == SQLITE_ROW) {
        index = sqlite3_column_int(pstmt, 0);
        filename = (const char *)sqlite3_column_text(pstmt, 1);
        if (cb) {
            ret = (*cb)(index, filename, data);
            if (ret)
                return ret;
        }
    }
    return 0;
}

int Trainer::add_training_face(const char *name, const cv::Mat &img)
{
    const char *base = "data/";
    const char *ext = ".bmp";
    char *filename = (char *) malloc(strlen(base) + strlen(name) + strlen(ext) + 1);
    int ret;

    if (!filename)
        return -1;
    strcat(filename, base);
    strcat(filename + strlen(filename), name);
    strcat(filename + strlen(filename), ext);

    cv::imwrite(filename, img);
    ret = db_add_picture(name, filename);
    if (ret) return ret;
    return 0;
}

const char *const namestable = "CREATE TABLE names(id INTEGER, name, PRIMARY KEY(id ASC));";
const char *const picstable = "CREATE TABLE pictures(pid REFERENCES names(id) ON DELETE CASCADE, path);";

int Trainer::opendb(void)
{
   int ret = sqlite3_open(dbname, &db);
   RET_CHECK(ret);
   if (check_table_init() == -1)
       ret = create_tables();
   return ret;
}

int Trainer::set_sync(void)
{
    int ret;
    char *err;
    ret = sqlite3_exec(db, "PRAGMA synchronous = 1;", NULL, NULL, &err);
    ERROR_CHECK(ret, err);
    return ret;
}

int Trainer::set_async(void)
{
    int ret;
    char *err;
    ret = sqlite3_exec(db, "PRAGMA synchronous = 0;", NULL, NULL, &err);
    ERROR_CHECK(ret, err);
    return ret;
}

int Trainer::add_person(const char *name)
{
    sqlite3_stmt *pstmt;
    int ret = sqlite3_prepare_v2(db, "INSERT INTO names VALUES (NULL, ?);",
                                 -1, &pstmt, NULL);
    RET_CHECK(ret);
    ret = sqlite3_bind_text(pstmt, 1, name, -1, NULL);
    RET_CHECK(ret);
    ret = sqlite3_step(pstmt);
    if (ret != SQLITE_DONE) return ret;
    ret = sqlite3_finalize(pstmt);
    RET_CHECK(ret);
    return 0;
}

int Trainer::db_add_picture(int index, const char *filename)
{
    sqlite3_stmt *pstmt;
    int ret = sqlite3_prepare_v2(db, "INSERT INTO pictures VALUES (?, ?);",
                                 -1, &pstmt, NULL);
    RET_CHECK(ret);
    ret = sqlite3_bind_int(pstmt, 1, index);
    RET_CHECK(ret);
    ret = sqlite3_bind_text(pstmt, 2, filename, -1, NULL);
    RET_CHECK(ret);
    ret = sqlite3_step(pstmt);
    if (ret != SQLITE_DONE) return ret;
    ret = sqlite3_finalize(pstmt);
    RET_CHECK(ret);
    return 0;
}

int Trainer::db_add_picture(const char *name, const char *filename)
{
    int index = get_person_index(name);
    if (index < 0) {
        add_person(name);
        index = get_person_index(name);
    }
    if (index > 0)
        return db_add_picture(index, filename);
    return index;
}

int Trainer::get_person_index(const char *name)
{
    sqlite3_stmt *pstmt;
    int id = -1;
    int ret = sqlite3_prepare_v2(db, "SELECT id FROM names WHERE name = ?;",
                                 -1, &pstmt, NULL);
    RET_CHECK(ret);
    ret = sqlite3_bind_text(pstmt, 1, name, -1, NULL);
    RET_CHECK(ret);
    if (sqlite3_step(pstmt) == SQLITE_ROW) {
        id = sqlite3_column_int(pstmt, 0);
    } else {
        return -1;
    }
    ret = sqlite3_finalize(pstmt);
    RET_CHECK(ret);
    return id;
}

int Trainer::get_picture_count(void)
{
    sqlite3_stmt *pstmt;
    int ret = sqlite3_prepare_v2(db, "SELECT COUNT(*) FROM pictures;",
                                 -1, &pstmt, NULL);
    int count;
    RET_CHECK(ret);
    if (sqlite3_step(pstmt) == SQLITE_ROW) {
        count = sqlite3_column_int(pstmt, 0);
        return count;
    } else {
        return -1;
    }
}

int Trainer::create_tables(void)
{
    char *err;
    puts("Creating database tables");
    int ret = sqlite3_exec(db, namestable, NULL, NULL, &err);
    ERROR_CHECK(ret, err);
    ret = sqlite3_exec(db, picstable, NULL, NULL, &err);
    ERROR_CHECK(ret, err);
    return 0;
}

int Trainer::check_table_init(void)
{
    sqlite3_stmt *pstmt;
    int found = 0;
    int ret = sqlite3_prepare_v2(db, "SELECT name FROM sqlite_master where type = 'table';",
                                 -1, &pstmt, NULL);
    RET_CHECK(ret);
    while (sqlite3_step(pstmt) == SQLITE_ROW) {
        const unsigned char *table = sqlite3_column_text(pstmt, 0);
        if (strcmp((const char *)table, "names") == 0 || strcmp((const char *)table, "pictures") == 0)
            found += 1;
    }
    if (found == 2) {
        printf("Database is initialized\n");
        return 0;
    } else {
        printf("Database is not initialized\n");
        return -1;
    }
}
