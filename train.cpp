#include <unistd.h>

#include "trainer.h"

int main(void)
{
    unlink("faces.db");
    Trainer t("faces.db");
    t.loadDbFromList("faces.txt");
    t.learn();
    t.storeTrainingData("facedata.xml");
    t.storeEigenfaceImages();
}
