#include <sys/stat.h>
#include <sys/types.h>

#include <sys/stat.h>
#include <unistd.h>
#include <time.h>

bool OpData::makeDir(string path)
{
    if(mkdir(path.c_str(),0777) == -1)
        return false;
    else
        return true;
}

int OpData::getTimestamp()
{
    int timestamp = 0;

    struct tm* time;               // create a time structure

    struct stat attrib;         // create a file attribute structure

    stat("afile.txt", &attrib);     // get the attributes of afile.txt

    time = gmtime(&(attrib.st_mtime)); // Get the last modified time and put it into the time structure

    return timestamp;
}
