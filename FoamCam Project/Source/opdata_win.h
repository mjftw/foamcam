/*
    File structure operations for windows machines.
*/

#include <windows.h>
#include <time.h>

bool OpData::makeDir(string path)
{
    vector<string> path_parts;
    string temp_path = path;

    string::size_type slash_pos = temp_path.rfind('/');

    while(slash_pos != string::npos)
    {
        path_parts.push_back(temp_path);
        temp_path.erase(slash_pos, temp_path.length());
        slash_pos = temp_path.rfind('/');
    }
    if(temp_path.rfind(':') == string::npos)
        path_parts.push_back(temp_path);

    while(!path_parts.empty())
    {
        if (!(CreateDirectory(path_parts[path_parts.size()-1].c_str(), NULL) || ERROR_ALREADY_EXISTS == GetLastError()))
        {
            cerr << "Could not create directory: " << path_parts[path_parts.size()-1] << endl;
            return false;
        }

        path_parts.pop_back();
    }
    return true;
}

string OpData::getTimestamp()
{
    stringstream timestamp;

    WIN32_FILE_ATTRIBUTE_DATA wfad;
    SYSTEMTIME st;
    struct tm temp;

    GetFileAttributesEx(src_img_path->c_str(), GetFileExInfoStandard, &wfad);
    FileTimeToSystemTime(&wfad.ftLastWriteTime, &st);

    temp.tm_sec = st.wSecond;
	temp.tm_min = st.wMinute;
	temp.tm_hour = st.wHour;
	temp.tm_mday = st.wDay;
	temp.tm_mon = st.wMonth - 1;
	temp.tm_year = st.wYear - 1900;
	temp.tm_isdst = -1;

    time_t time = mktime(&temp);

    timestamp << time;
    if(st.wMilliseconds < 10)
        timestamp << ".00";
    else if(st.wMilliseconds < 100)
        timestamp << ".0";
    else
        timestamp << ".";

    timestamp << st.wMilliseconds;
    return timestamp.str();
}
