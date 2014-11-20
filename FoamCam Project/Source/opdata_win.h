#include <windows.h>

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

int OpData::getTimestamp() //UNDER CONSTRUCTION http://stackoverflow.com/questions/1938939/get-file-last-modify-time-and-compare
{
    int timestamp = 0;

    FILETIME creationTime,
         lpLastAccessTime,
         lastWriteTime;

    bool err = GetFileTime( h, &creationTime, &lpLastAccessTime, &lastWriteTime );
    if( !err )
    {
        //error code
    }


    return timestamp;
}
