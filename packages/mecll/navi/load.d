import std.stdio;
import std.file;
import std.string;
import std.file;
import std.algorithm;


bool string_contains(string str, string target)
{
    if (indexOf(str,target)==-1)
    {
        return false;
    }
    return true;
}

string[] find_navi_sessions(string path)
{
    string[] sessions;
    foreach (DirEntry e; dirEntries(path, SpanMode.breadth))
    {
        if (string_contains(e.name,"navi")!=-1 && indexOf(e.name,"taskFile")==-1)
        {
            // writeln(indexOf(e.name,"navi"),e.name);
            sessions ~= e.name;
        }
    }

    return sessions;
}


struct SessionMetaData{
    string subject_id;
    string experiment_name;
    string task_name;
    string start_date;

}

string remove_whitespace(string str)
{
    return str.replace(" ", "");
}


void populate_session_struct(char[] line, auto ref SessionMetaData session_data)
{
    import std.conv;
    string stringline = line.to!string;
    if (string_contains(stringline, "Experiment name")){
        session_data.experiment_name = stringline.findSplit(":")[2];
    }

    if (string_contains(stringline, "Task name")){
        session_data.task_name = stringline.findSplit(":")[2];
    }

    if (string_contains(stringline, "Subject ID")){
        session_data.subject_id = stringline.findSplit(":")[2];
    }


}

void parse_session_file(string file_path)
{
    auto file = File(file_path); // Open for reading

    SessionMetaData sess_data;  
    auto file_lines  = file.byLine();
    foreach (line; file_lines)
    {
        if (line[0]=='I')
        {
            populate_session_struct(line, &sess_data);
        }
    }

    writeln(sess_data.tupleof);


}


void main(){


    string path = "/Users/yves/team_mouse Dropbox/Yves Weissenberger/pyhomecage/data/ec_lineloop_xmas1/516495_1/16316114895508215_task/";
    auto navi_sessions = find_navi_sessions(path);
    parse_session_file(navi_sessions[0]);
}