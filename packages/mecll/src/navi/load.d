import std.stdio;
import std.file;
import std.string;
import std.file;
import std.algorithm;
import std.conv;


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

    void print() 
    {
        writeln("Subject ID:", subject_id);
        writeln("Experiment Name:", experiment_name);
        writeln("Task Name:", task_name);
        writeln("Start Date:", start_date);
    }

}

string remove_whitespace(string str)
{
    return str.replace(" ", "");
}


void populate_session_struct(char[] line, ref SessionMetaData session_data)
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

    if (string_contains(stringline, "Start date")){
        session_data.start_date = stringline.findSplit(":")[2];
    }

}


int[string] parse_state_or_event_map(char[] str)
{   
    // writeln(str);
    int[string] state_map;
    foreach (pair; str.split(','))
    {
        auto split_string = pair.findSplit(":");
        string event = to!string(strip(split_string[0],"ES{ '}"));

        int num = to!int(strip(split_string[2],"ES{ '}"));
        state_map[event] = num;
    }
    return state_map;
}

void parse_session_file(string file_path)
{
    auto file = File(file_path); // Open for reading

    SessionMetaData sess_data;  
    auto file_lines  = file.byLine();

    int[string] state_map;
    int[string] event_map;
    foreach (line; file_lines)
    {
        
        if (line.length != 0){

            if (line[0]=='S'){
                state_map = parse_state_or_event_map(line);
            }
            if (line[0]=='E')
            {
                event_map = parse_state_or_event_map(line);
            }
            if (line[0]=='I')
            {
                populate_session_struct(line, sess_data);
            }
        }
    }

    sess_data.print();


}


void main(){


    string path = "/Users/yves/team_mouse Dropbox/Yves Weissenberger/pyhomecage/data/ec_lineloop_xmas1/516495_1/16316114895508215_task/";
    auto navi_sessions = find_navi_sessions(path);
    parse_session_file(navi_sessions[0]);
}