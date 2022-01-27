# standard library
from dataclasses import dataclass, field
from typing import List, Tuple
import re
from datetime import datetime

from matplotlib.pyplot import get

#local modules
from .datasets import session_behaviour_dataset

# other libraries
import numpy as np





def load_behavioural_data(fpath: str) -> session_behaviour_dataset:


    lines = open(fpath,'r').readlines()
    out = get_metadata(lines)

    experiment_name, task_name, subject_id, task_nr, graph,lineloop,date,test,summary_dict = out
    dat_dict,events,event_times,nRews,event_dict = parse_data(lines,experiment_name)
    date = datetime.fromisoformat(date.replace(' ','T').replace('/','-'))
    task_times = get_task_ranges(lines,event_times[-1])

    rsync_times = get_rsync_times_behaviour(lines)

    dset = session_behaviour_dataset(experiment_name=experiment_name,
                                    task_name=task_name,
                                    subject_id=subject_id,
                                    task_nr=task_nr,
                                    line_loop=lineloop,
                                    graph=graph,
                                    date=date,
                                    event_dict=event_dict,
                                    summary_dict=summary_dict,
                                    dat_dict=dat_dict,
                                    events=events,
                                    event_times=event_times,
                                    task_times=task_times)
    return dset


def get_rsync_times_behaviour(lines: List[str]) -> np.ndarray:

    rsync_times_pyc = np.array([int(re.findall(' ([0-9]*) ' ,i)[0]) for i in lines if '32\n' in i[-4:]])*30

    return rsync_times_pyc

def get_task_ranges(lines: List[str],session_end_time: int):
    """ Return task ranges in ms in the reference frame of 
        the behaviour system
    """

    task_switches = [re.findall(' ([0-9]*)', i)[0] for i in lines if 'change_task' in i and i.startswith('V')]
    task_boundU = [1+int(re.findall(' ([0-9]*) ',lines[ix+1])[0]) for ix,l_ in enumerate(lines) if "OLD" in l_] + [1000*session_end_time]
    task_boundL = [0] + [int(re.findall(' ([0-9]*) ',lines[ix+1])[0]) for ix,l_ in enumerate(lines) if "NEW" in l_]
    # this detects when it changes from an old task
    task_order = [int(re.findall('([0-9])\n',lines[ix+3])[0]) for ix,l_ in enumerate(lines) if "OLD" in l_]

    # this is a hack to get the first task as task_order reports
    # only what new task
    if task_order[0]==2:
        task_order = [1] + task_order
    elif task_order[1]==2:
        task_order = [2] + task_order


    cTask = task_order[0]
    task1_times = []
    task2_times = []

    #print(task_boundL,task_boundU)
    for tl_,tu_ in zip(task_boundL,task_boundU):

        if cTask==1:
            task1_times.append([tl_,tu_])
        elif cTask==2:
            task2_times.append([tl_,tu_])
        if cTask==1: cTask=2
        elif cTask==2: cTask=1
    task_times = [task1_times,task2_times]
    return task_times


def get_metadata(lines: List[str]) -> str:
    """ Get metadata from the beginning of the file """
    
    summary_lines = []
    experiment_name = task_name = subject_id = task_nr = graph = lineloop = date = test = None
    for l in lines:
        if re.findall('I Experiment name  : (.*?)\n',l): experiment_name = re.findall('I Experiment name  : (.*?)\n',l)[0]
        
        if re.findall('Task name : (.*?)\n',l): task_name = re.findall('Task name : (.*?)\n',l)[0]
        
        if re.findall('Subject ID : (.*?)\n',l): subject_id = eval(re.findall('Subject ID : (.*?)\n',l)[0])
            
        if re.findall('Start date : (.*?)\n',l): date = re.findall('Start date : (.*?)\n',l)[0]
        
        #################################################################################
        #### Specific to this task
        #################################################################################

        if re.findall('V.*? task_nr (.*?)\n',l): task_nr = re.findall('V.*? task_nr (.*?)\n',l)[0]
            
        if re.findall('P.*? (G[0-9]_[0-9])\n',l): graph = re.findall('P.*? (G[0-9]_[0-9])\n',l)[0]

        if re.findall('P .* (LOOP|LINE|loop|line)\n',l): lineloop = re.findall('P .* (LOOP|LINE|loop|line)\n',l)[0].lower()
        if re.findall('TEST ([A-z]*)\\n',l): test = eval(re.findall('TEST ([A-z]*)\\n',l)[0])

        if re.findall('V -1 *',l):
            summary_lines.append(l)

    summary_dict = _get_summary_dict(summary_lines)
    return experiment_name, task_name, subject_id, task_nr, graph, lineloop, date, test, summary_dict


def _get_summary_dict(summary: List[str]) -> dict:
    """Get summary of performance in the session

    Args:
        summary (List[str]): [description]

    Returns:
        dict: [description]
    """
    summary_dict = {}
    for i in summary:
        try:

            k =re.findall('-1 (.*?) ',i)[0]
            if '\n' in i:
                summary_dict[k] = eval(re.findall('-1 .*? (.*?)\n',i)[0]) if (k!='subject_id' and k!='graph_type') else re.findall('-1 .*? (.*?)\n',i)[0]
            else:
                summary_dict[k] = eval(re.findall('-1 .*? (.*$)',i)[0]) if (k!='subject_id' and k!='graph_type') else re.findall('-1 .*? (.*$)',i)[0]
        except Exception as e:
            print('exception in _get_summary_dict')
            print(e)
            print(i)
    return summary_dict



def _parse_dat(text: str):
    """ function that takes data in and returns meaningful stuff """

    if 'POKEDPORT' in text:
        now = int(re.findall('POKEDPORT_([0-9]{1,2})',text)[0])
        avail = eval(re.findall('_NOWPOKES_(\[.*?\])_',text)[0])[0]
        prev = eval(re.findall('_PREVPOKE[S]?_([\[]?.*?[\]]?)_',text)[0])

        dtype = 'port'
    elif 'NOWSTATE' in text:
        now = int(re.findall('_NOWSTATE_([0-9]{1,2})',text)[0])
        avail = eval(re.findall('_AVAILSTATES_(\[.*?\])',text)[0])
        prev = eval(re.findall('_PREVSTATE[S]?_([\[]?.*?[\]]?)_',text)[0])
        dtype = 'state'
    else:
        print("WARNING the following line was not processed")
        print(text)

    if 'PROBE' in text:
        probe = eval(re.findall('PROBE_([A-z]*?)_',text)[0])
    else:
        probe = None

    if type(prev)==list:
        if now in prev: 
            teleport=False
        else:
            teleport = True
    else:
        teleport = False

    return now,avail,dtype,teleport,probe



def parse_data(lines: List[str], experiment_name: str) -> Tuple[dict,np.ndarray,np.ndarray,int,dict]:
    """Parse a single file of mouse behavioural data to extract events

    Args:
        lines (List[str]): list of lines from the text file
        experiment_name (str): name the experiement to direct parsing

    Returns:
        Tuple[dict,np.ndarray,np.ndarray,int,dict]: [description]
    """
    start_read = False  # placeholder variable that ignores the period where just free rewards are available
    event_times = []
    events = []
    alldat = []
    dat_times = []
    dat_dict = {'state': [],
                'port': [],
                'random': [],
                'rew_locations': [],
                'rews': [],
                'rew_list': [] }


    tot_pokes = 0
    nRews = 0

    text2 = ''.join(lines)
    state_dict0 = eval(re.findall('({.*)\n',text2)[0])
    state_dict = dict([(v,k) for k,v in state_dict0.items()])
    event_dict = eval(re.findall('({.*)\n',text2)[1])
    event_dict = dict([(v,k) for k,v in event_dict.items()])


    ##NEW CODE##
    event_dict = {**state_dict,**event_dict}
    ##END NEW CODE ##
    #print(event_dict.keys())
    rew_list = []
    for ln,l in enumerate(lines):
        try:

            if (str(state_dict0['handle_poke'])+'\n' in l and  l[0]=='D'):
                start_read = True

            if l[0]=='D':
                if start_read:
                    tmp = float(re.findall('D ([-0-9]*)',l)[0])/1000.
                    ev = int(re.findall('D [-0-9]* ([0-9]{1,3})',l)[0])
                    #print(ev)
                    #print(event_dict[ev])
                    if ev in list(event_dict.keys()):
                        #print(event_dict[ev],ev)
                        events.append(event_dict[ev])
                        event_times.append(tmp)

                        if event_dict[ev][-1] in [str(i) for i in range(9)]:
                            tot_pokes += 1
                    #event_dict.keys()

            elif l[0]=='P':
                if 'POKEDSTATE' in l:
                    start_read = True
                

                if start_read:
                    tmp_t = float(re.findall('P ([0-9]*)',l)[0])/1000.
                    dat = re.findall('P [-0-9]* (.*)\n',l)[0]
                    if 'POKE' in dat and 'TELEPORT' not in dat:

                        now,avail,dtype,teleport,probe = _parse_dat(dat)

                        if dtype=='port':
                            if '_REW_True' in l:
                                dat_dict['rew_list'].append(1)
                            else:
                                dat_dict['rew_list'].append(0)


                        #tmp = re.findall("RANDOM_([A-z]*?)_",dat)
                        #if tmp: dat_dict['random'].append(eval(tmp[0]))
                        if 'POKEDSTATE' in dat:
                            dat_dict['random'].append(teleport)

                        #if teleport==False:
                        dat_dict[dtype].append([now,avail,tmp_t,probe])
                    elif 'REWARD LOCATIONS' in l:
                        #print(l)
                        tmp_t = float(re.findall('P ([0-9]*)',l)[0])/1000.
                        if 'NAVI' in experiment_name:
                            tmp = eval(re.findall('LOCATIONS(.*)',l)[0])
                        else:
                            tmp = eval(re.findall('LOCATIONS(\[.*\])',l)[0])
                        dat_dict['rew_locations'].append([tmp,tmp_t])
                    if '_REW_True' in l:
                        nRews += 1
                        dat_dict['rews'].append([now,tmp_t,probe])
                        #rew_list.append()


        except Exception as e:
            print(l)
            print(ln)
            print(e)
        #    raise Exception
    #print(event_dict.keys())
    return dat_dict, np.array(events), np.array(event_times), nRews, event_dict


