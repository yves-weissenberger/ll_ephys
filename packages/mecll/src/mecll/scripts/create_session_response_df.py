import os
import numpy as np


from mecll.load import load_data
from mecll.process_data.proc_beh import build_poke_df
from mecll.process_data.proc_neural import get_activity_for_each_poke_inpoke_to_outpoke


def main(save_path: str) -> None:

    session_directory = '/Users/yves/team_mouse Dropbox/MEC_data/spike_sorted/'
    sessions = [os.path.join(session_directory, i) for i in os.listdir(session_directory) if 'ks25' in i]
    for session in sessions:
        try:
            session_name = os.path.split(session)[1]
            out = load_data(session,align_to='task')
            spkT, spkC, single_units, events, lines, aligner = out
            df = build_poke_df(lines, events)
            poke_array = get_activity_for_each_poke_inpoke_to_outpoke(df, 
                                                                    spkT, spkC,
                                                                    single_units, 
                                                                    aligner
                                                                    )
            save_dir = os.path.join(save_path,session_name)
            if not os.path.isdir(save_dir):
                os.mkdir(save_dir)
            neural_save_path = os.path.join(save_dir, 'neural_response_table.npy')
            beh_save_path = os.path.join(save_dir, 'task_event_table.csv')
            np.save(neural_save_path, poke_array)
            df.to_csv(beh_save_path)
        except:
          print("failed to process {}".format(session))





if __name__ == "__main__":

    results_dir = "/Users/Yves/Documents/spike"
    if not os.path.isdir(results_dir):
        os.mkdir(results_dir)
    main(results_dir)
    print("done!")