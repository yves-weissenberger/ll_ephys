split_params = ['direction','task_nr']

fixed_params = {'port_repeat': False,
               'correct': True}
#l_ = [df[p]==v for p,v in fixed_params.items()]
#for il_ in l_:
fixed_param_conds = [all([row[p]==v for p,v in fixed_params.items()]) for ix,row in df.iterrows()]
    
uniq_param_sets = []
for param in split_params:
    uniq_param_sets.append([param,np.unique(df[param])])


for port_nr in np.unique(df['port'].values):
    
    dict_nr = 0
    for ups in uniq_param_sets:
        #fixed_param_conds = 
        for param,entry in ups:
            v = df.loc[(df['port']==port_nr) &
                        fixed_param_conds &
                       (df['direction']==direction) &
                       (df[param]==entry) ]['time'].values
            
        
        dict_nr += 1
            
    for direction in unique_directions:
        for task_nr in range(2):
            #task_nr = str(task_nr)
            v = df.loc[(df['port']==port_nr) &
                       #(df['correct']==True) & 
                       (df['direction']==direction) &
                       (df['port_repeat']==False) & 
                       (df['task_nr']==task_nr)]['time'].values
            #v = np.array(v).astype('float')
            if task_nr=='0':
                print(task_nr,len(v),str(port_nr),)
                poke_dict_t1[str(port_nr)] = [float(i) for i in v]
                poke_dict_t1['task_nr'] = str(task_nr)
                poke_dict_t1['graph_type'] = df.loc[df['task_nr']==task_nr]['graph_type'].values[0]

                poke_dict_t1['seq'] = df.loc[df['task_nr']==task_nr]['current_sequence'].values[0]

            else:
                poke_dict_t2[str(port_nr)] = [float(i) for i in v]
                poke_dict_t2['task_nr'] = str(task_nr)
                poke_dict_t2['graph_type'] = df.loc[df['task_nr']==task_nr]['graph_type'].values[0]
                poke_dict_t2['seq'] = df.loc[df['task_nr']==task_nr]['current_sequence'].values[0]

   