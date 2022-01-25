from dataclasses import dataclass, field
from datetime import datetime

# external libs
import numpy as np



@dataclass(frozen=True,order=True)
class session_ephys_dataset:
    unaligned_spike_times: np.ndarray
    spike_clusters: np.ndarray
    rsync_times_spike: np.ndarray
    cluster_quality: np.ndarray
    offset: int = field(default=0)




@dataclass(frozen=True,order=True)
class session_behaviour_dataset:
    """Stores all of the behavioural data from one session in one place
    """
    experiment_name: str
    task_name: str
    subejct_id: str
    task_nr: str =  field(repr=False)
    graph: str = field(repr=False)
    date: datetime
    event_dict: dict =  field(repr=False)
    dat_dict: dict = field(repr=False)
    events: np.ndarray =  field(repr=False)
    event_times: np.ndarray = field(repr=False)


@dataclass(frozen=True)
class session_dataset:
    behaviour_dataset: session_behaviour_dataset
    ephys_dataset: session_ephys_dataset

    
