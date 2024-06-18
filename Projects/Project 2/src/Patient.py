import bisect
import numpy as np
class Patient:
    def __init__(self, arrival_time, occupancy_time, type):
        """
        Args:
            arrival_time (float): Time at which the patient arrives
            occupancy_time (float): Time at which the patient occupies a bed
            type (Char): Patient type (A, B, C, D, E)
        """
        self.occupancy_time = occupancy_time
        self.rejected = False
        self.type = type
        
        #event
        self.next_event = "Admission"
        self.next_event_time = arrival_time
        

    def __str__(self):
        return f"Patient {self.type} with {self.next_event} at time {self.next_event_time}"
    def __repr__(self):
        return f"Patient {self.type} with {self.next_event} at time {self.next_event_time}"
    
    def occupy_bed(self, wards, event_list):
        ward = [ward for ward in wards if ward.type == self.type][0]
        if ward.admit():
            self.next_event = "Discharge"
            self.next_event_time = self.next_event_time + self.occupancy_time
            #insert patient into sorted event list
            bisect.insort(event_list, self, key=lambda x: x.next_event_time)
        else:
            self.get_rejected(wards)
        return
        
    def get_rejected(self, wards):
        self.rejected = True
        #TODO
        return
    
def initialize_patients(num_patients, arrival_interval_function, occupancy_time_function):
    """
    Initializes the patients
    
    Args:
        num_patients (Dict): Number of patients of each type. Example, {'A' : 10, 'B' : 20, 'C' : 30, 'D' : 40, 'E' : 50}
        arrival_interval_function (function): Function that generates the arrival interval between patients
        occupancy_time_function (function): Function that generates the occupancy time of a patient
    Returns:
        list: List of Patient sorted by arrival time
    """
    patients = []
    for type, num in num_patients.items():
        arrival_intervals = arrival_interval_function(type, num)
        arrival_times = np.cumsum(arrival_intervals)
        occupancy_times = occupancy_time_function(type, num)
        patients = patients + [Patient(arrival_time, occupancy_time, type) for arrival_time, occupancy_time in zip(arrival_times, occupancy_times)]
    #sort patients by event time
    patients = sorted(patients, key=lambda x: x.next_event_time)
    return patients