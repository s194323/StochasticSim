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
        self.lost = False
        self.type = type
        
        #event
        self.next_event = "Admission"
        self.next_event_time = arrival_time
        
        #get ward lookup
        self.ward_lookup = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F" : 5}
        

    def __str__(self):
        return f"Patient {self.type} with {self.next_event} at time {self.next_event_time}"
    def __repr__(self):
        return f"Patient {self.type} with {self.next_event} at time {self.next_event_time}"
    
    def occupy_bed(self, wards, patient_list, relocation_probabilities):
        """occupies a bed in the ward if there is space

        Args:
            wards (list): list of ward objects
            patient_list (list): sorted list of patient objects
            relocation_probabilities (np.array): probability matrix for patient relocation

        Returns:
            Rejected: Boolean indicating whether the patient was rejected or not
        """
        #get ward corresponding to patient type
        ward = wards[self.ward_lookup[self.type]]
        if ward.admit(self.next_event_time):
            self.next_event = "Discharge"
            self.next_event_time = self.next_event_time + self.occupancy_time
            #re-insert patient into sorted event list
            bisect.insort(patient_list, self, key=lambda x: x.next_event_time)
            return True
        else:
            self.get_rejected(wards, patient_list, relocation_probabilities)
            return False
    
    def discharge(self, wards):
        """Disconnects a patient from the ward

        Args:
            wards (list): list of ward objects
        """
        #get ward corresponding to patient type
        ward = wards[self.ward_lookup[self.type]]
        ward.discharge(self.next_event_time)
        return
    
    def get_rejected(self, wards, patient_list, relocation_probabilities):
        if self.rejected: #if patient has already been rejected then he's lost.
            self.lost = True
            return
        else: # Otherwise, try to admit him to the new ward
            self.rejected = True
            #get idx from ward the rejected ward
            idx = [i for i, ward in enumerate(wards) if ward.type == self.type][0]
            
            #Get new ward type
            new_ward = np.random.choice(wards, p=relocation_probabilities[idx])
            self.type = new_ward.type #change patient type
            #try to admit patient to new ward
            relocated = self.occupy_bed(wards, patient_list, relocation_probabilities)
            #update performance metric
            if relocated:
                wards[idx].total_relocations += 1
        return
    
def initialize_patients(total_time, patient_types, arrival_interval_function, occupancy_time_function):
    """Initializes patients based on the input parameters

    Args:
        total_time (int): total time of simulation
        patient_types (list): list of patient types (A, B, C, D, E)
        arrival_interval_function (function): function that takes type as input and returns the arrival interval
        occupancy_time_function (function): function that takes type as input and returns the occupancy time

    Returns:
        patients: list of patients sorted by arrival time
    """
    
    patients = []
    for type in patient_types:
        arrival_time = 0
        new_time = arrival_interval_function(type)
        while arrival_time + new_time < total_time:
            #generate occupancy time
            arrival_time += new_time
            occupancy_time = occupancy_time_function(type)
            #create patient object
            patient = Patient(arrival_time, occupancy_time, type)
            #insert patient into sorted list
            bisect.insort(patients, patient, key=lambda x: x.next_event_time)
            
            #generate arrival time
            new_time = arrival_interval_function(type)
    return patients

