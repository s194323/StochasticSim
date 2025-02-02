from src.Ward import Ward, initialize_wards
from src.Patient import Patient, initialize_patients, get_performance_metrics
import numpy as np

def simulation_loop(patients, wards, relocation_probabilities, burn_in_time = 15):
    """Runs the simulation loop

    Args:
        patients (list): list of patient objects
        wards (list): list of ward objects
        relocation_probabilities (np.array): probability matrix for patient relocation

    Returns:
        (Dict): dictionary containing the performance metrics for each ward
    """
    burned = False
    performance = {}
    finished_patients = []
    while len(patients) > 0:
        patient = patients.pop(0)
        #reset performance metrics first time burn in time is reached
        if patient.next_event_time >= burn_in_time and not burned:
            for ward in wards:
                ward.reset_metrics()
            burned = True

        if patient.next_event == "Admission":
            patient.occupy_bed(wards, patients, relocation_probabilities)
            finished_patients.append(patient)
        else:
            patient.discharge(wards)
    for ward in wards:
        performance[ward] = ward.get_performance_metrics()
    
    performance["Patients"] = get_performance_metrics(finished_patients)
    return performance


def run_simulations(total_time,  #total
                    wards, #list of ward objects
                    relocation_probability, #probability matrix for patient relocation
                    arrival_interval_function, #function to sample arrival intervals
                    occupancy_time_function, #function to sample occupancy times
                    n_simulations = 10, #number of simulations to run
                    verbose = False, #whether to print the performance metrics of each simulation
                    burn_in_time = 15 #burn in time for the simulation
                    ):
        
    average_performance = {ward: {metric: 0 for metric in ward.get_performance_metrics().keys()} for ward in wards}
    patient_performance = {ward.type : {"Admitted": 0, "Rejected": 0, "Lost": 0, "Relocated": 0} for ward in wards}

    for i in range(n_simulations):
        for ward in wards:
            ward.reset_metrics()
        patients = initialize_patients(total_time+burn_in_time, [ward.type for ward in wards], arrival_interval_function, occupancy_time_function) #initialize patients
        
        performance_dict = simulation_loop(patients, wards, relocation_probability, burn_in_time = burn_in_time) #run simulation loop
        
        if verbose:
            print(f"Simulation {i+1} results:")
            for ward in performance_dict:
                print(f"{ward}: {performance_dict[ward]}")
                
        average_performance = {ward: {metric: average_performance[ward][metric] + performance_dict[ward][metric]/n_simulations for metric in performance_dict[ward].keys()} for ward in wards}

        patient_performance = {ward.type: {metric: patient_performance[ward.type][metric] + performance_dict["Patients"][ward.type][metric]/n_simulations for metric in performance_dict["Patients"][ward.type].keys()} for ward in wards}

    #compute urgency weighted rejection penalty
    average_performance["Weighted penalty"] = np.sum(average_performance[ward]["Estimated rejections"]*ward.urgency_points for ward in wards)
    average_performance["Patients"] = patient_performance
    return average_performance

def compute_gradient(total_time,  #total
                    wards, #list of ward objects
                    relocation_probability, #probability matrix for patient relocation
                    arrival_interval_function, #function to sample arrival intervals
                    occupancy_time_function, #function to sample occupancy times
                    n_simulations = 10, #number of simulations to run
                    verbose = False #whether to print the performance metrics of each simulation
                    ):
    for ward in wards:
        ward.capacity += 1
        performance_upper = run_simulations(total_time, wards, relocation_probability, arrival_interval_function, occupancy_time_function, n_simulations = n_simulations, verbose = verbose)
        ward.capacity -= 2
        performance_lower = run_simulations(total_time, wards, relocation_probability, arrival_interval_function, occupancy_time_function, n_simulations = n_simulations, verbose = verbose)
        ward.capacity += 1
        gradient = (performance_upper["Weighted penalty"] - performance_lower["Weighted penalty"])/2
        ward.gradient = gradient
    return np.array([ward.gradient for ward in wards])