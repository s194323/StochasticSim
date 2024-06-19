from src.Ward import Ward, initialize_wards
from src.Patient import Patient, initialize_patients
import numpy as np

def simulation_loop(patients, wards, relocation_probabilities):
    """Runs the simulation loop

    Args:
        patients (list): list of patient objects
        wards (list): list of ward objects
        relocation_probabilities (np.array): probability matrix for patient relocation

    Returns:
        (Dict): dictionary containing the performance metrics for each ward
    """
    while len(patients) > 0:
        patient = patients.pop(0)
        if patient.next_event == "Admission":
            patient.occupy_bed(wards, patients, relocation_probabilities)
        else:
            patient.discharge(wards)
    return {ward: ward.get_performance_metrics() for ward in wards}


def run_simulations(total_time,  #total
                    wards, #list of ward objects
                    relocation_probability, #probability matrix for patient relocation
                    arrival_interval_function, #function to sample arrival intervals
                    occupancy_time_function, #function to sample occupancy times
                    n_simulations = 10, #number of simulations to run
                    verbose = False #whether to print the performance metrics of each simulation
                    ):
        
    average_performance = {ward: {metric: 0 for metric in ["Occupied probability", "Estimated admissions", "Estimated rejections","Estimated relocations"]} for ward in wards}  #initialize average performance metrics
    
    for i in range(n_simulations):
        patients = initialize_patients(total_time, [ward.type for ward in wards], arrival_interval_function, occupancy_time_function) #initialize patients
        
        performance_dict = simulation_loop(patients, wards, relocation_probability)
        
        if verbose:
            print(f"Simulation {i+1} results:")
            for ward in performance_dict:
                print(f"{ward}: {performance_dict[ward]}")
                
        average_performance = {ward: {metric: average_performance[ward][metric] + performance_dict[ward][metric]/n_simulations for metric in ["Occupied probability", "Estimated admissions", "Estimated rejections","Estimated relocations"]} for ward in wards}  #update average ward specific performance metrics
        for ward in wards:
            ward.reset_metrics()
            
        #compute urgency weighted rejection penalty
        average_performance["Weighted penalty"] = np.sum(average_performance[ward]["Estimated rejections"]*ward.urgency_points for ward in wards)
    return average_performance