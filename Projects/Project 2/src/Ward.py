class Ward:
    def __init__(self, type, capacity, urgency_points, real_time_tracking = False):
        """
        Args:
            type (Char): Ward type (A, B, C, D, E)
            capacity (int): Maximum number of patients that can be admitted to the ward
            urgency_points (int): Urgency points associated with the ward
        """
        self.type = type
        self.capacity = capacity
        self.current_occupancy = 0
        self.urgency_points = urgency_points
        self.real_time_tracking = real_time_tracking
        
        #Performance Metrics
        self.total_arrivals = 0 #Total number of patients that have arrived at the ward
        self.total_admissions = 0 #Total number of patients that have been admitted to the ward
        self.total_rejections = 0 #Total number of patients that have been rejected from the ward
        self.total_lost = 0 #Total number of patients that have been lost
        self.total_relocations = 0 #Total number of patients that have been successfully relocated to another ward
        
        if self.real_time_tracking:
            self.occupancy = []
            self.rejections = []
                    
    @classmethod
    def from_dataframe(cls, df, type, real_time_tracking = False):
        """
        Initialization method for the Ward object that takes in a dataframe and a ward type and returns a Ward object with the corresponding data from the dataframe.
        
        Args:
            df (pandas.dataframe): _description_
            type (Char): Ward type (A, B, C, D, E)
        Returns:
            Ward: Ward object with the corresponding data from the dataframe
        """
        type = type
        capacity = df["Bed Capacity"][type]
        urgency_points = df["Urgency Points"][type]
        return cls(type, capacity, urgency_points, real_time_tracking = real_time_tracking)
    
    def admit(self, time):
        """ admits a patient to the ward if there is space

        Returns:
            Bool: Whether a patient was admitted or not
        """
        self.total_arrivals += 1
        if self.current_occupancy < self.capacity:
            self.current_occupancy += 1
            self.total_admissions += 1
            if self.real_time_tracking:
                self.occupancy.append((time, self.current_occupancy))
            return True
        else:
            self.total_rejections += 1
            if self.real_time_tracking:
                self.rejections.append(time)
            return False
        
    def discharge(self, time):
        """discharges a patient from the ward if there is a patient to discharge

        Returns:
            Bool: Whether a patient was discharged or not
        """
        if self.current_occupancy > 0:
            self.current_occupancy -= 1
            if self.real_time_tracking:
                self.occupancy.append((time, self.current_occupancy))
            return True
        else:
            return False
        
    def get_performance_metrics(self):
        """
        Returns the performance metrics of the ward
        
        Returns:
            dict: Dictionary containing the performance metrics of the ward
        """
        return {"Occupied probability": self.total_rejections/(self.total_arrivals if self.total_arrivals > 0 else 1),
                "Estimated admissions": self.total_admissions,
                "Estimated rejections": self.total_rejections,
                "Estimated lost": self.total_lost,
                "Estimated relocations": self.total_relocations}
        
    
    def reset_metrics(self):
        """
        Resets the performance metrics of the ward
        """
        #Performance Metrics
        self.total_arrivals = 0 #Total number of patients that have arrived at the ward
        self.total_admissions = 0 #Total number of patients that have been admitted to the ward
        self.total_rejections = 0 #Total number of patients that have been rejected from the ward
        self.total_lost = 0 #Total number of patients that have been lost
        self.total_relocations = 0 #Total number of patients that have been successfully relocated to another ward
        #self.current_occupancy = 0
        self.occupancy = []
        self.rejections = []
    
    def __repr__(self):
        return f"{self.type} Ward with {self.capacity} beds and {self.urgency_points} urgency points."
    def __str__(self):
        return f"{self.type} Ward with {self.capacity} beds and {self.urgency_points} urgency points."

def initialize_wards(df, real_time_tracking = False):
    """
    Initializes the wards using the data from the dataframe
    
    Args:
        df (pandas.dataframe): Dataframe containing the data for the wards
    Returns:
        list: List of Ward objects
    """
    wards = []
    for ward_type in df.index:
        ward = Ward.from_dataframe(df, ward_type, real_time_tracking = real_time_tracking)
        wards.append(ward)
    return wards