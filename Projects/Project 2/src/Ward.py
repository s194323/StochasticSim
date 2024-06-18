class Ward:
    def __init__(self, type, capacity, urgency_points):
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
        
    @classmethod
    def from_dataframe(cls, df, type):
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
        return cls(type, capacity, urgency_points)
    
    def admit(self):
        """ admits a patient to the ward if there is space

        Returns:
            Bool: Whether a patient was admitted or not
        """
        if self.current_occupancy < self.capacity:
            self.current_occupancy += 1
            return True
        else:
            return False
        
    def discharge(self):
        """discharges a patient from the ward if there is a patient to discharge

        Returns:
            Bool: Whether a patient was discharged or not
        """
        if self.current_occupancy > 0:
            self.current_occupancy -= 1
            return True
        else:
            return False
        
    def __repr__(self):
        return f"{self.type} Ward with {self.capacity} beds and {self.current_occupancy} patients currently admitted."
    def __str__(self):
        return f"{self.type} Ward with {self.capacity} beds and {self.current_occupancy} patients currently admitted."
    

def initialize_wards(df):
    """
    Initializes the wards using the data from the dataframe
    
    Args:
        df (pandas.dataframe): Dataframe containing the data for the wards
    Returns:
        list: List of Ward objects
    """
    wards = []
    for ward_type in df.index:
        ward = Ward.from_dataframe(df, ward_type)
        wards.append(ward)
    return wards