import random
import datetime
from typing import List, Dict, Any
from pm4py.objects.petri_net.semantics import enabled_transitions, execute

class SimulatorLogic:
    def __init__(self, net, initial_marking, final_marking):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.resource_names = ['user_1', 'user_2', 'user_3', 'user_4', 'user_5']
        # Tracks when each resource will be free. key: resource_name, value: datetime
        self.resource_availability = {name: datetime.datetime.min for name in self.resource_names}

    def generate_arrivals(self) -> List[Dict[str, Any]]:
        """
        Creates a list of arriving instances.
        """
        arrivals = []
        base_time = datetime.datetime(2025, 1, 1, 9, 0, 0)
        
        for i in range(10): #Currently hardcoded to 10 instances!!
            arrival_time = base_time + datetime.timedelta(minutes=60 * i) #Currently hardcoded, so that every hour one instance arrives
            instance_data = { #Also currently hardcorded
                'instance_id': f"Application_{652823628 + i}",
                'arrival_time': arrival_time,
                'attributes': {
                    'LoanGoal': 'Existing loan takeover',
                    'ApplicationType': 'New credit',
                    'RequestedAmount': 20000.0
                }
            }
            arrivals.append(instance_data)
        return arrivals

    def select_transition(self, enabled_trans: list):
        """
        Selects a transition, based on enabled transitions
        """
        if not enabled_trans:
            return None
        return random.choice(enabled_trans) #Is currently chosen randomly 
    #Ich glaube es könnte auch Sinn ergeben, hier vielleicht alle bisherigen transitions zu übergeben, um eine gute prediction zu haben.

    def get_processing_time(self, transition):
        """
        Returns minutes for a certain event in minutes
        """
        return random.uniform(1, 10)

    def get_resource_slot(self, ready_time: datetime.datetime, duration_minutes: float):
        """
        Allocates the earliest available resource.
        Returns (resource_name, start_time, end_time)
        """
        # Find the resource that is free earliest
        best_resource = None
        earliest_free_time = datetime.datetime.max
        
        # Determine actual start time for each resource if specific resource is chosen
        for res in self.resource_names:
            free_at = self.resource_availability[res]
            # If resource is free before ready_time, they can start at ready_time.
            # Otherwise they start when they become free.
            start_at = max(ready_time, free_at)
            
            if start_at < earliest_free_time:
                earliest_free_time = start_at
                best_resource = res
            elif start_at == earliest_free_time:
                 # Tie-breaker: random or first found. Let's just keep first found or pick random?
                 # Deterministic is usually better for debugging, so keeping first found.
                 pass
        
        # Calculate end time
        duration = datetime.timedelta(minutes=duration_minutes)
        end_time = earliest_free_time + duration
        
        # Update availability
        self.resource_availability[best_resource] = end_time
        
        return best_resource, earliest_free_time, end_time

class SimulationEngine:
    def __init__(self, net, initial_marking, final_marking):
        self.logic = SimulatorLogic(net, initial_marking, final_marking)
        self.log = []

    def run_simulation(self):
        # 1. Get the list of arriving instances from the generator
        arrivals = self.logic.generate_arrivals()
        
        flat_log = []
        for app in arrivals: 
            # 2. Simulate the process for each arrival
            trace = self._simulate_single_trace(app)
            self.log.append(trace)
            
            # 3. Flatten for DataFrame compatibility
            # Prefix case attributes with 'case:' to distinguish them
            case_attrs = {f"case:{k}": v for k, v in trace['attributes'].items()}
            
            for event in trace['events']:
                merged_event = {**event, **case_attrs}
                flat_log.append(merged_event)
        
        return flat_log

    def _simulate_single_trace(self, app_data: Dict[str, Any]) -> Dict[str, Any]:
        trace = {
        'attributes': {
            'concept:name': app_data['instance_id'],
            'LoanGoal': app_data['attributes']['LoanGoal'],
            'ApplicationType': app_data['attributes']['ApplicationType'],
            'RequestedAmount': app_data['attributes']['RequestedAmount']
        },
        'events': []
        }
        
        current_marking = self.logic.initial_marking
        current_time = app_data['arrival_time']
        
        while current_marking != self.logic.final_marking: #Works given our process model is final, but might lead to infinite loop
            enabled = list(enabled_transitions(self.logic.net, current_marking)) 
            if not enabled: break 
            
            t = self.logic.select_transition(enabled)
            
            if t.label:
                # Calculate duration and find resource slot
                processing_time = self.logic.get_processing_time(t)
                resource, start, end = self.logic.get_resource_slot(current_time, processing_time)
                
                # Advance time to when this specific step finishes
                # Note: This updates 'current_time' for the *token* (processing instance)
                # equal to the end of the activity.
                current_time = end 
                
                trace['events'].append({
                    'concept:name': t.label,
                    'org:resource': resource,
                    'time:timestamp': current_time, 
                    # If you want start timestamp too, you could add 'time:start': start
                    'lifecycle:transition': 'complete'
                })
            else:
                # Hidden transition: immediate, no resource
                pass
            
            current_marking = execute(t, self.logic.net, current_marking)
            
            print(trace)
                
        return trace