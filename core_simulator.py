import random
import datetime
from typing import List, Dict, Any
from pm4py.objects.petri_net.semantics import enabled_transitions, execute

class SimulatorLogic:
    def __init__(self, net, initial_marking, final_marking):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking

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

class SimulationEngine:
    def __init__(self, net, initial_marking, final_marking):
        self.logic = SimulatorLogic(net, initial_marking, final_marking)
        self.log = []

    def run_simulation(self):
        # 1. Get the list of arriving instances from the generator
        arrivals = self.logic.generate_arrivals()
        
        for app in arrivals: 
            # 2. Simulate the process for each arrival
            trace = self._simulate_single_trace(app)
            self.log.append(trace)
        
        return self.log

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
            
            # Don't think we have this in our process model, but this is just done so hidden transitions do not have any process time
            duration = self.logic.get_processing_time(t) if t.label else 0
            current_time += datetime.timedelta(minutes=duration)
            
            current_marking = execute(t, self.logic.net, current_marking)
            
            if t.label is not None:
                trace['events'].append({
                    'concept:name': t.label,
                    'org:resource': "System",
                    'time:timestamp': current_time,
                    'lifecycle:transition': 'complete'
                })
                
        return trace