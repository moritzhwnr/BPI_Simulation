import random
import datetime
from typing import List, Dict, Any
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
import statistics
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

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

    def allocate_resource(self):
        """
        Allocates a resource for a certain transition
        """
        return random.choice(['user_1', 'user_2', 'user_3', 'user_4', 'user_5'])

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
                    'org:resource': self.logic.allocate_resource(),
                    'time:timestamp': current_time,
                    'lifecycle:transition': 'complete'
                })
                
        return trace
    
 
class log_analyzer:
    def __init__(self, log):
        self.log = log
        self.num_traces = len(self.log)

    # 1. Trace Length Metrics
    def get_trace_length_stats(self):
        lengths = [len(trace['events']) for trace in self.log if trace['events']]
        return {
            "max": max(lengths) if lengths else 0,
            "min": min(lengths) if lengths else 0,
            "avg": statistics.mean(lengths) if lengths else 0
        }

    # 2. Total Time Metrics (Trace Duration)
    def get_total_time_stats(self):
        durations = []
        for trace in self.log:
            if len(trace['events']) > 1:
                start = trace['events'][0]['time:timestamp']
                end = trace['events'][-1]['time:timestamp']
                duration_sec = (end - start).total_seconds()
                durations.append(duration_sec / 60)  # minutes

        return {
            "max": max(durations) if durations else 0,
            "min": min(durations) if durations else 0,
            "avg": statistics.mean(durations) if durations else 0
        }

    # 3. Activity Frequency Metrics (Normalized per Trace)
    def get_activity_frequency_metrics(self):
        total_counts = Counter()
        presence_counts = Counter()

        for trace in self.log:
            activities_in_trace = set()

            for event in trace['events']:
                act = event['concept:name']
                total_counts[act] += 1
                activities_in_trace.add(act)

            for act in activities_in_trace:
                presence_counts[act] += 1

        return {
            act: {
                "presence_per_trace": presence_counts[act] / self.num_traces,
                "avg_occurrences_per_trace": total_counts[act] / self.num_traces
            }
            for act in total_counts
        }

    # 4. Activity Duration Metrics
    def get_activity_duration_stats(self):
        act_durations = defaultdict(list)

        for trace in self.log:
            for i in range(1, len(trace['events'])):
                act_name = trace['events'][i]['concept:name']
                prev_time = trace['events'][i - 1]['time:timestamp']
                curr_time = trace['events'][i]['time:timestamp']

                diff = (curr_time - prev_time).total_seconds() / 60
                act_durations[act_name].append(diff)

        return {
            act: {
                "max": max(times),
                "min": min(times),
                "avg": statistics.mean(times)
            }
            for act, times in act_durations.items()
        }

    # 5. Arrival Interval Metrics
    def get_arrival_stats(self):
        arrivals = [
            trace['events'][0]['time:timestamp']
            for trace in self.log
            if trace['events']
        ]

        intervals = []
        for i in range(1, len(arrivals)):
            diff = (arrivals[i] - arrivals[i - 1]).total_seconds() / 60
            intervals.append(diff)

        return {
            "max": max(intervals) if intervals else 0,
            "min": min(intervals) if intervals else 0,
            "avg": statistics.mean(intervals) if intervals else 0
        }

    # 6. Resource Frequency Metrics (Normalized per Trace)
    def get_resource_frequency_metrics(self):
        total_counts = Counter()
        presence_counts = Counter()

        for trace in self.log:
            resources_in_trace = set()

            for event in trace['events']:
                res = event['org:resource']
                total_counts[res] += 1
                resources_in_trace.add(res)

            for res in resources_in_trace:
                presence_counts[res] += 1

        return {
            res: {
                "presence_per_trace": presence_counts[res] / self.num_traces,
                "avg_occurrences_per_trace": total_counts[res] / self.num_traces
            }
            for res in total_counts
        }

    # --- Main Analysis Method ---
    def print_analysis(self):
        print("=== PROCESS SIMULATION ANALYSIS ===")

        len_stats = self.get_trace_length_stats()
        print(
            f"\nTrace Length | "
            f"Max: {len_stats['max']} | "
            f"Min: {len_stats['min']} | "
            f"Avg: {len_stats['avg']:.2f}"
        )

        time_stats = self.get_total_time_stats()
        print(
            f"Total Time (min) | "
            f"Max: {time_stats['max']:.2f} | "
            f"Min: {time_stats['min']:.2f} | "
            f"Avg: {time_stats['avg']:.2f}"
        )

        arr_stats = self.get_arrival_stats()
        print(
            f"Inter-Arrival (min) | "
            f"Max: {arr_stats['max']:.2f} | "
            f"Min: {arr_stats['min']:.2f} | "
            f"Avg: {arr_stats['avg']:.2f}"
        )

        print("\nActivity Performance (min):")
        act_stats = self.get_activity_duration_stats()
        for act, s in act_stats.items():
            print(
                f" - {act:25} | "
                f"Avg: {s['avg']:6.2f} | "
                f"Max: {s['max']:6.2f}"
            )

        print("\nActivity Frequency (per trace):")
        act_freq = self.get_activity_frequency_metrics()
        for act, s in act_freq.items():
            print(
                f" - {act:25} | "
                f"Presence: {s['presence_per_trace']:.2f} | "
                f"Avg Occurrences: {s['avg_occurrences_per_trace']:.2f}"
            )

        print("\nResource Frequency (per trace):")
        res_freq = self.get_resource_frequency_metrics()
        for res, s in res_freq.items():
            print(
                f" - {res:25} | "
                f"Presence: {s['presence_per_trace']:.2f} | "
                f"Avg Occurrences: {s['avg_occurrences_per_trace']:.2f}"
            )