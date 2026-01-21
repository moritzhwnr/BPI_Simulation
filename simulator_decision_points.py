import random
import datetime
import json
import joblib
import pandas as pd
from typing import List, Dict, Any
from pm4py.objects.petri_net.semantics import enabled_transitions, execute

schedule_dict = {(0, 0): 8.500303582270796e-05, (0, 1): 5.843958712811172e-05, (0, 2): 3.187613843351548e-05, (0, 3): 4.250151791135398e-05, (0, 4): 4.7814207650273224e-05, (0, 5): 0.00017000607164541592, (0, 6): 0.0004834547662416515, (0, 7): 0.0017585003035822707, (0, 8): 0.003097298117789921, (0, 9): 0.0028316636308439586, (0, 10): 0.0027625986642380086, (0, 11): 0.002666970248937462, (0, 12): 0.0026510321797207042, (0, 13): 0.0025075895567698847, (0, 14): 0.0024597753491196112, (0, 15): 0.002013509411050395, (0, 16): 0.001933819064966606, (0, 17): 0.0020772616879174257, (0, 18): 0.002125075895567699, (0, 19): 0.0017691256830601092, (0, 20): 0.0012591074681238617, (0, 21): 0.0007756527018822102, (0, 22): 0.0003506375227686703, (0, 23): 0.0001593806921675774, (1, 0): 0.0001009411050394657, (1, 1): 5.312689738919247e-05, (1, 2): 4.7814207650273224e-05, (1, 3): 3.7188828172434734e-05, (1, 4): 6.375227686703097e-05, (1, 5): 0.0001912568306010929, (1, 6): 0.0005631451123254402, (1, 7): 0.0012591074681238617, (1, 8): 0.001933819064966606, (1, 9): 0.002087887067395264, (1, 10): 0.0019178809957498483, (1, 11): 0.0019019429265330905, (1, 12): 0.002061323618700668, (1, 13): 0.001992258652094718, (1, 14): 0.0021782027929568913, (1, 15): 0.001955069823922283, (1, 16): 0.0015778688524590164, (1, 17): 0.0017585003035822707, (1, 18): 0.0021622647237401335, (1, 19): 0.0018966302367941714, (1, 20): 0.0012644201578627808, (1, 21): 0.0007650273224043716, (1, 22): 0.00034532483302975107, (1, 23): 0.00022844565877352764, (2, 0): 6.906496660595022e-05, (2, 1): 5.843958712811172e-05, (2, 2): 4.7814207650273224e-05, (2, 3): 3.7188828172434734e-05, (2, 4): 8.500303582270796e-05, (2, 5): 0.00022313296903460838, (2, 6): 0.00037720097146326655, (2, 7): 0.0012591074681238617, (2, 8): 0.0018010018214936249, (2, 9): 0.0021675774134790526, (2, 10): 0.0021622647237401335, (2, 11): 0.002061323618700668, (2, 12): 0.0021410139647844565, (2, 13): 0.00225789313904068, (2, 14): 0.0022313296903460835, (2, 15): 0.001997571341833637, (2, 16): 0.001572556162720097, (2, 17): 0.0013706739526411658, (2, 18): 0.001870066788099575, (2, 19): 0.001700060716454159, (2, 20): 0.0011634790528233152, (2, 21): 0.0006906496660595021, (2, 22): 0.0004037644201578628, (2, 23): 0.0001912568306010929, (3, 0): 6.906496660595022e-05, (3, 1): 3.187613843351548e-05, (3, 2): 3.187613843351548e-05, (3, 3): 6.375227686703097e-05, (3, 4): 0.0001009411050394657, (3, 5): 0.00018063145112325442, (3, 6): 0.0003878263509411051, (3, 7): 0.001354735883424408, (3, 8): 0.0018647540983606557, (3, 9): 0.001992258652094718, (3, 10): 0.0021357012750455374, (3, 11): 0.0020985124468731026, (3, 12): 0.0019072556162720098, (3, 13): 0.0018010018214936249, (3, 14): 0.0020931997571341835, (3, 15): 0.0015034911961141469, (3, 16): 0.0013706739526411658, (3, 17): 0.0014928658166363084, (3, 18): 0.0018488160291438979, (3, 19): 0.001460989678202793, (3, 20): 0.0011103521554341226, (3, 21): 0.0007544019429265331, (3, 22): 0.00037720097146326655, (3, 23): 0.00011156648451730419, (4, 0): 9.031572556162721e-05, (4, 1): 5.843958712811172e-05, (4, 2): 3.187613843351548e-05, (4, 3): 4.7814207650273224e-05, (4, 4): 6.906496660595022e-05, (4, 5): 0.00018594414086217363, (4, 6): 0.00036126290224650884, (4, 7): 0.0012803582270795387, (4, 8): 0.001891317547055252, (4, 9): 0.0021410139647844565, (4, 10): 0.0021569520340012143, (4, 11): 0.00197632058287796, (4, 12): 0.0019656952034001218, (4, 13): 0.002172890103217972, (4, 14): 0.0020560109289617487, (4, 15): 0.0016416211293260475, (4, 16): 0.0011581663630843959, (4, 17): 0.001120977534911961, (4, 18): 0.0010837887067395263, (4, 19): 0.0009244080145719489, (4, 20): 0.0006906496660595021, (4, 21): 0.00046751669702489376, (4, 22): 0.00030282331511839706, (4, 23): 0.00011156648451730419, (5, 0): 0.0001009411050394657, (5, 1): 3.187613843351548e-05, (5, 2): 2.6563448694596234e-05, (5, 3): 3.187613843351548e-05, (5, 4): 2.6563448694596234e-05, (5, 5): 6.375227686703097e-05, (5, 6): 0.00025500910746812386, (5, 7): 0.00035595021250758955, (5, 8): 0.001030661809350334, (5, 9): 0.0014291135397692775, (5, 10): 0.0014238008500303582, (5, 11): 0.0012803582270795387, (5, 12): 0.0014663023679417123, (5, 13): 0.0011900425015179115, (5, 14): 0.001333485124468731, (5, 15): 0.000945658773527626, (5, 16): 0.0007225258044930177, (5, 17): 0.0005790831815421979, (5, 18): 0.0007650273224043716, (5, 19): 0.0006481481481481482, (5, 20): 0.00046220400728597447, (5, 21): 0.00035595021250758955, (5, 22): 0.0002656344869459624, (5, 23): 0.00015406800242865818, (6, 0): 8.500303582270796e-05, (6, 1): 6.375227686703097e-05, (6, 2): 2.6563448694596234e-05, (6, 3): 3.7188828172434734e-05, (6, 4): 3.187613843351548e-05, (6, 5): 0.00011687917425622344, (6, 6): 0.0002018822100789314, (6, 7): 0.00043032786885245904, (6, 8): 0.0005418943533697632, (6, 9): 0.0008234669095324833, (6, 10): 0.0007756527018822102, (6, 11): 0.0008606557377049181, (6, 12): 0.0006322100789313904, (6, 13): 0.0007012750455373406, (6, 14): 0.0008181542197935641, (6, 15): 0.0006906496660595021, (6, 16): 0.0006322100789313904, (6, 17): 0.0006693989071038252, (6, 18): 0.0009244080145719489, (6, 19): 0.001009411050394657, (6, 20): 0.0009031572556162721, (6, 21): 0.0005578324225865209, (6, 22): 0.00028688524590163934, (6, 23): 9.562841530054645e-05}

class SimulatorLogic:
    def __init__(self, net, initial_marking, final_marking, schedule_dict, fatigue_factor=0.01):
        """
        :param fatigue_factor: How much to boost the 'Stop' probability for every loop iteration.
                               0.05 = 5% increase per loop.
        """
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.schedule_dict = schedule_dict
        self.fatigue_factor = fatigue_factor  # <--- NEW CONTROL PARAMETER
        
        # 1. Load Model
        print(f"Loading decision model with Fatigue Factor: {self.fatigue_factor}")
        try:
            self.clf = joblib.load('decision_model.pkl')
            self.vectorizer = joblib.load('feature_vectorizer.pkl')
            print("Model loaded")
        except:
            print("Model not found.")
            self.clf = None

        # 2. Build Lookahead Map
        self.transition_outcome_map = {}
        self._build_lookahead_map()
        
        # 3. Resource Setup
        self.resource_names = ['User_1']
        self.resource_availability = {name: datetime.datetime(2016, 1, 1, 9, 0) for name in self.resource_names}

    def _build_lookahead_map(self):
        self.transition_outcome_map = {}
        for t in self.net.transitions:
            if t.label is None: 
                self.transition_outcome_map[t] = self._find_next_visible(t)
            else:
                self.transition_outcome_map[t] = t.label

    def _find_next_visible(self, start_trans, visited=None):
        if visited is None: visited = set()
        if start_trans in visited: return None
        visited.add(start_trans)
        
        for arc in start_trans.out_arcs:
            target_place = arc.target
            
            # Robust Check for Final Marking
            for p in self.final_marking:
                if p.name == target_place.name:
                    return '__END__'
            
            for next_arc in target_place.out_arcs:
                next_t = next_arc.target
                if next_t.label: return next_t.label
                res = self._find_next_visible(next_t, visited)
                if res: return res
        return None

    def select_transition(self, enabled_trans, current_trace, app_attributes):
        if not enabled_trans: return None
        if len(enabled_trans) == 1: return enabled_trans[0]
        
        history_names = [e['concept:name'] for e in current_trace]
        last_act = history_names[-1] if history_names else "START"
        
        # Count consecutive repetitions of the current activity
        current_loop_count = 0
        for act in reversed(history_names):
            if act == last_act: current_loop_count += 1
            else: break
            
        # Calculate the bonus probability
        fatigue_bonus = current_loop_count * self.fatigue_factor
        
        candidates = []
        weights = []
        
        # Get Model Predictions
        prob_map = {}
        if self.clf:
            try:
                hist_counts = {f"past_{k}": history_names.count(k) for k in set(history_names)}
                features = {
                    'last_activity': last_act,
                    'RequestedAmount': float(app_attributes.get('RequestedAmount', 0)),
                    'LoanGoal': str(app_attributes.get('LoanGoal', 'Unknown')),
                    **hist_counts
                }
                vec = self.vectorizer.transform(features)
                probs = self.clf.predict_proba(vec)[0]
                prob_map = dict(zip(self.clf.classes_, probs))
            except:
                pass

        # Assign Weights
        for t in enabled_trans:
            outcome = self.transition_outcome_map.get(t)
            w = 0.0
            
            if outcome == '__END__':
                base_prob = prob_map.get('__END__', 0.0)
                w = base_prob + fatigue_bonus
                if w > 1.0: w = 1.0
                
            elif outcome is None:
                w = 0.01 
            else:
                w = prob_map.get(outcome, 0.0)

            candidates.append(t)
            weights.append(w)
            
        total_weight = sum(weights)
        if total_weight > 0:
            norm_weights = [x / total_weight for x in weights]
            return random.choices(candidates, weights=norm_weights, k=1)[0]
        else:
            return random.choice(enabled_trans)
        
        
        
    def _get_arrival_rate(self, current_time: datetime.datetime) -> float:
        """
        Helper: Looks up the lamgbda (rate) for the specific time.
        """
        # 1. Extract context (Weekday 0-6, Hour 0-23)
        day_idx = current_time.weekday()
        hour_idx = current_time.hour
        
        # 2. Look up rate in your dictionary
        # Default to a tiny number (not 0) to avoid math errors if data is missing
        return self.schedule_dict.get((day_idx, hour_idx), 0.000001)
    
    def generate_arrivals(self) -> List[Dict[str, Any]]:
        """
        Creates a list of arriving instances.
        """
        arrivals = []
        # (Paste your existing generate_arrivals code here)
        # ... (same as before) ...
        # Just ensure you include 'attributes' like LoanGoal/RequestedAmount
        # matching what the model expects.
        arrivals = []
        start_date = datetime.datetime(2016, 1, 1)
        duration_days = 31
        current_time = start_date
        end_time = start_date + datetime.timedelta(days=duration_days)
        instance_counter = 0


        while (current_time < end_time):
            current_lambda = self._get_arrival_rate(current_time)
            wait_seconds = random.expovariate(current_lambda)
            current_time += datetime.timedelta(seconds=wait_seconds)
            
        
            if current_time < end_time:
                instance_counter += 1
                instance_data = {
                    'instance_id': f"Application_{instance_counter}",
                    'arrival_time': current_time,
                    'attributes': {
                        'LoanGoal': random.choice(['Existing loan takeover', 'Home improvement', 'Car']),
                        'ApplicationType': 'New credit',
                        'RequestedAmount': round(random.uniform(5000, 50000), 2)
                    }
                }
                arrivals.append(instance_data)
        return arrivals

    def get_processing_time(self, transition):
        return random.uniform(1, 10)

    def load_availability_from_json(self, filename="resource_availability.json"):
        # (Paste your existing load_availability code here)
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            reconstructed = {}
            for k, v in data.items():
                parts = k.split('|')
                reconstructed[(parts[0], int(parts[1]), int(parts[2]))] = v
            return reconstructed
        except:
            return {}

    def get_resource_slot_advanced(self, ready_time, duration_minutes):
        # (Paste your existing get_resource_slot_advanced code here)
        return "System", ready_time, ready_time + datetime.timedelta(minutes=duration_minutes)
    
    

class SimulationEngine:  
    def __init__(self, net, initial_marking, final_marking):
        self.logic = SimulatorLogic(net, initial_marking, final_marking, schedule_dict)
        self.log = []

    def run_simulation(self):
        # Generate arrivals
        arrivals = self.logic.generate_arrivals()
        print(f"Starting simulation for {len(arrivals)} cases...")
        
        # Reset log
        self.log = []
        
        for app in arrivals: 
            trace = self._simulate_single_trace(app)
            self.log.append(trace)
        return self.log
        
        

    def _simulate_single_trace(self, app_data: Dict[str, Any]) -> Dict[str, Any]:
        trace_attributes = app_data['attributes'].copy()
        
        # --- FIX: Inject the Case ID into attributes ---
        trace_attributes['concept:name'] = app_data['instance_id']
        
        trace = {
            'attributes': trace_attributes,
            'events': []
        }
        current_marking = self.logic.initial_marking
        current_time = app_data['arrival_time']
        
        
        
        print(f"\n--- Start Case {app_data['instance_id']} ---")

        while current_marking != self.logic.final_marking:
         
            enabled = list(enabled_transitions(self.logic.net, current_marking))
            if not enabled: break
            
            # Select Transition
            t = self.logic.select_transition(enabled, trace['events'], app_data['attributes'])
           
            if t.label:
                processing_time = self.logic.get_processing_time(t)
                # Note: ensure get_resource_slot_advanced is defined in logic
                resource, start, end = self.logic.get_resource_slot_advanced(current_time, processing_time)
                current_time = end
                
                trace['events'].append({
                    'concept:name': t.label,
                    'org:resource': resource,
                    'time:timestamp': current_time,
                    'lifecycle:transition': 'complete'
                })
            
            current_marking = execute(t, self.logic.net, current_marking)
            
        return trace