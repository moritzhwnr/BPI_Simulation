import random
import datetime
from typing import List, Dict, Any
from pm4py.objects.petri_net.semantics import enabled_transitions, execute
import json
import pickle 

schedule_dict = {(0, 0): 8.500303582270796e-05, (0, 1): 5.843958712811172e-05, (0, 2): 3.187613843351548e-05, (0, 3): 4.250151791135398e-05, (0, 4): 4.7814207650273224e-05, (0, 5): 0.00017000607164541592, (0, 6): 0.0004834547662416515, (0, 7): 0.0017585003035822707, (0, 8): 0.003097298117789921, (0, 9): 0.0028316636308439586, (0, 10): 0.0027625986642380086, (0, 11): 0.002666970248937462, (0, 12): 0.0026510321797207042, (0, 13): 0.0025075895567698847, (0, 14): 0.0024597753491196112, (0, 15): 0.002013509411050395, (0, 16): 0.001933819064966606, (0, 17): 0.0020772616879174257, (0, 18): 0.002125075895567699, (0, 19): 0.0017691256830601092, (0, 20): 0.0012591074681238617, (0, 21): 0.0007756527018822102, (0, 22): 0.0003506375227686703, (0, 23): 0.0001593806921675774, (1, 0): 0.0001009411050394657, (1, 1): 5.312689738919247e-05, (1, 2): 4.7814207650273224e-05, (1, 3): 3.7188828172434734e-05, (1, 4): 6.375227686703097e-05, (1, 5): 0.0001912568306010929, (1, 6): 0.0005631451123254402, (1, 7): 0.0012591074681238617, (1, 8): 0.001933819064966606, (1, 9): 0.002087887067395264, (1, 10): 0.0019178809957498483, (1, 11): 0.0019019429265330905, (1, 12): 0.002061323618700668, (1, 13): 0.001992258652094718, (1, 14): 0.0021782027929568913, (1, 15): 0.001955069823922283, (1, 16): 0.0015778688524590164, (1, 17): 0.0017585003035822707, (1, 18): 0.0021622647237401335, (1, 19): 0.0018966302367941714, (1, 20): 0.0012644201578627808, (1, 21): 0.0007650273224043716, (1, 22): 0.00034532483302975107, (1, 23): 0.00022844565877352764, (2, 0): 6.906496660595022e-05, (2, 1): 5.843958712811172e-05, (2, 2): 4.7814207650273224e-05, (2, 3): 3.7188828172434734e-05, (2, 4): 8.500303582270796e-05, (2, 5): 0.00022313296903460838, (2, 6): 0.00037720097146326655, (2, 7): 0.0012591074681238617, (2, 8): 0.0018010018214936249, (2, 9): 0.0021675774134790526, (2, 10): 0.0021622647237401335, (2, 11): 0.002061323618700668, (2, 12): 0.0021410139647844565, (2, 13): 0.00225789313904068, (2, 14): 0.0022313296903460835, (2, 15): 0.001997571341833637, (2, 16): 0.001572556162720097, (2, 17): 0.0013706739526411658, (2, 18): 0.001870066788099575, (2, 19): 0.001700060716454159, (2, 20): 0.0011634790528233152, (2, 21): 0.0006906496660595021, (2, 22): 0.0004037644201578628, (2, 23): 0.0001912568306010929, (3, 0): 6.906496660595022e-05, (3, 1): 3.187613843351548e-05, (3, 2): 3.187613843351548e-05, (3, 3): 6.375227686703097e-05, (3, 4): 0.0001009411050394657, (3, 5): 0.00018063145112325442, (3, 6): 0.0003878263509411051, (3, 7): 0.001354735883424408, (3, 8): 0.0018647540983606557, (3, 9): 0.001992258652094718, (3, 10): 0.0021357012750455374, (3, 11): 0.0020985124468731026, (3, 12): 0.0019072556162720098, (3, 13): 0.0018010018214936249, (3, 14): 0.0020931997571341835, (3, 15): 0.0015034911961141469, (3, 16): 0.0013706739526411658, (3, 17): 0.0014928658166363084, (3, 18): 0.0018488160291438979, (3, 19): 0.001460989678202793, (3, 20): 0.0011103521554341226, (3, 21): 0.0007544019429265331, (3, 22): 0.00037720097146326655, (3, 23): 0.00011156648451730419, (4, 0): 9.031572556162721e-05, (4, 1): 5.843958712811172e-05, (4, 2): 3.187613843351548e-05, (4, 3): 4.7814207650273224e-05, (4, 4): 6.906496660595022e-05, (4, 5): 0.00018594414086217363, (4, 6): 0.00036126290224650884, (4, 7): 0.0012803582270795387, (4, 8): 0.001891317547055252, (4, 9): 0.0021410139647844565, (4, 10): 0.0021569520340012143, (4, 11): 0.00197632058287796, (4, 12): 0.0019656952034001218, (4, 13): 0.002172890103217972, (4, 14): 0.0020560109289617487, (4, 15): 0.0016416211293260475, (4, 16): 0.0011581663630843959, (4, 17): 0.001120977534911961, (4, 18): 0.0010837887067395263, (4, 19): 0.0009244080145719489, (4, 20): 0.0006906496660595021, (4, 21): 0.00046751669702489376, (4, 22): 0.00030282331511839706, (4, 23): 0.00011156648451730419, (5, 0): 0.0001009411050394657, (5, 1): 3.187613843351548e-05, (5, 2): 2.6563448694596234e-05, (5, 3): 3.187613843351548e-05, (5, 4): 2.6563448694596234e-05, (5, 5): 6.375227686703097e-05, (5, 6): 0.00025500910746812386, (5, 7): 0.00035595021250758955, (5, 8): 0.001030661809350334, (5, 9): 0.0014291135397692775, (5, 10): 0.0014238008500303582, (5, 11): 0.0012803582270795387, (5, 12): 0.0014663023679417123, (5, 13): 0.0011900425015179115, (5, 14): 0.001333485124468731, (5, 15): 0.000945658773527626, (5, 16): 0.0007225258044930177, (5, 17): 0.0005790831815421979, (5, 18): 0.0007650273224043716, (5, 19): 0.0006481481481481482, (5, 20): 0.00046220400728597447, (5, 21): 0.00035595021250758955, (5, 22): 0.0002656344869459624, (5, 23): 0.00015406800242865818, (6, 0): 8.500303582270796e-05, (6, 1): 6.375227686703097e-05, (6, 2): 2.6563448694596234e-05, (6, 3): 3.7188828172434734e-05, (6, 4): 3.187613843351548e-05, (6, 5): 0.00011687917425622344, (6, 6): 0.0002018822100789314, (6, 7): 0.00043032786885245904, (6, 8): 0.0005418943533697632, (6, 9): 0.0008234669095324833, (6, 10): 0.0007756527018822102, (6, 11): 0.0008606557377049181, (6, 12): 0.0006322100789313904, (6, 13): 0.0007012750455373406, (6, 14): 0.0008181542197935641, (6, 15): 0.0006906496660595021, (6, 16): 0.0006322100789313904, (6, 17): 0.0006693989071038252, (6, 18): 0.0009244080145719489, (6, 19): 0.001009411050394657, (6, 20): 0.0009031572556162721, (6, 21): 0.0005578324225865209, (6, 22): 0.00028688524590163934, (6, 23): 9.562841530054645e-05}


class SimulatorLogic:
    def __init__(self, net, initial_marking, final_marking):
        self.net = net
        self.initial_marking = initial_marking
        self.final_marking = final_marking
        self.schedule_dict = schedule_dict
        self.holidays = {
            datetime.date(2016, 1, 1), 
            datetime.date(2016, 3, 28), # Easter Monday
            datetime.date(2016, 4, 27), # King's Day
            datetime.date(2016, 5, 5), # Ascension Day
            datetime.date(2016, 5, 16), # Whit Monday
            datetime.date(2016, 12, 25),
            datetime.date(2016, 12, 26)
        }
        self.monthly_multipliers = {
            1: 1.00,
            2: 1.10,
            3: 1.12,
            4: 0.99,
            5: 0.94,
            6: 1.37,
            7: 1.39,
            8: 1.41,
            9: 1.39,
            10: 1.37,
            11: 1.22,
            12: 1.08,
        }
        
        # Load availability from JSON
        self.availability_probs = self._load_availability_from_json() 

        # Load OrdinoR organizational model for resource permission
        self.org_model = self._load_organizational_model()

        # Dynamically set resource names from the loaded availability data
        if self.availability_probs:
            self.resource_names = sorted(list(set(k[0] for k in self.availability_probs.keys())))
        else:
            self.resource_names = ['User_1', 'User_2', 'User_3', 'User_4', 'User_5']
            
        # Tracks when each resource will be free. key: resource_name, value: datetime
        self.resource_availability = {name: datetime.datetime.min for name in self.resource_names}

    def _get_arrival_rate(self, current_time: datetime.datetime) -> float:
        """
        Helper: Looks up the lambda (rate) for the specific time.
        """
        # 1. Extract context (Weekday 0-6, Hour 0-23)
        day_idx = current_time.weekday()
        hour_idx = current_time.hour
        
        # 2. Look up rate in your dictionary
        # Default to a tiny number (not 0) to avoid math errors if data is missing
        return self.schedule_dict.get((day_idx, hour_idx), 0.000001)
    
    def _get_monthly_multiplier(self, current_time: datetime.datetime) -> float:
        # Returns the specific multiplier for the current month.
        current_month = current_time.month
        return self.monthly_multipliers.get(current_month, 1.0)
    
    def generate_arrivals(self) -> List[Dict[str, Any]]:
        """
        Creates a list of arriving instances.
        """
        arrivals = []
        
        # Setup the clock (Running for 1 Year)
        start_date = datetime.datetime(2016, 1, 1)
        end_time = start_date + datetime.timedelta(days=365)
        current_time = start_date
        
        instance_counter = 0

        while current_time < end_time:
            
            # rate for this specific time
            current_lambda = self._get_arrival_rate(current_time)
            
            # --- STEP 2: Apply Holiday Filter ---
            is_holiday = current_time.date() in self.holidays
            
            if is_holiday:
                current_lambda = 0.000001 
            else:
                # --- STEP 3: Apply Monthly Multiplier ---
                multiplier = self._get_monthly_multiplier(current_time)
                current_lambda = current_lambda * multiplier

            # --- STEP 4: Math ---
            if current_lambda <= 0.000001:
                wait_seconds = 3600
            else:
                wait_seconds = random.expovariate(current_lambda)
            
            # Advance Time
            current_time += datetime.timedelta(seconds=wait_seconds)
            
            # If it's a holiday, we DO NOT spawn. We just loop again.
            if current_time.date() in self.holidays:
                continue # Skip the append, go back to top of loop
            
            # Create Instance
            if current_time < end_time:
                instance_counter += 1
                
                monthly_mult = self._get_monthly_multiplier(current_time)
                season_label = "High Season" if monthly_mult > 1.2 else "Normal"

                instance_data = {
                    'instance_id': f"Application_{instance_counter}",
                    'arrival_time': current_time,
                    'attributes': {
                        'LoanGoal': random.choice(['Existing loan takeover', 'Home improvement', 'Car']),
                        'ApplicationType': 'New credit',
                        'RequestedAmount': round(random.uniform(5000, 50000), 2),
                        'SeasonalityFactor': monthly_mult, 
                        'SeasonLabel': season_label
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
    
    def _get_time_type(self, timestamp: datetime.datetime) -> str:
        """Convert timestamp to OrdinoR time type."""
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        if day_of_week >= 5:
            return 'Weekend'
        elif 9 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 14:
            return 'Lunch'
        elif 14 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 20:
            return 'Evening'
        else:
            return 'Off-Hours'

    def _determine_case_type(self, case_attributes: dict) -> str:
        """
        Determine case complexity from attributes.
        You can make this more sophisticated based on your needs.
        """
        requested_amount = case_attributes.get('RequestedAmount', 0)
        
        # Simple heuristic based on loan amount
        if requested_amount < 10000:
            return 'Simple'
        elif requested_amount < 25000:
            return 'Standard'
        elif requested_amount < 40000:
            return 'Complex'
        else:
            return 'Very Complex'

    def _get_permitted_resources(self, activity: str, case_type: str, time_type: str) -> list:
        """Get resources permitted for this activity in this context."""
        if not self.org_model or 'context_permission_map' not in self.org_model:
            # No org model - allow all resources
            return self.resource_names
        
        # Build context key (note: loaded from JSON, so key is a string)
        context_key = f"{activity}|{case_type}|{time_type}"
        
        permitted = self.org_model['context_permission_map'].get(context_key)
        
        if permitted:
            return permitted
        
        # Fallback: try simple permission map (context-free)
        return self._get_simple_permitted_resources(activity)

    def _get_simple_permitted_resources(self, activity: str) -> list:
        """Get resources permitted for this activity (context-free)."""
        if not self.org_model or 'permission_map' not in self.org_model:
            return self.resource_names
        
        return self.org_model['permission_map'].get(activity, self.resource_names)

    def _load_availability_from_json(self, filename="resource_allocation/resource_availability.json"):
        with open(filename, 'r') as f:
            data = json.load(f)
        
        # Reconstruct the original tuple keys
        reconstructed = {}
        for k, v in data.items():
            parts = k.split('|')
            # parts[0] is resource, parts[1] is weekday (int), parts[2] is hour (int)
            reconstructed[(parts[0], int(parts[1]), int(parts[2]))] = v
        return reconstructed
    
    def _load_organizational_model(self, filename="resource_permissions/organizational_model.pkl"):
        """Load the OrdinoR organizational model with permissions."""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Loaded organizational model:")
            print(f"Simple permissions: {len(data['permission_map'])} activities")
            print(f"Context permissions: {len(data['context_permission_map'])} contexts")
            
            return data
        except FileNotFoundError:
            print(f"Warning: {filename} not found. Using all resources for all activities.")
            return None

    def get_resource_slot_advanced(self, ready_time: datetime.datetime, duration_minutes: float, 
                               activity: str, case_attributes: dict):
        """
        Allocates a resource based on:
        1. OrdinoR permissions (who CAN do this activity in this context)
        2. Availability probabilities (who IS working at this time)
        3. Resource calendar (who is FREE right now)
    
        Args:
            ready_time: When the activity is ready to start
            duration_minutes: How long the activity takes
            activity: Activity name (e.g., "A_Accepted")
            case_attributes: Case attributes for context (e.g., complexity)
    
        Returns:
            (resource_name, start_time, end_time)
        """
        current_check = ready_time
    
        # Determine case type (complexity) from attributes
        case_type = self._determine_case_type(case_attributes)
        
        # Search forward hour-by-hour
        for _ in range(168):  # Max 1 week ahead
            day, hour = current_check.weekday(), current_check.hour
            time_type = self._get_time_type(current_check)
            
            # STEP 1: Get permitted resources for this context (OrdinoR)
            permitted_resources = self._get_permitted_resources(activity, case_type, time_type)
            
            if not permitted_resources:
                # No one has permission for this context - try next hour
                current_check += datetime.timedelta(hours=1)
                current_check = current_check.replace(minute=0, second=0)
                continue
            
            # STEP 2: Filter to resources likely working at this time (Availability)
            active_resources = [
                res for res in permitted_resources
                if self.availability_probs.get((res, day, hour), 0) > 0.3  # 30% threshold
            ]
            
            if not active_resources:
                # No permitted resources are working at this hour
                current_check += datetime.timedelta(hours=1)
                current_check = current_check.replace(minute=0, second=0)
                continue
            
            # STEP 3: Pick the one free earliest (Resource Calendar)
            best_res = min(active_resources, key=lambda r: self.resource_availability[r])
            actual_start = max(current_check, self.resource_availability[best_res])
            
            end_time = actual_start + datetime.timedelta(minutes=duration_minutes)
            self.resource_availability[best_res] = end_time
            
            return best_res, actual_start, end_time
        
        # Fallback: No suitable resource found within 1 week
        # Use simple permission map (ignore context)
        simple_permitted = self._get_simple_permitted_resources(activity)
        if simple_permitted:
            best_res = min(simple_permitted, key=lambda r: self.resource_availability[r])
            actual_start = max(ready_time, self.resource_availability[best_res])
            end_time = actual_start + datetime.timedelta(minutes=duration_minutes)
            self.resource_availability[best_res] = end_time
            return best_res, actual_start, end_time
        
        # Ultimate fallback: System does it
        return "System", ready_time, ready_time + datetime.timedelta(minutes=duration_minutes)


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

        print(f"Generated {len(arrivals)} arrivals based on dynamic rates.")
        
        for app in arrivals: 
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
            
            if t.label:
                # Calculate duration and find resource slot
                processing_time = self.logic.get_processing_time(t)
                resource, start, end = self.logic.get_resource_slot_advanced(current_time, processing_time, t.label, app_data['attributes'])
                
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
            
            #print(trace)
                
        return trace