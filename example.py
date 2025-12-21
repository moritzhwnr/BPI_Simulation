import pm4py 
from core_simulator import SimulationEngine

if __name__ == "__main__":
    
    bpmn_model = pm4py.read_bpmn('BPMN.bpmn')
    net, im , fm = pm4py.convert_to_petri_net(bpmn_model)

    simulator = SimulationEngine(net, im, fm)
    log = simulator.run_simulation()
    
    df = pm4py.convert_to_dataframe(log)
    print(df)
    
    
    
