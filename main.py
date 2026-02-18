from data.orchestrator import initialize
from models.equity_processes import EquityProcessSimulator
from validation.orchestrator import run_question_1
from workflows.exposures_CVA_workflow import run_q2_workflow
from utils.output_writer import OutputWriter


def main():
    # step 0: initializing parameters
    data = initialize()

    # step 1: simulate equity processes
    simulator = EquityProcessSimulator(data, data['portfolio'])
    equity_paths = simulator.simulate_paths()

    # step 2: run validation tests
    validation_results = run_question_1(data, data['portfolio'])

    
    # step 3: run exposures and CVA workflow
    cva_results = run_q2_workflow(data, data['portfolio'])
    
    return data, equity_paths, validation_results, cva_results


if __name__ == "__main__":
    main()

