import pandas as pd

from models.equity_processes import EquityProcessSimulator
from validation.validation_tests.martingale_test import run_martingale_test
from validation.validation_tests.option_pricing_test import run_option_pricing_test
from validation.validation_tests.correlation_structure_check import run_correlation_test

from utils.output_writer import OutputWriter


def run_question_1(config, portfolio):
    """
    Runs Question 1 - Monte Carlo Model Validation.

    Executes:
        a) Martingale Test
        b) Option Pricing Test
        c) Correlation Structure Check

    Saves results to:
        output/q1/q1_validation_results.csv

    Returns:
        pandas.DataFrame of results
    """

    # --- Initialize simulator ---
    simulator = EquityProcessSimulator(config, portfolio)

    # --- Run Validation Tests ---
    martingale_results = run_martingale_test(simulator, portfolio, config)
    option_results = run_option_pricing_test(simulator, portfolio, config)
    correlation_result = run_correlation_test(simulator, config)

    # Flatten results
    all_results = []
    all_results.extend(martingale_results)
    all_results.extend(option_results)
    all_results.append(correlation_result)

    results_df = pd.DataFrame(all_results)

    # --- Save Results ---
    output_path = "output/q1"
    OutputWriter.ensure_directory(output_path)

    OutputWriter.save_dataframe(
        results_df,
        f"{output_path}/q1_validation_results.csv"
    )

    return results_df
