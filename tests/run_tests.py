#!/usr/bin/env python3
"""
Test runner for ReversiSB3 test suite.
Run all tests to identify potential bugs in the Reversi environment.
"""

import unittest
import sys
import os

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

def run_all_tests():
    """Run all test suites and provide summary."""
    
    # Import all test modules
    from test_reversi_environment import TestReversiEnvironment
    from test_game_scenarios import TestGameScenarios, TestEdgeCaseScenarios
    from test_training_components import TestTrainingComponents, TestEnvironmentIntegration
    from test_bc_system import (TestBCDatasetGeneration, TestBCTraining,
                                 TestBCDatasetClass, TestBCIntegration)
    
    # Create test suite
    test_classes = [
        TestReversiEnvironment,
        TestGameScenarios,
        TestEdgeCaseScenarios,
        TestTrainingComponents,
        TestEnvironmentIntegration,
        TestBCDatasetGeneration,
        TestBCTraining,
        TestBCDatasetClass,
        TestBCIntegration
    ]
    
    suite = unittest.TestSuite()
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split(chr(10))[-2] if chr(10) in traceback else traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    print(f"\nSuccess rate: {success_rate:.1f}%")
    
    if result.failures or result.errors:
        print("\n⚠️  ISSUES FOUND - Review failures and errors above")
        print("These could explain the training plateau!")
        return False
    else:
        print("\n✅ ALL TESTS PASSED - Environment appears to be working correctly")
        print("Training issues may be related to hyperparameters or network architecture")
        return True


def run_specific_test(test_name):
    """Run a specific test class or method."""
    
    # Map test names to classes
    test_map = {
        'environment': 'test_reversi_environment.TestReversiEnvironment',
        'scenarios': 'test_game_scenarios.TestGameScenarios',
        'edge_cases': 'test_game_scenarios.TestEdgeCaseScenarios',
        'training': 'test_training_components.TestTrainingComponents',
        'integration': 'test_training_components.TestEnvironmentIntegration',
        'bc_dataset': 'test_bc_system.TestBCDatasetGeneration',
        'bc_training': 'test_bc_system.TestBCTraining',
        'bc_dataset_class': 'test_bc_system.TestBCDatasetClass',
        'bc_integration': 'test_bc_system.TestBCIntegration',
        'bc': 'test_bc_system'  # Run all BC tests
    }
    
    if test_name in test_map:
        test_name = test_map[test_name]
    
    suite = unittest.TestLoader().loadTestsFromName(test_name)
    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        # Run specific test
        test_name = sys.argv[1]
        run_specific_test(test_name)
    else:
        # Run all tests
        run_all_tests()