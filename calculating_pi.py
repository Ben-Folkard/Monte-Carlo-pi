# Numpy is a 3rd-party module that has some stuff for maths
import numpy as np
# unittest is a module that is useful for unit testing
import unittest
# Numba is a 3rd-party module that is used for easy parallelisation
from numba import njit

#############################################################
# calc_pi:
# Implements a serial processing Monte-Carlo approach to
# calculate pi
#
# i.e. it creates looks at the positive quadrant of a unit
# circle in a unit square and each loop checks whether a
# normal uniformly randomly generated 2d point is inside
# the circle or outside of it, and then uses the ratio of
# the number of points in the circle to the total amount
# of points sampled to get a value of pi
#
# num_samples = Specifies how many data points are sampled
# seed        = Specifies the random number generator's 
#               so that the results are repeatable and
#               seed reproducable
# pi_collection_rate = The amount of iterations between 
#               storing the current value of pi
#############################################################
def calc_pi_serial(num_samples = 1e5, seed = 12345, pi_collection_rate = 1e3):
    # Sets the RNG's seed
    np.random.seed(seed)

    # Sets the initial amount of points in the circle to 0
    num_in_circle = 0

    # Stores previous values of pi
    pi_values  = np.zeros(int(num_samples//pi_collection_rate))
    num_pi_stored = 0

    # Takes a number of samples = num_samples
    for i in range(1, int(num_samples)+1):
        # Checks if the randomly generated 2d point is inside
        # the unit circle finding the magnitude of the vector
        if np.linalg.norm(np.random.rand(2)) < 1:
            num_in_circle += 1

        # Checks if current number of iterations is a multiple
        # of the pi_collection_rate
        if i%pi_collection_rate == 0:
            # Uses the ratio of points in the unit circle to total
            # number of points to get a value for pi and stores it
            pi_values[num_pi_stored] = 4*num_in_circle/i
            num_pi_stored += 1

    # Uses the ratio of points in the unit circle to total
    # number of points to get a value for pi
    pi = 4*num_in_circle/num_samples
    
    return pi, pi_values

#############################################################
# calc_pi_parallel:
# Implements a parallel processing Monte-Carlo approach to
# calculate pi
#
# i.e. it creates looks at the positive quadrant of a unit
# circle in a unit square and each loop checks whether a
# normal uniformly randomly generated 2d point is inside
# the circle or outside of it, and then uses the ratio of
# the number of points in the circle to the total amount
# of points sampled to get a value of pi
#
# num_samples = Specifies how many data points are sampled
# seed        = Specifies the random number generator's 
#               so that the results are repeatable and
#               seed reproducable
# pi_collection_rate = The amount of iterations between 
#               storing the current value of pi
#############################################################
@njit
def calc_pi(num_samples = 1e5, seed = 12345, pi_collection_rate = 1e3):
    # Sets the RNG's seed
    np.random.seed(seed)

    # Sets the initial amount of points in the circle to 0
    num_in_circle = 0

    # Stores previous values of pi
    pi_values  = np.zeros(int(num_samples//pi_collection_rate))
    num_pi_stored = 0

    # Takes a number of samples = num_samples
    for i in range(1, int(num_samples)+1):
        # Generates 2 uniformly pseudo-randomly distributed
        # numbers which represent a 2D point inside the unit
        # circle.
        x = np.random.rand()
        y = np.random.rand()
        
        # Checks if those points are inside the unit circle
        # by finding the square magnitude of the vector
        # (there's no point finding the sqrt as numbers < 1
        # will remain < 1, and numbers > 1 will remain > 1)
        if (x**2 + y**2) < 1:
            num_in_circle += 1

        # Checks if current number of iterations is a multiple
        # of the pi_collection_rate
        if i%pi_collection_rate == 0:
            # Uses the ratio of points in the unit circle to total
            # number of points to get a value for pi and stores it
            pi_values[num_pi_stored] = 4*num_in_circle/i
            num_pi_stored += 1

    # Uses the ratio of points in the unit circle to total
    # number of points to get a value for pi
    pi = 4*num_in_circle/num_samples
    
    return pi, pi_values

#############################################################
# display_difference:
# Finds and dispays the difference and percetage difference
# between a given new_value and it's original_value
#############################################################
def display_difference(original_value, new_value):
    diff = new_value-original_value
    print(f"\tAbsolute difference   = {diff:.2e}")
    percentage_diff = diff/original_value
    print(f"\tPercentage difference = {percentage_diff:.5%}")

#############################################################
# evaluate_results:
# Evalutes and displays pi vs the calculated value of pi
#
# pi               = the inputted calculated value of pi
# num_samples_used = how many samples were used to get
#                    the calculated value of pi
#############################################################
def evaluate_results(pi, num_samples_used = -1):
    # Checks if the num_samples_used is the final amount
    if num_samples_used == -1:
        print(f"Final calculated value of pi = {pi}")
    # Or just one of the intermediary steps  
    else:
        print(f"Calculated value of pi after {num_samples_used:.0f} samples = {pi:.5f}...")
    print("Difference between pi and calculated value of pi:")
    display_difference(np.pi, pi)

#############################################################
# output_results:
# Evalutes and displays pi vs the calculated value of pi
#
# pi          = the inputted calculated value of pi
# pi_values   = holds stored values of pi
# num_samples = how many total samples which will be
#               used to get the calculated value of pi
# seed        = Specifies the random number generator's 
#               so that the results are repeatable and
#               seed reproducable
# pi_collection_rate = The amount of samples between storing 
#                      the current value of calculated pi
# tilde_length       = Speficies how long the outputted
#                      tildes (~) are.
#############################################################
def output_results(pi, pi_values, num_samples, seed, pi_collection_rate, tilde_length = 60):
    print("Calculating pi using a Monte-Carlo approach:")
    print("~"*tilde_length)
    print("Initial variables:")
    print(f"Max number of samples = {num_samples:,.0f}")
    print(f"RNG seed              = {seed:,}")
    print(f"The amount of samples between storing the current value of \n\tcalculated pi = {pi_collection_rate:,.0f}")
    print("~"*tilde_length)
    print(f"Actual value of pi = {np.pi}\n")

    # Evalutes and displays pi vs the calculated values of pi
    for i, value in enumerate(pi_values):       
        evaluate_results(value, num_samples_used=(i+1)*pi_collection_rate)

        # If it isn't the first value of calculated pi:
        if i != 0:
            print("Difference from the previous value:")
            display_difference(value, pi_values[i-1])

        print()

    # Evalutes and displays pi vs the final calculated value of pi
    evaluate_results(pi)

    print("~"*tilde_length)

#############################################################
# Test_Pi_Calculation:
# A class that contains all the unit tests
#############################################################
class Test_Pi_Calculation(unittest.TestCase):
    # A basic test to see that the returned values of pi #
    # are outputting the expected values                 #
    def test_calc_pi_basic(self):
        for i in range(2):
            # Checks both serial
            if i == 0:
                pi, pi_values = calc_pi_serial(num_samples=1000, seed=42)
            # And parallel versions of calc_pi
            else:
                pi, pi_values = calc_pi(num_samples=1000, seed=42)

            # Testing to see whether the outputted value of pi
            # is within a very wide range (between 2.5 and 4)
            self.assertTrue(2.5 < pi < 4.0)

            # Testing to see that with a sample rate = 1000
            # and a number of samples = 1000, that the
            # returned pi_values array only outputs
            # 1 value of pi
            self.assertEqual(len(pi_values), 1)
        
    # Tests that calc_pi returns consistent results with same seed #
    def test_calc_pi_with_seed(self):
        pi1, _ = calc_pi_serial(num_samples=1000, seed=123)
        pi2, _ = calc_pi_serial(num_samples=1000, seed=123)
        pi3, _ = calc_pi(num_samples=1000, seed=123)
        pi4, _ = calc_pi(num_samples=1000, seed=123)

        # Testing that all values of returned pi are equal
        self.assertEqual(pi1, pi2)
        self.assertEqual(pi2, pi3)
        self.assertEqual(pi3, pi4)
        
    # Tests that pi_values has correct length based on collection rate #
    def test_calc_pi_collection_rate(self):
        num_samples = 10000
        pi_collection_rate = 500
        expected_length = num_samples // pi_collection_rate

        for i in range(2):
            # Checks both serial
            if i == 0:
                _, pi_values = calc_pi_serial(num_samples=num_samples, pi_collection_rate=pi_collection_rate)
            # And parallel versions of calc_pi
            else:
                _, pi_values = calc_pi(num_samples=num_samples, pi_collection_rate=pi_collection_rate)

            # Tests the length = the expected length
            self.assertEqual(len(pi_values), expected_length)
        
    # Tests display_difference calculation (though it prints) #
    def test_display_difference(self):
        import io
        import sys
        
        # Sets it to listen for print statment output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        display_difference(10.0, 10.5)

        # Stores what the print statments outputted
        output = captured_output.getvalue()
        
        # Restores it to output to the standard output
        sys.stdout = sys.__stdout__
        
        # Tests that the printed output is what we'd expect it to be
        self.assertIn("Absolute difference   = 5.00e-01", output)
        self.assertIn("Percentage difference = 5.00000%", output)
        
    # Tests evaluate_results output #
    def test_evaluate_results(self):
        import io
        import sys
        
        # Sets it to listen for print statment output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        evaluate_results(3.14, 1000)

        # Stores what the print statments outputted
        output = captured_output.getvalue()
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        
        # Tests that the printed output is what we'd expect it to be
        self.assertIn("Calculated value of pi after 1000 samples = 3.14000...", output)
        self.assertIn("Difference between pi and calculated value of pi:", output)

#############################################################
# parallel_vs_serial_benchmark:
# Times the difference between the calc_pi with and without
# parallelisation (using njit)
#############################################################
def parallel_vs_serial_benchmark():
    import time
    start = time.time()
    calc_pi(1e7)
    print("With Numba:", time.time() - start)

    start = time.time()
    calc_pi_serial(1e7)
    
    print("Without Numba:", time.time() - start)

# Only runs the following if this file is directly being run
# Rather than if it's being used as a module
if __name__ == "__main__":
    #############################################################
    # Inputs:
    # num_samples = Specifies how many data points are sampled.
    # seed        = Specifies the random number generator's seed
    #               so that the results are repeatable and
    #               reproducable
    # pi_collection_rate = The amount of samples between storing 
    #                      the current value of calculated pi
    #############################################################
    num_samples = 1e5
    seed        = 12345
    pi_collection_rate = 1e3

    # Calulates pi by using a Monte-Carlo approach
    pi, pi_values = calc_pi(num_samples=num_samples, seed=seed, pi_collection_rate=pi_collection_rate)

    # Outputs the results to the console
    output_results(pi, pi_values, num_samples, seed, pi_collection_rate)

    # Run the unit tests
    print("\nRunning unit tests...")
    unittest.main()

    # Alternative unittest syntaxes #

    #unittest.main(argv=['first-arg-is-ignored'], exit=False)

    #runner = unittest.TextTestRunner()
    #suite = unittest.TestLoader().loadTestsFromTestCase(Test_Pi_Calculation)
    #runner.run(suite)
