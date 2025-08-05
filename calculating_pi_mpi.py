import numpy as np
import unittest
from mpi4py import MPI

def calc_pi_serial(num_samples = 1e5, seed = 12345):
    """
    Implements a serial processing Monte-Carlo approach to
    calculate pi
    
    i.e. it creates looks at the positive quadrant of a unit
    circle in a unit square and each loop checks whether a
    normal uniformly randomly generated 2d point is inside
    the circle or outside of it, and then uses the ratio of
    the number of points in the circle to the total amount
    of points sampled to get a value of pi
    
    num_samples = Specifies how many data points are sampled
    seed        = Specifies the random number generator's 
                  so that the results are repeatable and
                  seed reproducable
    pi_collection_rate = The amount of iterations between 
                  storing the current value of pi
    """
    # Sets the RNG's seed
    np.random.seed(seed)

    # Sets the initial amount of points in the circle to 0
    num_in_circle = 0

    for i in range(1, int(num_samples)+1):
        # Checks if the randomly generated 2d point is inside
        # the unit circle finding the magnitude of the vector
        if np.linalg.norm(np.random.rand(2)) < 1:
            num_in_circle += 1

    pi = 4*num_in_circle/num_samples
    
    return pi

def calc_pi(num_samples = 1e5, seed = 12345):
    """
    Implements a parallel processing Monte-Carlo approach to
    calculate pi (using mpi4py)
    
    i.e. it creates looks at the positive quadrant of a unit
    circle in a unit square and each loop checks whether a
    normal uniformly randomly generated 2d point is inside
    the circle or outside of it, and then uses the ratio of
    the number of points in the circle to the total amount
    of points sampled to get a value of pi
    
    num_samples = Specifies how many data points are sampled
    seed        = Specifies the random number generator's 
                  so that the results are repeatable and
                  seed reproducable
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Divides work among processes
    samples_per_proc = int(num_samples)//size

    # Rank 0 potentially does extra work if the number of
    # samples can't be split nicely among the processes
    if rank == 0:
        samples_per_proc += int(num_samples)%size

    # Sets a unique RNG's seed to each rank
    np.random.seed(seed + rank)

    # Sets the initial amount of points in the circle to 0
    num_in_circle = 0
    
    for i in range(samples_per_proc):
        # Checks if the randomly generated 2d point is inside
        # the unit circle finding the magnitude of the vector
        if np.linalg.norm(np.random.rand(2)) < 1:
            num_in_circle += 1

    # Combines all processes to a value on rank 0
    total_in_circle = comm.reduce(num_in_circle, op=MPI.SUM, root=0)

    if rank == 0:
        pi = 4*total_in_circle/num_samples
        return pi
    else:
        return None
    
    return pi

def display_difference(original_value, new_value):
    diff = new_value-original_value
    print(f"Absolute difference   = {diff:.2e}")
    percentage_diff = diff/original_value
    print(f"Percentage difference = {percentage_diff:.5%}")

def output_results(pi, num_samples, seed, tilde_length = 60):
    print("Calculating pi using a Monte-Carlo approach:")
    print("~"*tilde_length)
    print("Initial variables:")
    print(f"Max number of samples = {num_samples:,.0f}")
    print(f"RNG seed              = {seed:,}")
    print("~"*tilde_length)
    print(f"Actual value of pi = {np.pi}\n")
    print(f"Final calculated value of pi = {pi}")
    print("Difference between pi and calculated value of pi:")
    display_difference(np.pi, pi)

    print("~"*tilde_length)

class Test_Pi_Calculation(unittest.TestCase):
    """
    This class contains the unit tests
    """
    def test_calc_pi_basic(self):
        """
        A basic test to see that the returned values of pi
        are outputting the expected values                 
        """
        pi_serial = calc_pi_serial(num_samples=1000, seed=42)
        pi_parallel = calc_pi(num_samples=1000, seed=42)
        
        self.assertTrue(2.5 < pi_serial < 4.0)
        self.assertTrue(2.5 < pi_parallel < 4.0)

    def test_calc_pi_with_seed(self):
        """
        Tests that calc_pi returns consistent results with same seed
        """
        pi1 = calc_pi_serial(num_samples=1000, seed=123)
        pi2 = calc_pi_serial(num_samples=1000, seed=123)
        pi3 = calc_pi(num_samples=1000, seed=123)
        pi4 = calc_pi(num_samples=1000, seed=123)

        # Testing that all values of returned pi are equal
        self.assertEqual(pi1, pi2)
        self.assertEqual(pi2, pi3)
        self.assertEqual(pi3, pi4)

def parallel_vs_serial_benchmark():
    """
    Times the difference between the calc_pi with and without
    parallelisation (using njit)
    """
    import time
    start = time.time()
    calc_pi(1e7)
    print("With Numba:", time.time() - start)

    start = time.time()
    calc_pi_serial(1e7)
    
    print("Without Numba:", time.time() - start)

if __name__ == "__main__":
    """
    Inputs:
    num_samples = Specifies how many data points are sampled.
    seed        = Specifies the random number generator's seed
                  so that the results are repeatable and
                  reproducable
    """
    num_samples = 1e5
    seed        = 12345

    # Calulates pi by using a Monte-Carlo approach
    pi = calc_pi(num_samples=num_samples, seed=seed)

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        # Outputs the results to the console
        output_results(pi, num_samples, seed)

        # Run the unit tests
        print("\nRunning unit tests...")
        unittest.main()

    # Alternative unittest syntaxes #

    #unittest.main(argv=['first-arg-is-ignored'], exit=False)

    #runner = unittest.TextTestRunner()
    #suite = unittest.TestLoader().loadTestsFromTestCase(Test_Pi_Calculation)
    #runner.run(suite)
