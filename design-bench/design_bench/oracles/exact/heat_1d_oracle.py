from design_bench.oracles.exact_oracle import ExactOracle
from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.datasets.continuous.heat_1d_dataset import Heat1DDataset
import numpy as np
import sys
import os, pdb

sys.path.append("/home/ubuntu/codebase/")
from ExPO.generation.simulator import Simulator


class Heat1DOracle(ExactOracle):
    """An exact oracle for 1D heat transfer optimization problems that evaluates
    simulation parameters using the Simulator.metric() function.
    
    This oracle takes parameter combinations (cfl, n_space) and returns the
    metric score which balances simulation accuracy and computational cost.
    
    The oracle supports evaluation across different heat transfer profiles
    and handles parameter validation and conversion.

    Public Attributes:

    name: str
        Name identifier for this oracle type
    profile: str
        Current heat transfer profile being evaluated (p1-p10)
    simulator: Simulator
        Instance of the Simulator class for metric evaluation
    """

    name = "heat_1d_prediction"

    @classmethod
    def supported_datasets(cls):
        """Defines the set of dataset classes compatible with this oracle
        
        Returns:
        
        supported_datasets: set
            Set containing Heat1DDataset class
        """
        return {Heat1DDataset}

    @classmethod
    def fully_characterized(cls):
        """Indicates whether all possible inputs have been pre-evaluated
        
        Returns:
        
        fully_characterized: bool
            False, as this oracle performs live simulation
        """
        return False

    @classmethod
    def is_simulated(cls):
        """Indicates whether values come from simulation vs real experiments
        
        Returns:
        
        is_simulated: bool
            True, as this uses computational simulation
        """
        return True

    def protected_predict(self, x, observation=None):
        """Core prediction function that evaluates simulation parameters
        
        Arguments:
        
        x: np.ndarray
            Parameter combination [cfl, n_space] as a numpy array
            
        Returns:
        
        y: np.ndarray
            Metric score as a single-element array
        """
        
        # Extract parameters
        cfl = float(x[0])
        n_space = int(x[1])
        
        # Validate parameter ranges
        if not (0.01 <= cfl <= 1.5):
            pdb.set_trace()
            return np.array([0.0], dtype=np.float32)
        
        if not (10 <= n_space <= 500):
            pdb.set_trace()
            return np.array([0.0], dtype=np.float32)
        
        try:
            # Evaluate using simulator
            params = {"cfl": cfl, "n_space": n_space}
            #pdb.set_trace()
            if observation is not None:
                self.set_profile(f"p{observation}")
            else:
                self.set_profile(f"p{np.random.randint(1, 11)}")
            metric_score = self.simulator.metric(params=params)
            return np.array([metric_score], dtype=np.float32)
            
        except Exception as e:
            print(f"Error evaluating parameters cfl={cfl}, n_space={n_space}: {e}")
            return np.array([0.0], dtype=np.float32)

    def batch_predict(self, x_batch):
        """Efficiently evaluate a batch of parameter combinations
        
        Arguments:
        
        x_batch: np.ndarray
            Batch of parameter combinations [batch_size, 2]
            
        Returns:
        
        y_batch: np.ndarray
            Batch of metric scores [batch_size, 1]
        """
        
        batch_size = x_batch.shape[0]
        results = np.zeros((batch_size, 1), dtype=np.float32)
        
        for i in range(batch_size):
            #pdb.set_trace()
            results[i] = self.protected_predict(x_batch[i])
        
        return results

    def set_profile(self, profile):
        """Change the heat transfer profile for evaluation
        
        Arguments:
        
        profile: str
            Profile identifier (p1, p2, ..., p10)
        """
        
        self.profile = profile
        self.simulator = Simulator("1D_heat_transfer", profile, verbose=True)

    def __init__(self, dataset: ContinuousDataset, profile="p1", **kwargs):
        """Initialize the 1D heat transfer oracle
        
        Arguments:
        
        dataset: ContinuousDataset
            Dataset instance containing parameter combinations and scores
        profile: str
            Heat transfer profile to use for evaluation (default: p1)
        **kwargs: dict
            Additional keyword arguments for oracle configuration
        """
        
        # Store profile and initialize simulator
        self.profile = profile
        
        try:
            self.simulator = Simulator("1D_heat_transfer", profile, verbose=True)
        except Exception as e:
            print(f"Warning: Could not initialize simulator for profile {profile}: {e}")
            self.simulator = None
        
        # Initialize the oracle using the parent class
        super(Heat1DOracle, self).__init__(
            dataset, 
            internal_batch_size=1, 
            is_batched=False,
            expect_normalized_y=False,
            expect_normalized_x=False, 
            expect_logits=None, 
            **kwargs)