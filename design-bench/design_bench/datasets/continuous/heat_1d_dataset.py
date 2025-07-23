from design_bench.datasets.continuous_dataset import ContinuousDataset
from design_bench.disk_resource import DiskResource, SERVER_URL
import numpy as np
import sys
import os

sys.path.append("/home/ubuntu/codebase/")
from ExPO.generation.simulator import Simulator


HEAT_1D_FILES = ["heat_1d/heat_1d-x-0.npy",
                  "heat_1d/heat_1d-x-1.npy", 
                  "heat_1d/heat_1d-x-2.npy",
                  "heat_1d/heat_1d-x-3.npy",
                  "heat_1d/heat_1d-x-4.npy"]


class Heat1DDataset(ContinuousDataset):
    """A dataset for 1D heat transfer optimization problems where the goal is 
    to find optimal simulation parameters (cfl, n_space) that maximize the 
    Simulator.metric() score which balances simulation accuracy and computational cost.

    The dataset contains parameter combinations and their corresponding metric scores
    from the 1D heat transfer simulator across different problem profiles.

    Input parameters:
    - cfl: Courant-Friedrichs-Lewy number (stability parameter) [0.1, 1.0]
    - n_space: Number of spatial grid points (resolution parameter) [30, 300]

    Output:
    - metric_score: Simulator.metric() value (success/cost ratio)

    Public Attributes:

    name: str
        Name of the dataset for identification in plots and experiments
    x_name: str
        Name of the design variables (simulation parameters)
    y_name: str
        Name of the prediction variable (metric score)
    """

    name = "heat_1d/heat_1d"
    x_name = "simulation_parameters"
    y_name = "metric_score"

    @staticmethod
    def register_x_shards():
        """Registers remote files containing design values (cfl, n_space combinations)
        
        Returns:
        
        resources: list of DiskResource
            List of DiskResource objects for parameter combinations data
        """
        return [DiskResource(
            file, is_absolute=False,
            download_target=f"{SERVER_URL}/{file}",
            download_method="direct") for file in HEAT_1D_FILES]

    @staticmethod
    def register_y_shards():
        """Registers remote files containing prediction values (metric scores)
        
        Returns:
        
        resources: list of DiskResource
            List of DiskResource objects for metric score data
        """
        return [DiskResource(
            file.replace("-x-", "-y-"), is_absolute=False,
            download_target=f"{SERVER_URL}/{file.replace('-x-', '-y-')}",
            download_method="direct") for file in HEAT_1D_FILES]

    @staticmethod
    def generate_data(num_samples=1000,
                      save_dir="/home/ubuntu/codebase/design-bench/design-bench/design_bench_data/heat_1d_data",
                      profiles=None):
        """Generate synthetic dataset by sampling parameter combinations
        and evaluating them using the Simulator
        
        Arguments:
        
        num_samples: int
            Total number of parameter combinations to generate
        save_dir: str
            Directory to save the generated data shards
            
        Returns:
        
        x_data: np.ndarray
            Array of parameter combinations [num_samples, 2]
        y_data: np.ndarray  
            Array of metric scores [num_samples, 1]
        """
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Parameter ranges based on ground truth data
        cfl_min, cfl_max = 0.1, 1.0
        n_space_min, n_space_max = 30, 300
        
        # Heat transfer profiles
        if not profiles:
            profiles = [f"p{i}" for i in range(1,11)]
        samples_per_profile = num_samples // len(profiles)
        
        all_x = []
        all_y = []
        
        print(f"Generating {num_samples} parameter combinations across {len(profiles)} profiles...")
        
        for profile in profiles:
            print(f"Processing profile {profile}...")
            
            # Generate random parameter combinations for this profile
            cfl_samples = np.random.uniform(cfl_min, cfl_max, samples_per_profile)
            n_space_samples = np.random.uniform(n_space_min, n_space_max, samples_per_profile).astype(int)
            
            # Create parameter combinations
            profile_x = np.column_stack([cfl_samples, n_space_samples])
            profile_y = []
            
            # Initialize simulator for this profile
            simulator = Simulator("1D_heat_transfer", profile)
            from tqdm import tqdm
            # Evaluate each parameter combination
            for params in tqdm(profile_x, desc=f"Generating samples for {profile}"):
                try:
                    metric_score = simulator.metric(params={"cfl": params[0], "n_space": int(params[1])})
                    profile_y.append(metric_score)
                except Exception as e:
                    print(f"Error evaluating params {params}: {e}")
                    profile_y.append(0.0)  # Default to 0 for failed evaluations
            
            all_x.append(profile_x)
            all_y.extend(profile_y)
        
        # Combine all data
        x_data = np.vstack(all_x).astype(np.float32)
        y_data = np.array(all_y).reshape(-1, 1).astype(np.float32)
        
        print(f"Generated {len(x_data)} parameter combinations")
        print(f"Parameter ranges: CFL [{x_data[:, 0].min():.3f}, {x_data[:, 0].max():.3f}], "
              f"n_space [{x_data[:, 1].min():.0f}, {x_data[:, 1].max():.0f}]")
        print(f"Metric score range: [{y_data.min():.6f}, {y_data.max():.6f}]")
        
        # Save data as shards
        shard_size = len(x_data) // len(HEAT_1D_FILES)
        for i, filename in enumerate(HEAT_1D_FILES):
            start_idx = i * shard_size
            end_idx = (i + 1) * shard_size if i < len(HEAT_1D_FILES) - 1 else len(x_data)
            
            x_shard = x_data[start_idx:end_idx]
            y_shard = y_data[start_idx:end_idx]
            
            x_path = os.path.join(save_dir, filename)
            y_path = os.path.join(save_dir, filename.replace("-x-", "-y-"))
            
            os.makedirs(os.path.dirname(x_path), exist_ok=True)
            np.save(x_path, x_shard)
            np.save(y_path, y_shard)
            
            print(f"Saved shard {i+1}: {x_shard.shape} samples to {x_path} and {y_path}")
        
        return x_data, y_data

    def __init__(self, **kwargs):
        """Initialize the 1D heat transfer dataset
        
        Arguments:
        
        **kwargs: dict
            Additional keyword arguments for dataset configuration
        """
        
        # Check if data files exist, generate if not
        data_dir = "/home/ubuntu/codebase/design-bench/design-bench/design_bench_data/heat_1d_data"
        if not os.path.exists(os.path.join(data_dir, HEAT_1D_FILES[0])):
            print("Heat 1D data not found, generating...")
            self.generate_data(num_samples=1000, save_dir=data_dir)
        
        # Create local disk resources pointing to generated data
        x_shards = []
        y_shards = []
        
        for filename in HEAT_1D_FILES:
            x_path = os.path.join(data_dir, filename)
            y_path = os.path.join(data_dir, filename.replace("-x-", "-y-"))
            
            x_shards.append(DiskResource(x_path, is_absolute=True))
            y_shards.append(DiskResource(y_path, is_absolute=True))
        
        # Initialize the dataset using the parent class
        super(Heat1DDataset, self).__init__(x_shards, y_shards, **kwargs)