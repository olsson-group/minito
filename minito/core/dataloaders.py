import torch
import numpy as np

class BaseDensity:
    def __init__(self, std, translation_invariant_subspace=False):
        self.standard_deviation = std
        self.translation_invaraiant_subspace = translation_invariant_subspace
    def sample_as(self, batch):
        if self.translation_invaraiant_subspace:
            assert NotImplementedError()
        else:
            return torch.normal(0, 1, size=batch.shape, device=batch.device)*self.standard_deviation 


class NumpyDataBaseCached:
    def __init__(self, path, uniform=False):
        self.path = path
        self.uniform = uniform
        npz = np.load(self.path)
        
        all_x = []
        all_betas = []
        traj_lens = []
        
        for k in sorted(npz.keys(), key=float):
            data = npz[k]
            beta_val = float(k)
            n_frame, n_traj = data.shape[0], data.shape[1]
            
            traj_lens.extend([n_frame] * n_traj)
            
            tensor_data = torch.from_numpy(data).float()
            if tensor_data.ndim == 2:
                tensor_data = tensor_data.unsqueeze(-1)
            
            flat_x = tensor_data.transpose(0, 1).reshape(-1, tensor_data.shape[-1])
            all_x.append(flat_x)
            
            betas = torch.full((flat_x.shape[0],), beta_val, dtype=torch.float32)
            all_betas.append(betas)

        self.traj_lens = traj_lens
        self.traj_boundaries = np.cumsum([0] + self.traj_lens)

        self.x_data = torch.cat(all_x, dim=0)
        self.beta_data = torch.cat(all_betas, dim=0)
        
        npz.close()

    def __len__(self):
        return self.x_data.shape[0]

    def __getitem__(self, index):
        return {
            "x": self.x_data[index],
            "beta": self.beta_data[index],
            "index": index
        }
        
    def get_trajectory_info(self, index):
        """Helper to find which trajectory a global index belongs to"""
        traj_idx = np.searchsorted(self.traj_boundaries, index, side="right") - 1
        frame_within_traj = index - self.traj_boundaries[traj_idx]
        return traj_idx, frame_within_traj

class NumpyDataBase:
    """
    Numpy dataset that loads data on-the-fly from npz file.
    Slow -- use for large datasets that do not fit in memory.
    Consider implementing something based on hdf5 or LMDB with memory mapping.
    """

    def __init__(self, path, uniform=False):
        self.path = path
        self.uniform = uniform
        self.npz = np.load(self.path)
        
        # Ensure all keys are numeric
        assert all(self._is_float(k) for k in self.npz.keys())

        self.traj_lens = []
        self.keys = []
        self.local_traj_indices = [] # Track which column to pick within the key

        for k in self.npz.keys():
            data = self.npz[k]
            n_frame, n_traj = data.shape[0], data.shape[1]
            
            self.traj_lens.extend([n_frame] * n_traj)
            self.keys.extend([k] * n_traj)
            # Store the index of the trajectory within THIS key
            self.local_traj_indices.extend(list(range(n_traj)))

        self.traj_boundaries = np.cumsum([0] + self.traj_lens)

    def _is_float(self, val):
        try:
            float(val)
            return True
        except ValueError:
            return False

    def __len__(self):
        return self.traj_boundaries[-1]

    def __getitem__(self, index):
        traj_idx = np.searchsorted(self.traj_boundaries, index, side="right") - 1
        config_idx = index - self.traj_boundaries[traj_idx]
        key = self.keys[traj_idx]
        local_col = self.local_traj_indices[traj_idx]
        config = self.npz[key][config_idx, local_col]
        x = torch.as_tensor(config, dtype=torch.float32)
        if x.ndim == 0:
            x = x.unsqueeze(0)
        return {"x": x.unsqueeze(0), "index": index, "beta": torch.tensor(float(key), dtype=torch.float32)}  # unsqueeze to add atom dimension

class LaggedDatasetMixin:
    def __init__(self, max_lag, fixed_lag=False, transform=None, uniform=False):
        #if max_lag is not float, is provided per molecule and we need create a max_lag per this dataset datapoint
        self.preprocess(max_lag) #this can be max(max_lag) for now
        self.fixed_lag = fixed_lag
        self.transform = transform if transform is not None else lambda x: x
        self.uniform = uniform
        self.base_density = BaseDensity(std=1.0) #removed name-magling to be able to use from hygher level datasets

    def preprocess(self, max_lag):
        #print("Preprocessing  LaggedDatasetMixin ... ")
        if isinstance(max_lag, (float, int)):  # this should be made more robust
            max_lag = [max_lag] * (len(self.traj_boundaries) - 1)
        max_lag = np.array(max_lag, dtype=int)
        total_size = sum(
            max(0, end - start - max_lag[i_mol])
            for i_mol, (start, end) in enumerate(zip(self.traj_boundaries[:-1], self.traj_boundaries[1:]))
        )
        data0_idxs = np.empty(total_size, dtype=int)
        new_max_lag = np.empty(total_size, dtype=int)
        lag_traj_boundaries = [0]
        idx = 0
        for i_mol, (start, end) in enumerate(zip(self.traj_boundaries[:-1], self.traj_boundaries[1:])):
            traj_len = end - start
            non_lagged_length = max(0, traj_len - max_lag[i_mol])
            data0_idxs[idx : idx + non_lagged_length] = np.arange(start, start + non_lagged_length)
            new_max_lag[idx : idx + non_lagged_length] = max_lag[i_mol]
            lag_traj_boundaries.append(lag_traj_boundaries[-1] + non_lagged_length)
            idx += non_lagged_length

        self.lag_traj_boundaries = np.array(lag_traj_boundaries, dtype=int)
        self.data0_idx = data0_idxs
        self.max_lag = new_max_lag

    def __getitem__(self, idx):
        max_lag_item = self.max_lag[idx]
        if self.fixed_lag:
            lag = max_lag_item 
        elif self.uniform:
            lag = np.random.randint(1, max_lag_item+1)
        else:
            log_lag = np.random.uniform(0, np.log(max_lag_item+1))
            lag = int(np.floor(np.exp(log_lag)))

        data0_idx = self.data0_idx[idx]
        data0 = super().__getitem__(data0_idx)
        datat = super().__getitem__(data0_idx + lag)

        base_sample = self.base_density.sample_as(datat['x'])
  
        base, target = base_sample, datat['x']

        datat['x'] = target
        datat['xbase'] = base

        item = {"cond": data0, "target": datat, "lag": torch.tensor(lag, dtype=torch.long) }
        return self.transform(item)     

    def __len__(self):
        return len(self.data0_idx)
    

class LaggedNumpyData(LaggedDatasetMixin, NumpyDataBase):
    def __init__(self, path, max_lag, fixed_lag=False, transform=None, uniform=False):
        NumpyDataBase.__init__(self, path=path)
        LaggedDatasetMixin.__init__(self, max_lag=max_lag, fixed_lag=fixed_lag, transform=transform)

    def __getitem__(self, idx):
        item = LaggedDatasetMixin.__getitem__(self, idx)
        return item

class LaggedNumpyDataInMemory(LaggedDatasetMixin, NumpyDataBaseCached):
    def __init__(self, path, max_lag, fixed_lag=False, transform=None, uniform=False):
        NumpyDataBaseCached.__init__(self, path=path)
        LaggedDatasetMixin.__init__(self, max_lag=max_lag, fixed_lag=fixed_lag, transform=transform)

    def __getitem__(self, idx):
        item = LaggedDatasetMixin.__getitem__(self, idx)
        return item






