import cv2
import numba
import numpy as np
from itertools import groupby


class PathOrderer:
    """
    Class to reorder vessel segments from a skeletonized mask into ordered paths.
    """
    def __init__(self, 
                 labeled_skel: np.ndarray, 
                 origin: tuple):
        """

        Args:
            labeled_skel (np.ndarray): Labeled vessel segment skeleton (vessel skeleton without branching points) array of shape (H,W)
            origin (tuple): Origin coordinates (r, c), usually the optic disc center.
        """
        self.skel: np.ndarray = labeled_skel
        self.origin: tuple = origin
    
    def reorder_coords(self) -> list[np.ndarray]:
        """Reorder coordinates of skeleton w.r.t. origin.
        
        Args:
            origin: Origin coordinates (r, c)
        
        Returns:
            list[np.ndarray]: List of ordered paths.
        """
        
        # Get endpoints
        endpoints = self._get_endpoints() # [[x_i,y_i,c_i],...]
        
        # Transform origin to match endpoints shape
        origin_array = np.array(self.origin).reshape(1,2)
        origin_array = np.repeat(origin_array, endpoints.shape[0], axis=0)
        
        # Sort endpoints by distance to origin
        distances = np.linalg.norm(endpoints[:,0:2] - origin_array, axis=1)
        idx = np.argsort(distances)
        endpoints = endpoints[idx]
        
        # Re-sort by channel without modifying the order of the other columns
        endpoints = endpoints[endpoints[:,2].argsort(stable=True)]
        labels = endpoints[:,2]
        
        # Non-repeating labels
        nrl = [num for num, group in groupby(labels) if sum(1 for _ in group) == 1]
        rl = [num for num, group in groupby(labels) if sum(1 for _ in group) > 1]
        idx_nrl = np.isin(labels, nrl)
        nr_endpoints = endpoints[idx_nrl, :2]
        endpoints = endpoints[~idx_nrl]
        
        # Sort segments with 2 endpoints
        start = endpoints[::2, :2]
        
        # Find paths
        paths = self._reorder_all_in_one_pass(start)
        
        # Reorganise paths to include non-repeating endpoints
        out = []
        for label in np.unique(labels):
            if label in nrl:
                # Add non-repeating endpoints
                out.append(nr_endpoints[np.where(nrl == label)[0]])
            else:
                # Add path
                out.append(paths[np.where(rl == label)[0][0]])
        
        return out
    
    def _get_endpoints(self) -> np.ndarray:
        """Get endpoint coordinates of skeleton.
        
        Args:
            skel: Skeleton of shape (H,W)
        Returns:
            coords: Coordinates of endpoints of shape (N,3) where N is the number of endpoints and the 3rd dimension is the pixel value.
        """
        
        # Define convolution kernel
        kernel = np.ones((3,3), dtype=np.uint8)
        kernel[1,1,...] = 10
        
        # Convolve skeleton with kernel
        _skel = self.skel.astype(bool).astype(np.float32) # Turn all nonzero values of skel to 1
        skel_conv = cv2.filter2D(_skel, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        
        # Keep only endpoints
        endpoints = (skel_conv >= 10) & (skel_conv <= 11) # 10 for segments of length 1, 11 for segments of length > 1
        endpoints = self.skel * endpoints # Keep the values of skel where endpoints is True
        
        # Get coordinates indexes (row, col)
        coords = np.nonzero(endpoints)
        
        # Add pixel value as 3rd coordinate
        coords = np.stack(coords, axis=1)
        coords = coords.reshape(-1,2)
        value = endpoints[coords[:,0], coords[:,1]]
        coords = np.concatenate([coords, value[:,np.newaxis]], axis=1)
        
        labels = np.ones((coords.shape[0], 1))
        coords = np.concatenate([coords, labels], axis=1)
            
        return coords
        
    def _reorder_all_in_one_pass(self, 
                                 start_coords: np.ndarray) -> list[np.ndarray]:
        """
        Wraps everything up:
        1) Extract coords from a single channel skeleton containing multiple disjoint linear paths.
        2) Build a global adjacency matrix for all coords.
        3) Find connected components, reorder each from start → end, return them all.
        """
        
        # 1) Extract all 'on' pixel coords
        coords = np.argwhere(self.skel > 0)  # shape (N, 2)
        start_idx = [np.where((coords == start).all(axis=1))[0][0] for start in start_coords]
        
        # 2) Build NxN adjacency with broadcasting
        adjacency = build_adjacency(coords)
        
        # 3) Reorder each connected component
        paths = self._reorder_multiple_paths(coords, adjacency, start_idx)
        
        return paths
    
    def _reorder_multiple_paths(self, 
                               coords: np.ndarray, 
                               adjacency: np.ndarray, 
                               start_indices: list[int]) -> list[np.ndarray]:
        """
        Reorder multiple 8-connected paths by only looping over the known start indices,
        rather than every pixel in `coords`. Each path is discovered by DFS from its start.
        
        Parameters:
        ----------
            coords (np.ndarray): (N, 2) array of all 'on' pixels.
            adjacency (np.ndarray): (N, N) boolean adjacency for 8-connected neighbors.
            start_indices (list[int]): Indices in `coords` for each path's known start pixel.
        
        Returns:
        -------
            list[np.ndarray]: A list of reordered path coordinates, one per start index.
        """
        visited = np.zeros(coords.shape[0], dtype=bool)
        paths = []

        for s in start_indices:
            if visited[s]:
                # Already discovered in a previous DFS
                continue
            # DFS to find all pixels in this connected component
            stack = [s]
            component_indices = []
            
            while stack:
                current = stack.pop()
                if visited[current]:
                    continue
                visited[current] = True
                component_indices.append(current)
                # Traverse neighbors
                neighbors = adjacency[current].nonzero()[0]
                for nbr in neighbors:
                    if not visited[nbr]:
                        stack.append(nbr)
            
            # Now `component_indices` holds all pixels in this path's connected component
            if len(component_indices) == 0:
                # No path discovered from this start
                continue

            component_coords = coords[component_indices]

            # For a strictly linear path, find endpoints or just reorder from the known start.
            # Build adjacency for the component alone:
            comp_row_diff = np.abs(component_coords[:, 0:1] - component_coords[:, 0])
            comp_col_diff = np.abs(component_coords[:, 1:2] - component_coords[:, 1])
            comp_adj = (comp_row_diff <= 1) & (comp_col_diff <= 1) & (
                ~((comp_row_diff == 0) & (comp_col_diff == 0))
            )

            # In a strictly linear path, exactly two endpoints will have 1 neighbor
            neighbor_counts = np.sum(comp_adj, axis=1)
            endpoints = np.where(neighbor_counts == 1)[0]
            if len(endpoints) == 2:
                local_start_idx = endpoints[0]
            else:
                # fallback if no distinct endpoints
                local_start_idx = 0
            
            path_sorted = self._dfs_order_path(component_coords, comp_adj, local_start_idx)
            paths.append(path_sorted)

        return paths
    
    def _dfs_order_path(self, coords_2d, adjacency_2d, start_idx):
        """
        Perform a simple DFS from start_idx to produce a linear ordering
        of coords_2d for a strictly linear path.
        """
        visited_local = np.zeros(coords_2d.shape[0], dtype=bool)
        order = []
        stack = [start_idx]
        while stack:
            current = stack.pop()
            if visited_local[current]:
                continue
            visited_local[current] = True
            order.append(current)
            # push neighbors
            nbrs = adjacency_2d[current].nonzero()[0]
            for nbr in nbrs:
                if not visited_local[nbr]:
                    stack.append(nbr)

        # Convert indices back to coordinates
        return coords_2d[order]
    
@numba.njit()
def build_adjacency(coords: np.ndarray) -> np.ndarray:
    N = coords.shape[0]
    adjacency = np.zeros((N, N), dtype=np.bool_)
    for i in numba.prange(N):
        for j in range(N):
            if i == j:
                continue
            # Check if coords are within 1 row and 1 col
            if (abs(coords[i,0] - coords[j,0]) < 2 and 
                abs(coords[i,1] - coords[j,1]) < 2):
                adjacency[i, j] = True
    return adjacency


