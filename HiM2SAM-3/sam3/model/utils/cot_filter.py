import torch
import sys
import cv2
import numpy as np
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
import time
import math
import torch.nn.functional as F
from random import randint


class torch_GaussianKDE:
    def __init__(self, data, bandwidth=None):
        """
        Initialize the Gaussian Kernel Density Estimator.
        Args:
            data (torch.Tensor): Input data points, shape (n_samples, n_features).
            bandwidth (float, optional): Bandwidth for the Gaussian kernel. Defaults to Silverman's rule if not provided.
        """
        self.data = data  # (n_samples, n_features)
        self.n_samples, self.n_features = data.shape
        
        self.bandwidth = bandwidth
    def calculate_density(self,pred_tracks, kernel_size=3):
       
        N = pred_tracks.shape[0]
        distances = torch.cdist(pred_tracks, pred_tracks)  #  (N, N)
        density = (distances < kernel_size).float().sum(dim=1) 
        return density

    def dynamic_bandwidth(self, density, min_bandwidth=1.0, max_bandwidth=5.0):
    
        density_min, density_max = density.min(), density.max()
        normalized_density = (density - density_min) / (density_max - density_min + 1e-6)
        bandwidths = max_bandwidth - normalized_density * (max_bandwidth - min_bandwidth)
        return bandwidths
    def evaluate(self, points):
        
        if self.bandwidth is None:
            density = self.calculate_density(self.data, kernel_size=3)
    
            bandwidths = self.dynamic_bandwidth(density, min_bandwidth=1.0, max_bandwidth=9.0)
        else:
            bandwidths = self.bandwidth
        points = points.unsqueeze(1)  # (n_points, 1, n_features)
        data = self.data.unsqueeze(0)  # (1, n_samples, n_features)

        # Compute pairwise distances
        diff = points - data  # (n_points, n_samples, n_features)
        dists = torch.sum(diff**2, dim=-1)  # (n_points, n_samples)

        # Apply Gaussian kernel
        kernel_vals = torch.exp(-0.5 * dists / bandwidths**2)  # (n_points, n_samples)
        kernel_vals = kernel_vals/ (math.sqrt(2 * math.pi) * bandwidths) ** self.n_features

        # Average over samples to compute density
        density = kernel_vals.mean(dim=-1)  # (n_points,)
        return density
    
    
class RVCotFilter():
    def __init__(self, pth_path=None,
                 device='cuda:0',
                 box_iou=False, 
                 mirrow_padding=False,
                 frame_far = 5,
                 iouweight = 0.5,
                 **args):
        
        # initialize cotracker3 using torch.hub
        self.cot_model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")
        self.cot_model.to(device).to(dtype=torch.float32).eval()
        self.num_point_per_mask = 100  # limited by cuda and fps
        self.box_iou = box_iou 
        self.device = device
        self.mirrow_padding = mirrow_padding
        # pre-make the grid 
        grid_size = 1024
        x = torch.linspace(0, grid_size, grid_size)
        y = torch.linspace(0, grid_size, grid_size)
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        self.positions = torch.stack([yy.ravel(), xx.ravel()], dim=1).to(self.device)
        
        self.max_pre_iou = -1

        h, w = 1024,1024
        grid_x, grid_y = torch.meshgrid(
            torch.arange(0, w, 4, device=device),
            torch.arange(0, h, 4, device=device),
            indexing='xy'
        )
        self.grid_points = torch.stack([grid_y.flatten(), grid_x.flatten()], dim=1).cpu()  # (N, 2)
        self.ct = 0
        self.frame_far = frame_far
        self.iouweight = iouweight
        self.debugflag = True

    def predict(self,  frame_idx, inference_state, cur_masks,iou_aggregation_method = 'mean', debug_flag=False,sample_count=100):
        # merge the cur_masks, reverse cot tracking, return iou for per historical frame
        assert(iou_aggregation_method in ['farest','mean']) # farest: the farest frame from current frame
        img_mean=torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1).to(self.device)
        img_std=torch.tensor((0.229, 0.224, 0.225)).view(1,3,1,1).to(self.device)
        assert(frame_idx is not None)
        #cot inference frame indx
        N = cur_masks.shape[1] #num of masks
        cur_masks_bin = self.prepare_masks(cur_masks)
        #the farest frame to recap
        frame_far = max(frame_idx - self.frame_far,0) #track last 5 frames
        union_mask = cur_masks_bin[0]
        for mask in cur_masks_bin[1:]:
            union_mask = union_mask | mask
        mask_area = union_mask.sum()
        if mask_area<100:
            return torch.zeros([3]).to(device=self.device)
        
        # construct inference frames : reversely
        feed_cot_video_frames = torch.stack([inference_state['cot_cache_frames'][i][0]  \
                    for i in range(frame_idx-1, frame_far-1,-1) ],dim=0)
        if self.mirrow_padding:
            reverse_frames = feed_cot_video_frames.flip(dims=[0])[1:,:,:]
            feed_cot_video_frames = torch.cat([feed_cot_video_frames,reverse_frames],dim=0)
        feed_cot_video_frames = feed_cot_video_frames*img_std+img_mean # reverse to 0-1
        feed_cot_video_frames = feed_cot_video_frames.clamp_min(0).clamp_max(1)

        r = int(torch.sqrt(mask_area/torch.pi)/3.5/1.2) #
        r = min(21,r)
        r = max(5,r)
        k = r
        bw = r-2
        bw = int(bw*0.8)
        sparse = mask_area > 150000

        points = self.fps_sample_from_mask(union_mask[0], sample_count,errod=True, kernel=k,sparse=sparse)
        if points is None:
            return torch.zeros([3]).to(self.device) # TODO 
        #### debug
        x, y = points[:, 1].int(), points[:, 2].int()
        x.clamp_(0, 1023)
        y.clamp_(0, 1023)
        points_mask_mapping = torch.stack([cur_masks_bin[0][0][y, x], 
                                        cur_masks_bin[1][0][y, x],
                                        cur_masks_bin[2][0][y,x]], dim=1)
        # reverse track sampled points
        pred_tracks, pred_visibility = self.cot_model( feed_cot_video_frames.unsqueeze(0).to(dtype=torch.float32),
                points.unsqueeze(0).to(dtype=torch.float32))
        # # points iou with historical frame masks
        device = torch.device("cuda") 

        historical_masks = []
        #inreverse idx
        for f in range(frame_idx-1, frame_far-1, -1):
            if f ==0:
                his_mask = inference_state['output_dict']['cond_frame_outputs'][0]['pred_masks']
            else:
                his_mask = inference_state['output_dict']['non_cond_frame_outputs'][f]['pred_masks']
            his_mask = his_mask.to(self.device)
            his_mask =  torch.nn.functional.interpolate(
                his_mask,
                size=(1024, 1024),
                align_corners=False,
                mode="bilinear",
                antialias=True,  # use antialias for downsampling
            )
            historical_masks.append(his_mask)

        cot_iou_score = self.iou_score_point_mask(pred_tracks, pred_visibility,
                                                            historical_masks,points_mask_mapping, bw=bw,)
        #aggregate result
        if iou_aggregation_method=='mean' or self.mirrow_padding:
            res = torch.mean(cot_iou_score.float(), dim = 1)
        elif iou_aggregation_method=='farest':
            res =  cot_iou_score.select(dim=1, index=-1)
        return res 
        
    
    def prepare_masks(self, masks):
        B = masks.shape[0]
        assert(B==1),"Hard code, assume bs=1"
        N = masks.shape[1]
        masks_list = []
        for i in range(N):
            cur_masks = masks[0,i,:,:].unsqueeze(0)
            binary_mask = (cur_masks > 0.).int()
            masks_list.append(binary_mask)
        return masks_list
        
    def fps_sample_from_mask(self, binary_mask, sample_count=50,errod=True, kernel = None, sparse=False):
        if kernel is None:
            k=7
        else:
            k = kernel
        if errod:
            binary_mask = F.max_pool2d(-binary_mask[None][None].to(torch.float32),  kernel_size=k,stride=1, padding=k//2)[0][0]
            binary_mask =  -binary_mask.clamp(max=0)
        if binary_mask.sum()<10:
            return None
        y_coords, x_coords = torch.where(binary_mask) 
        if sparse:
            y_coords = y_coords[::6]
            x_coords = x_coords[::6]

        points = torch.stack([  x_coords,y_coords],axis=1)
        zeros = torch.zeros([sample_count,1]).to(self.device)
        fps_points,_ = self.sample_farthest_points_naive(points=points[None], K = sample_count)
        p = torch.concat([zeros, fps_points[0]],dim=1)
        return p
    

    def sample_farthest_points_naive(
        self,
        points: torch.Tensor,
        lengths  = None,
        K = 50,
        random_start_point: bool = False,
    ):
        
        # FPS method from pytorch.3d
        # Iterative farthest point sampling algorithm [1] to subsample a set of
        # K points from a given pointcloud. At each iteration, a point is selected
        # which has the largest nearest neighbor distance to any of the
        # already selected points.

        # Farthest point sampling provides more uniform coverage of the input
        # point cloud compared to uniform random sampling.
        # [1] Charles R. Qi et al, "PointNet++: Deep Hierarchical Feature Learning
        #       on Point Sets in a Metric Space", NeurIPS 2017.
        """
        Args:
        points: (N, P, D) array containing the batch of pointclouds
        lengths: (N,) number of points in each pointcloud (to support heterogeneous
            batches of pointclouds)
        K: samples required in each sampled point cloud (this is typically << P). If
            K is an int then the same number of samples are selected for each
            pointcloud in the batch. If K is a tensor is should be length (N,)
            giving the number of samples to select for each element in the batch
        random_start_point: bool, if True, a random point is selected as the starting
            point for iterative sampling.

        Returns:
            selected_points: (N, K, D), array of selected values from points. If the input
                K is a tensor, then the shape will be (N, max(K), D), and padded with
                0.0 for batch elements where k_i < max(K).
            selected_indices: (N, K) array of selected indices. If the input
                K is a tensor, then the shape will be (N, max(K), D), and padded with
                -1 for batch elements where k_i < max(K).`
        """
        N, P, D = points.shape
        device = points.device

        # Validate inputs
        if lengths is None:
            lengths = torch.full((N,), P, dtype=torch.int64, device=device)
        else:
            if lengths.shape != (N,):
                raise ValueError("points and lengths must have same batch dimension.")
            if lengths.max() > P:
                raise ValueError("Invalid lengths.")

        # TODO: support providing K as a ratio of the total number of points instead of as an int
        if isinstance(K, int):
            K = torch.full((N,), K, dtype=torch.int64, device=device)
        elif isinstance(K, list):
            K = torch.tensor(K, dtype=torch.int64, device=device)

        if K.shape[0] != N:
            raise ValueError("K and points must have the same batch dimension")

        # Find max value of K
        max_K = torch.max(K)

        # List of selected indices from each batch element
        all_sampled_indices = []

        for n in range(N):
            # Initialize an array for the sampled indices, shape: (max_K,)
            sample_idx_batch = torch.full(
                # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
                #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
                (max_K,),
                fill_value=-1,
                dtype=torch.int64,
                device=device,
            )

            # Initialize closest distances to inf, shape: (P,)
            # This will be updated at each iteration to track the closest distance of the
            # remaining points to any of the selected points
            closest_dists = points.new_full(
                # pyre-fixme[6]: For 1st param expected `Union[List[int], Size,
                #  typing.Tuple[int, ...]]` but got `Tuple[Tensor]`.
                (lengths[n],),
                float("inf"),
                dtype=torch.float32,
            )

            # Select a random point index and save it as the starting point
            # pyre-fixme[6]: For 2nd argument expected `int` but got `Tensor`.
            selected_idx = randint(0, lengths[n] - 1) if random_start_point else 0
            sample_idx_batch[0] = selected_idx

            # If the pointcloud has fewer than K points then only iterate over the min
            # pyre-fixme[6]: For 1st param expected `SupportsRichComparisonT` but got
            #  `Tensor`.
            # pyre-fixme[6]: For 2nd param expected `SupportsRichComparisonT` but got
            #  `Tensor`.
            k_n = min(lengths[n], K[n])
            # Iteratively select points for a maximum of k_n
            for i in range(1, k_n):
                # Find the distance between the last selected point
                # and all the other points. If a point has already been selected
                # it's distance will be 0.0 so it will not be selected again as the max.
                dist = points[n, selected_idx, :] - points[n, : lengths[n], :]
                # pyre-fixme[58]: `**` is not supported for operand types `Tensor` and
                #  `int`.
                dist_to_last_selected = (dist**2).sum(-1)  # (P - i)

                # If closer than currently saved distance to one of the selected
                # points, then updated closest_dists
                closest_dists = torch.min(dist_to_last_selected, closest_dists)  # (P - i)

                # The aim is to pick the point that has the largest
                # nearest neighbour distance to any of the already selected points
                selected_idx = torch.argmax(closest_dists)
                sample_idx_batch[i] = selected_idx

            # Add the list of points for this batch to the final list
            all_sampled_indices.append(sample_idx_batch)

        all_sampled_indices = torch.stack(all_sampled_indices, dim=0)

        # Gather the points
        all_sampled_points = self.masked_gather(points, all_sampled_indices)

        # Return (N, max_K, D) subsampled points and indices
        return all_sampled_points, all_sampled_indices

    def masked_gather(self, points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """
        Helper function for torch.gather to collect the points at
        the given indices in idx where some of the indices might be -1 to
        indicate padding. These indices are first replaced with 0.
        Then the points are gathered after which the padded values
        are set to 0.0.

        Args:
            points: (N, P, D) float32 tensor of points
            idx: (N, K) or (N, P, K) long tensor of indices into points, where
                some indices are -1 to indicate padding

        Returns:
            selected_points: (N, K, D) float32 tensor of points
                at the given indices
        """

        if len(idx) != len(points):
            raise ValueError("points and idx must have the same batch dimension")

        N, P, D = points.shape

        if idx.ndim == 3:
            # Case: KNN, Ball Query where idx is of shape (N, P', K)
            # where P' is not necessarily the same as P as the
            # points may be gathered from a different pointcloud.
            K = idx.shape[2]
            # Match dimensions for points and indices
            idx_expanded = idx[..., None].expand(-1, -1, -1, D)
            points = points[:, :, None, :].expand(-1, -1, K, -1)
        elif idx.ndim == 2:
            # Farthest point sampling where idx is of shape (N, K)
            idx_expanded = idx[..., None].expand(-1, -1, D)
        else:
            raise ValueError("idx format is not supported %s" % repr(idx.shape))

        idx_expanded_mask = idx_expanded.eq(-1)
        idx_expanded = idx_expanded.clone()
        # Replace -1 values with 0 for gather
        idx_expanded[idx_expanded_mask] = 0
        # Gather points
        selected_points = points.gather(dim=1, index=idx_expanded)
        # Replace padded values
        selected_points[idx_expanded_mask] = 0.0
        return selected_points
    
    def iou_score_point_mask(self, pred_tracks, pred_visibility, historical_masks,points_mask_mapping,
                              only_visible=True, bw=None, ):
        # return a list of score
        
        if bw is not None:
            bw = bw
        else:
            bw = 4
        cot_ious = []
        self.ct = 0
        for i in range(points_mask_mapping.shape[1]): # 3 proposals
            cur_ious = []
            for t in range(len(historical_masks)):
                validpoints = pred_visibility[0,t,:] & points_mask_mapping[:,i] # viliad flag at time t, proposal i-th
                if  validpoints.sum()<10:
                    conf_mask = torch.zeros([1024,1024]).to(device=self.device)
                else:
                    if not self.mirrow_padding:
                        cur_pred_tracks = pred_tracks[0,t,:,:][validpoints.bool()]
                    else:
                        cur_pred_tracks = pred_tracks[:,-1-t,:,:][validpoints.bool()]
                    conf_mask = self.torch_gaussian_kernel(cur_pred_tracks,bandwidth=bw)
                cur_ious.append(self.nonbinary_iou(conf_mask ,historical_masks[t][0][0] ))

                self.ct += 1

            cot_ious.append(cur_ious)
        #ious from t , t-1 to t_far
        return torch.tensor(cot_ious).to(self.device)
    
    def torch_gaussian_kernel(self, points, bandwidth):
        "estimate conf mask from points"
    
        kde = torch_GaussianKDE(points,bandwidth=bandwidth)
        # Evaluate density
        grid_size=1024
        
        density = kde.evaluate(self.positions)

        density = density.reshape(grid_size, grid_size)

        thred = density[density>0].mean() +2*density[density>0].std()
        density = torch.clip(density,0, thred)
        density = density/density.max()
        return torch.tensor(density).to(self.device)

    def nonbinary_iou(self, mask1, mask2,):       
        # similar to dice loss:
        #mask1: soft point mask
        #mask2: 0-1 binary mask
        box_iou = self.box_iou
        mask1[mask1<0.4]=0

        mask2 = torch.tensor(mask2)
        mask2[mask2<0.15]=0
        if mask2.sum() < 0.1:
            return 0
        mask2 /= (mask2.max()+0.00001)
       
        numerator = 2 * (mask1 * mask2).sum()
        denominator = mask1.sum() + mask2.sum()
        iou = (numerator ) / (denominator + 0.01)

        if self.debugflag:
            print(self.iouweight , "rvcot IOU weight")
            self.debugflag = False 


        if box_iou:
        # Calculate bounding boxes
            def get_bounding_box(mask):
                mask = mask.view(1024, 1024)

                non_zero_indices = torch.argwhere(mask> 0.1)
                    #有效mask
                if len(non_zero_indices) == 0:
                    y_min, x_min,y_max, x_max = [0, 0, 0, 0]
                #initiate kf
                else:
                    y_min, x_min = non_zero_indices.min(dim=0).values
                    y_max, x_max = non_zero_indices.max(dim=0).values
                return x_min, y_min, x_max, y_max
            
            # Get bounding boxes for both masks
            x1_min, y1_min, x1_max, y1_max = get_bounding_box(mask1)
            x2_min, y2_min, x2_max, y2_max = get_bounding_box(mask2)
            
            # Intersection coordinates
            inter_x_min = max(x1_min, x2_min)
            inter_y_min = max(y1_min, y2_min)
            inter_x_max = min(x1_max, x2_max)
            inter_y_max = min(y1_max, y2_max)
            
            # Compute intersection area
            if inter_x_max > inter_x_min and inter_y_max > inter_y_min:
                inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
            else:
                inter_area = 0.
            
            # Compute areas of each box
            area1 = (x1_max - x1_min) * (y1_max - y1_min)
            area2 = (x2_max - x2_min) * (y2_max - y2_min)
            
            # Compute union area
            union_area = area1 + area2 - inter_area            
            # Compute box IoU
            box_iou = inter_area / union_area if union_area > 0 else 0.
            # print(iou, box_iou)
            return iou*self.iouweight + box_iou*(1-self.iouweight)
        return iou