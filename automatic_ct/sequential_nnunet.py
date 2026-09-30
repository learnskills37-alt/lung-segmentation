"""Use nnU-Net's official sequential predictor when multiprocessing is unavailable.

Uses sequential I/O and FP32 CPU window accumulation; retains pretrained weights,
preprocessing, window locations, mirroring, Gaussian weighting, and resampling.
"""


def enable_sequential_io():
    # Import TotalSegmentator first so its own compatibility patches are applied.
    import totalsegmentator.nnunet
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    import torch

    def sequential(self, list_of_lists_or_source_folder,
                   output_folder_or_list_of_truncated_output_files,
                   save_probabilities=False, overwrite=True,
                   num_processes_preprocessing=1, num_processes_segmentation_export=1,
                   folder_with_segs_from_prev_stage=None, num_parts=1, part_id=0,
                   use_cropped_logits_resampling=False):
        if num_parts != 1 or part_id != 0:
            raise ValueError('Sequential adapter supports one inference partition')
        return self.predict_from_files_sequential(
            list_of_lists_or_source_folder,
            output_folder_or_list_of_truncated_output_files,
            save_probabilities=save_probabilities, overwrite=overwrite,
            folder_with_segs_from_prev_stage=folder_with_segs_from_prev_stage,
            use_cropped_logits_resampling=use_cropped_logits_resampling)

    nnUNetPredictor.predict_from_files = sequential

    @torch.inference_mode()
    def fp32_windows(self, data, slicers, do_on_device=True):
        """Gaussian-weighted window aggregation in FP32 on CPU.

        Some restricted CPUs cannot initialize the mixed-FP16 kernel used by
        upstream accumulation. FP32 avoids that kernel without changing weights,
        patch locations, mirroring policy, or Gaussian weighting.
        """
        if self.device.type != 'cpu':
            raise ValueError('This compatibility adapter is CPU-only')
        data = data.to('cpu')
        shape = data.shape[1:]
        scores = torch.zeros((self.label_manager.num_segmentation_heads, *shape), dtype=torch.float32)
        counts = torch.zeros(shape, dtype=torch.float32)
        weight = (compute_gaussian(tuple(self.configuration_manager.patch_size),
                                  sigma_scale=1/8, value_scaling_factor=10,
                                  dtype=torch.float32, device=torch.device('cpu'))
                  if self.use_gaussian else 1.0)
        for index, location in enumerate(slicers):
            patch = data[location].unsqueeze(0).contiguous()
            prediction = self._internal_maybe_mirror_and_predict(patch)[0].float()
            scores[location] += prediction * weight
            counts[location[1:]] += weight
            print(f'CPU window {index+1}/{len(slicers)}', flush=True)
        if torch.any(counts == 0):
            raise RuntimeError('Sliding windows did not cover the full input')
        scores /= counts
        if not torch.isfinite(scores).all():
            raise RuntimeError('Non-finite output logits')
        return scores

    nnUNetPredictor._internal_predict_sliding_window_return_logits = fp32_windows
