import argparse
import concurrent.futures
import os

import numpy as np
import scipy


class DipDetector:
    def __init__(
        self,
        window_size,
        threshold,
        prominence,
        min_size,
        enrichment,
        color,
        threads,
        label,
    ):
        self.window_size = window_size
        self.threshold = threshold
        self.prominence = prominence

        self.min_size = min_size

        self.enrichment = enrichment

        self.color = color

        self.threads = threads
        self.label = label

    def smooth_methylation(self, methylation):
        # run data through savgol filtering
        methyl_frac_mod = np.array(methylation["fraction_modified"], dtype=float)
        methylation["savgol_frac_mod"] = scipy.signal.savgol_filter(
            x=methyl_frac_mod, 
            window_length=self.window_size, 
            polyorder=2, 
            mode='mirror'
        )
        methylation["savgol_frac_mod_dy"] = scipy.signal.savgol_filter(
            x=methyl_frac_mod, 
            window_length=self.window_size, 
            polyorder=2, 
            deriv=1,
            mode='mirror'
        )

        return methylation

    def detect_dips(self, methylation):
        data = np.array(methylation["savgol_frac_mod"], dtype=float)
        data_range = np.max(data) - np.min(data)

        height_threshold = np.mean(data)-(np.std(data)*self.threshold)          # calculate the height threshold
        prominence_threshold = self.prominence * data_range                     # calculate the prominence threshold

        if not self.enrichment:
            dips, _ = scipy.signal.find_peaks(
                -data,
                height=-height_threshold, 
                prominence=prominence_threshold,
                wlen=(len(data))
            )
        else:
            dips, _ = scipy.signal.find_peaks(
                data,
                height=height_threshold, 
                prominence=prominence_threshold,
                wlen=(len(data))
            )

        return dips

    def extend_dips(self, methylation, dips):
        data = np.array(methylation["savgol_frac_mod"], dtype=float)
        dy = np.array(methylation["savgol_frac_mod_dy"], dtype=float)
        median = np.median(data)

        mask = data > median if self.enrichment else data < median              # mask out invalid values
        prev = np.r_[False, mask[:-1]]
        next_ = np.r_[mask[1:], False]
        starts = np.flatnonzero(mask & ~prev)
        ends = np.flatnonzero(mask & ~next_)
        lefts = np.maximum(starts - 1, 0)                                       # safe finding of right/left index
        rights = np.minimum(ends + 1, data.size - 1)
        dips_arr = np.asarray(list(dips), dtype=int)                            # np array of dips
        idx = np.searchsorted(starts, dips_arr, side="right") - 1
        valid = (idx >= 0)
        valid &= mask[dips_arr]
        valid &= dips_arr <= ends[idx.clip(min=0)]                              # safe gather
        idx = idx[valid]
        dip_bounds = [(int(lefts[i]), int(rights[i])) for i in idx]

        # trim dip calls - set the edge to be where the slope is the most prominent, while within the bounds
        dip_bounds_adj = []
        for d, (l, r) in zip(dips, dip_bounds):
            l_adj = int( np.argmin(dy[range(l, d+1)]) + l )
            r_adj = int( np.argmax(dy[range(d, r+1)]) + d )
            dip_bounds_adj.append((l_adj, r_adj))

        return dip_bounds_adj

    def dip_detect_single_chromosome(self, region, methylation):
        # if the region has less CpG's than the window size do not process
        if len(methylation['starts']) < self.window_size:
            return ( region, {}, {}, {} )
        methylation_smoothed = self.smooth_methylation(methylation)

        dip_sites = self.detect_dips(methylation_smoothed)
        dip_idxs = self.extend_dips(methylation_smoothed, dip_sites)
        dips = self.filter_dips(methylation_smoothed, dip_idxs)

        return ( region, dips, methylation_smoothed)

    def dip_detect_all_chromosome(self, methylation_per_region, regions_per_chrom):
        dips_all_chroms, methylation_all_chroms = {}, {}

        regions = list(methylation_per_region.keys())
        if not regions:
            return dips_all_chroms, methylation_all_chroms

        if self.threads <= 1 or len(regions) == 1:
            for region in regions:
                region, mdrs, methylation_pvalues = self.centrodip_single_chromosome(
                    region, methylation_per_region[region]
                )
                dips_all_chroms[region] = mdrs
                methylation_all_chroms[region] = methylation_pvalues
            return dips_all_chroms, methylation_all_chroms

        with concurrent.futures.ProcessPoolExecutor(max_workers=self.threads) as executor:
            futures = {
                executor.submit(
                    self.centrodip_single_chromosome,
                    region,
                    methylation_per_region[region],
                ): region for region in regions
            }    

            for future in concurrent.futures.as_completed(futures):
                (    
                    region,
                    mdrs,
                    methylation_pvalues,
                ) = future.result()

                dips_all_chroms[region] = mdrs
                methylation_all_chroms[region] = methylation_pvalues

        return dips_all_chroms, methylation_all_chroms