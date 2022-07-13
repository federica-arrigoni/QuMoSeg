import collections
import pickle
from pathlib import Path
from typing import Dict, List
import random
import shutil
import os

import numpy as np
import scipy

from matcher import compute_all_pair_matchings


class Splitter:
    '''
    Class for splitting the dataset in sub-datasets
    '''

    @classmethod
    def load(cls, directory: str):
        '''
        Loads a dataset from disk
        :param directory: Directory from which to load the dataset
        :return:
        '''

        with open(os.path.join(directory, "descriptors.pkl"), "rb") as file:
            descriptors_by_filename = pickle.load(file)
        with open(os.path.join(directory, "keypoints.pkl"), "rb") as file:
            keypoints_by_filename = pickle.load(file)
        with open(os.path.join(directory, "keypoints_ids.pkl"), "rb") as file:
            keypoints_ids_by_filename = pickle.load(file)

        return cls(descriptors_by_filename, keypoints_by_filename, keypoints_ids_by_filename)

    def __init__(self, descriptors_by_filename: Dict[str, np.ndarray], keypoints_by_filename: Dict[str, np.ndarray], keypoints_ids_by_filename: Dict[str, np.ndarray]):
        '''

        :param descriptors_by_filename: map from filenames to descriptors
        :param keypoints_by_filename: map from filenames to keypoints locations
        :param keypoints_ids_by_filename: map from filenames to motion id
        '''

        self.descriptors_by_filename = self.clone_dictionary(descriptors_by_filename)
        self.keypoints_by_filename = self.clone_dictionary(keypoints_by_filename)
        self.keypoints_ids_by_filename = self.clone_dictionary(keypoints_ids_by_filename)

        # (selected_points, 1) array with original ids of the points that were selected.
        # Initially all points are selected
        self.selected_keypoints = np.asarray(range(self.selected_points_count()), dtype=int)

    def split(self, images_count: int, points_per_motion: List[int]):
        '''
        Creates a filtered version of the dataset
        :param images_count: Number of images to retain
        :param points_per_motion: Number of points per motion to retain. Motions for which no point is specified are not retained
        :return:
        '''

        # Filters images
        self.select_images(images_count)
        # Filters motions
        self.select_motions(len(points_per_motion))
        # Filters points for each motion
        self.select_points(points_per_motion)

    def save(self, output_path: str, original_image_directory: str):

        Path(output_path).mkdir(parents=True)

        # Computs matches
        keypoints_matches = compute_all_pair_matchings(self.descriptors_by_filename)

        # Saves the data
        with open(os.path.join(output_path, "descriptors.pkl"), "wb") as file:
            pickle.dump(self.descriptors_by_filename, file)
        with open(os.path.join(output_path, "keypoints.pkl"), "wb") as file:
            pickle.dump(self.keypoints_by_filename, file)
        with open(os.path.join(output_path, "keypoints_ids.pkl"), "wb") as file:
            pickle.dump(self.keypoints_ids_by_filename, file)
        with open(os.path.join(output_path, "keypoint_matches.pkl"), "wb") as file:
            pickle.dump(keypoints_matches, file)
        with open(os.path.join(output_path, "original_keypoint_ids.pkl"), "wb") as file:
            pickle.dump(self.selected_keypoints, file)

        scipy.io.savemat(os.path.join(output_path, "descriptors.mat"), {os.path.basename(k).split(".")[0] + "_descriptors": v for k, v in self.descriptors_by_filename.items()})
        scipy.io.savemat(os.path.join(output_path, "keypoints.mat"), {os.path.basename(k).split(".")[0] + "_keypoints": v for k, v in self.keypoints_by_filename.items()})
        scipy.io.savemat(os.path.join(output_path, "keypoints_ids.mat"), {os.path.basename(k).split(".")[0] + "_keypoints_ids": v for k, v in self.keypoints_ids_by_filename.items()})
        scipy.io.savemat(os.path.join(output_path, "keypoint_matches.mat"), {"keypoint_matches": keypoints_matches})
        scipy.io.savemat(os.path.join(output_path, "original_keypoint_ids.mat"), {"original_keypoint_ids": self.selected_keypoints})

        # Copies the images to the new directory
        for image_name in self.descriptors_by_filename:
            original_image_name = os.path.join(original_image_directory, image_name)
            target_image_name = os.path.join(output_path, image_name)

            shutil.copy(original_image_name, target_image_name)

    def selected_points_count(self) -> int:

        return list(self.descriptors_by_filename.values())[0].shape[0]

    def clone_dictionary(self, dictionary):
        '''
        Clones a dictioanry making a deep copy of numpy arrays
        :param dictionary:
        :return:
        '''

        cloned_dictionary = {k: np.copy(v) for k, v in dictionary.items()}
        return cloned_dictionary

    def filter_by_name(self, dictionary, allowed_names: List[str]):
        filtered_dictionary = {k: v for k, v in dictionary.items() if k in allowed_names}
        return filtered_dictionary

    def filter_by_mask(self, dictionary, mask: np.ndarray):
        filtered_dictionary = {k: v[mask] for k, v in dictionary.items()}
        return filtered_dictionary

    def filter_by_ids(self, dictionary, ids: List[int]):
        filtered_dictionary = {k: v[ids] for k, v in dictionary.items()}
        return filtered_dictionary

    def select_images(self, images_count: int):
        '''
        Randomly selects a subset of images. Drops non selected images.

        :param images_count: Number of images to select
        :return:
        '''

        if len(self.descriptors_by_filename) < images_count:
            raise Exception(f"Selection of {images_count} images was asked, but only {len(self.descriptors_by_filename)} images are present")

        selected_images = list(self.descriptors_by_filename)
        random.shuffle(selected_images)
        selected_images = selected_images[:images_count]

        # Keeps only the allowed images
        self.descriptors_by_filename = self.filter_by_name(self.descriptors_by_filename, selected_images)
        self.keypoints_by_filename = self.filter_by_name(self.keypoints_by_filename, selected_images)
        self.keypoints_ids_by_filename = self.filter_by_name(self.keypoints_ids_by_filename, selected_images)

    def select_motions(self, motions_count) -> List[int]:
        '''
        Randomly selects a subset of motions.

        :param motions_count: Number of motions to select
        :return: list of ids of the selected motions
        '''

        # Number of motions is the maximum id + 1
        original_motions_count = int(np.max(list(self.keypoints_ids_by_filename.values())[0])) + 1

        if original_motions_count < motions_count:
            raise Exception(f"Selection of {motions_count} motions was asked, but only {original_motions_count} motions are present")

        selected_motions = list(range(original_motions_count))
        random.shuffle(selected_motions)
        selected_motions = selected_motions[:motions_count]

        # Sorts the motions
        selected_motions.sort()

        # Computes the mask of points that are in one of the motions to select
        motion_ids = list(self.keypoints_ids_by_filename.values())[0]
        selection_mask = np.zeros_like(motion_ids, dtype=np.bool)
        for current_motion in selected_motions:
            selection_mask = np.logical_or(motion_ids == current_motion, selection_mask)

        # Keeps only the allowed motions
        self.descriptors_by_filename = self.filter_by_mask(self.descriptors_by_filename, selection_mask)
        self.keypoints_by_filename = self.filter_by_mask(self.keypoints_by_filename, selection_mask)
        self.keypoints_ids_by_filename = self.filter_by_mask(self.keypoints_ids_by_filename, selection_mask)
        self.selected_keypoints = self.selected_keypoints[selection_mask]

        # Renames the ids
        for target_idx, original_idx in enumerate(selected_motions):
            for current_keypoint_ids in self.keypoints_ids_by_filename.values():
                current_keypoint_ids[current_keypoint_ids == original_idx] = target_idx

    def select_points(self, points_count: List[int]):
        '''
        Selects a number of points for each motion
        :param points_count: List of points to select for each motion
        :return:
        '''

        # Counts the number of points in each motion
        motion_ids = list(self.keypoints_ids_by_filename.values())[0]
        keypoints_counts = collections.Counter(motion_ids)

        motions_count = len(keypoints_counts)
        if len(points_count) != motions_count:
            raise Exception(f"Must specify a number of points for each motion. {motions_count} motions were detected, specified points per motion {points_count}")

        # Maps from motion index to starting position of the points in that motion
        motion_start_index = [0]
        for motion_idx in range(motions_count - 1):
            motion_start_index.append(motion_start_index[-1] + keypoints_counts[motion_idx])

        selected_indexes = [] # All the points to be selected
        for motion_idx, (current_point_count, current_motion_start_index) in enumerate(zip(points_count, motion_start_index)):

            if current_point_count > keypoints_counts[motion_idx]:
                raise Exception(f"The number of requested points to sample for motion {motion_idx} is {current_point_count}, but the motion contains only {keypoints_counts[motion_idx]} points")

            # Indexes of points for the current motion
            point_indexes = list(range(current_motion_start_index, current_motion_start_index + keypoints_counts[motion_idx]))
            # Selects the points
            random.shuffle(point_indexes)
            selected_indexes.extend(point_indexes[:current_point_count])
        selected_indexes.sort()

        # Keeps only the selected points
        self.descriptors_by_filename = self.filter_by_ids(self.descriptors_by_filename, selected_indexes)
        self.keypoints_by_filename = self.filter_by_ids(self.keypoints_by_filename, selected_indexes)
        self.keypoints_ids_by_filename = self.filter_by_ids(self.keypoints_ids_by_filename, selected_indexes)
        self.selected_keypoints = self.selected_keypoints[selected_indexes]

