# vim: expandtab:ts=4:sw=4
from __future__ import absolute_import
import numpy as np
from . import kalman_filter
from . import linear_assignment
from . import iou_matching
from .track import Track


class Tracker:
    """
    This is the multi-target tracker.

    Parameters
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        A distance metric for measurement-to-track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.

    Attributes
    ----------
    metric : nn_matching.NearestNeighborDistanceMetric
        The distance metric used for measurement to track association.
    max_age : int
        Maximum number of missed misses before a track is deleted.
    n_init : int
        Number of frames that a track remains in initialization phase.
    kf : kalman_filter.KalmanFilter
        A Kalman filter to filter target trajectories in image space.
    tracks : List[Track]
        The list of active tracks at the current time step.

    """

    def __init__(self, session, model, max_appearance_distance=1.0, max_iou_distance=0.5, max_age=30, n_init=15):
        self.model = model
        self.sess = session
        self.max_appearance_distance = max_appearance_distance
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        self.kf = kalman_filter.KalmanFilter()
        self.tracks = []
        self._next_id = 1

    def predict(self):
        """Propagate track state distributions one time step forward.

        This function should be called once every time step, before `update`.
        """
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Perform measurement update and track management.

        Parameters
        ----------
        detections : List[deep_sort.detection.Detection]
            A list of detections at the current time step.

        """
        # Run matching cascade.
        matches, unmatched_tracks, unmatched_detections = \
            self._match(detections)

        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections):
        def gated_metric(tracks, detections, track_indices, detection_indices):
            detection_features = np.array([detections[i].feature for i in detection_indices])
            target_features = []
            for target_index in track_indices:
                features = np.array(tracks[target_index].features)
                if features.shape[0] < 6:
                    print ' It should not happen that confirmed target has less than 6 features'
                else:
                    target_features.append(features[features.shape[0] - 6 : , :])

            target_features = np.array(target_features)
            logits = self.sess.run([self.model.similarity_probs], feed_dict={
                self.model.detection_features_placeholder:detection_features,
                self.model.target_features_placeholder:target_features
                })

            num_targets = len(track_indices)
            num_detections = len(detection_indices)
            num_total_samples = num_targets * num_detections
            cost_matrix = np.zeros((num_detections, num_targets))
            logits = np.squeeze(np.array(logits), axis=0)

            for i in range(num_total_samples):
                target_idx = i % num_targets
                detection_idx = i % num_detections
                cost_matrix[detection_idx, target_idx] = logits[i, 1]

            cost_matrix = np.transpose(cost_matrix)
            cost_matrix = linear_assignment.gate_cost_matrix(
                self.kf, cost_matrix, tracks, detections, track_indices,
                detection_indices)
            return cost_matrix

        # Split track set into confirmed and unconfirmed tracks.
        confirmed_tracks = [
            i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [
            i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # # Associate confirmed tracks using appearance features.
        # matches_a, unmatched_tracks_a, unmatched_detections = \
        #     linear_assignment.matching_cascade(
        #         gated_metric, self.max_appearance_distance, self.max_age,
        #         self.tracks, detections, confirmed_tracks)

        # a_matches_track_id = [self.tracks[i].track_id for i, t in matches_a]
        # print '     Track ids matched by apperance: ' + str(a_matches_track_id)

        # # Associate remaining detections together with unconfirmed tracks using IOU.
        # iou_track_candidates = unconfirmed_tracks + [
        #     k for k in unmatched_tracks_a if
        #     self.tracks[k].time_since_update == 1]
        # matches_b, unmatched_tracks_b, unmatched_detections = \
        #     linear_assignment.min_cost_matching(
        #         iou_matching.iou_cost, self.max_iou_distance, self.tracks,
        #         detections, iou_track_candidates, unmatched_detections)

        # b_matches_track_id = [self.tracks[i].track_id for i, t in matches_b]
        # print '     Track ids matched by iou matching: ' + str(b_matches_track_id)

        matches_b, unmatched_tracks_b, unmatched_detections = \
            linear_assignment.min_cost_matching(
                iou_matching.iou_cost, self.max_iou_distance, self.tracks,
                detections)

        a_track_candidates = [
            k for k in unmatched_tracks_b if
            self.tracks[k].time_since_update <= 15 and len(self.tracks[k].features) >= 6]

        print '     Track index candidates for apperance: ' + str(a_track_candidates)
        print '     Detection index candidates for apperance: ' + str(unmatched_detections)
        matches_a, unmatched_tracks_a, unmatched_detections = \
            linear_assignment.matching_cascade(
                gated_metric, self.max_appearance_distance, self.max_age,
                self.tracks, detections, a_track_candidates, unmatched_detections)

        b_matches_track_id = [self.tracks[i].track_id for i, t in matches_b]
        print '     Track ids matched by iou matching: ' + str(b_matches_track_id)

        a_matches_track_id = [self.tracks[i].track_id for i, t in matches_a]
        print '     Track ids matched by apperance: ' + str(a_matches_track_id)

        matches = matches_a + matches_b
        track_indices = [i for i, t in enumerate(self.tracks)]
        unmatched_tracks = list(set(track_indices) - set(k for k, _ in matches))

        unmatched_track_ids = [self.tracks[i].track_id for i in unmatched_tracks]
        print '     All unmatched tracks in current frame: ' + str(unmatched_track_ids)
        
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(
            mean, covariance, self._next_id, self.n_init, self.max_age,
            detection.feature))
        self._next_id += 1
