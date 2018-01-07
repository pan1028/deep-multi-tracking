'''Deep tracker be used for JR.'''

import os
import numpy as np
import cv2
from cv2 import cv
import tensorflow as tf
import tensorflow.contrib.slim as slim
from deep_sort.tracker import Tracker
from deep_sort.detection import Detection
from deep_sort import nn_matching
from deep_sort.model import Model
from application_util import preprocessing
import sys
import rospy

def extract_image_patch(image, bbox, patch_shape):
    """Extract image patch from bounding box.
    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.
    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.
    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    image = cv2.resize(image, patch_shape[::-1])

    return image

def generate_detections(session, model, image, dets):
    """Generate detections with features.
    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    image :
	A BGR color image read by opencv
    dets
        A list of bounding boxes in format '(x, y, w, h)'
    """
    image_shape = 128, 64, 3
    image_patches = []
    for box in dets:
        patch = extract_image_patch(image, box[:4], image_shape[:2])
        if patch is None:
            print("WARNING: Failed to extract image patch: %s." % str(box))
            patch = np.random.uniform(0., 255., image_shape).astype(np.uint8)
        image_patches.append(patch)
    image_patches = np.asarray(image_patches)
    print '     detections shape is ' + str(image_patches.shape)

    features = session.run([model.features], feed_dict={
        model.image_patches_placeholder:image_patches
        })
    
    features = np.array(features)
    features = np.squeeze(features, axis=0)
    
    detections_out = [np.r_[(row, feature)] for row, feature in zip(dets, features)]
    return detections_out

def create_detections(detections, min_height=0):
    """Create detections for given frame index from the raw detection matrix.
    Parameters
    ----------
    detection_mat : ndarray
        Matrix of detections. The first 10 columns of the detection matrix are
        in the standard MOTChallenge detection format. In the remaining columns
        store the feature vector associated with each detection.
    min_height : Optional[int]
        A minimum detection bounding box height. Detections that are smaller
        than this value are disregarded.
    Returns
    -------
    List[tracker.Detection]
        Returns detection responses at given frame index.
    """
    detection_list = []
    for row in detections:
        bbox, confidence, feature = row[0:4], row[4], row[5:]
        if bbox[3] < min_height:
            continue
        detection_list.append(Detection(bbox, confidence, feature))
    return detection_list

class Deep_Tracker(object):
  def __init__(
    self, model_ckpts, n_init=15, max_age=45,
	min_confidence = 0.3, min_detection_height = 0, nms_max_overlap = 1.0):

    # Loading pretrained checkpoint
    self.model = Model()

    rospy.loginfo('model created....')
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    self.sess = tf.Session()
    
    saver = tf.train.Saver()
    ckpt = tf.train.get_checkpoint_state(model_ckpts)
    if ckpt is None:
        rospy.loginfo('no checkpoint for model found...')
        sys.exit(1)
    else:
        rospy.loginfo('Loading model from ' + ckpt.model_checkpoint_path)
        saver.restore(self.sess, ckpt.model_checkpoint_path)
    
    self.min_confidence = min_confidence
    self.min_detection_height = min_detection_height
    self.nms_max_overlap = nms_max_overlap
    self.tracker = Tracker(self.sess, self.model, max_age=max_age, n_init=n_init)
    
  def update(self, image, dets):
    """
    Params:
      dets - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    # generate the detection features
    print '\n\n\nStart processing current frame ..................................................................'
    detection_features = generate_detections(self.sess, self.model, image, dets)
    
    # use detection features for trackers to track
    detections = create_detections(detection_features)
    detections = [d for d in detections if d.confidence >= self.min_confidence]
    
    boxes = np.array([d.tlwh for d in detections])
    scores = np.array([d.confidence for d in detections])
    indices = preprocessing.non_max_suppression(boxes, self.nms_max_overlap, scores)
    detections = [detections[i] for i in indices]

    # update tracker
    self.tracker.predict()
    self.tracker.update(detections)

    # store results
    results = []
    for track in self.tracker.tracks:
        if not track.is_confirmed() or track.time_since_update > 0:
            continue
        bbox = track.to_tlwh()
        results.append([track.track_id, bbox[0], bbox[1], bbox[2], bbox[3]])
            
    return results
