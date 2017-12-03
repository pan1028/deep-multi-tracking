import os
import pickle
import numpy as np
import cv2

class DataLoader():

	def __init__(self, is_valid=False, valid_portion=0.2, batch_size=64, seq_number=7, preprocess=False):
		self.parent_dir = './2DMOT2015/train'

		self.all_data_dirs = ['TUD-Campus', 'TUD-Stadtmitte', 'ETH-Sunnyday', 'KITTI-17', 
							  'KITTI-13',  'ADL-Rundle-6', 'Venice-2', 'ADL-Rundle-8', 
							  'ETH-Pedcross2', 'ETH-Bahnhof', 'PETS09-S2L1']

		# self.all_data_dirs = ['TUD-Campus']

		self.batch_size = batch_size

		self.valid_portion = valid_portion
		
		self.patch_shape = 128, 64, 3
		
		self.seq_number = seq_number

		self.pointer = 0

		if preprocess:
			self.preprocess_data(self.parent_dir, self.all_data_dirs)

		self.load_preprocessed(self.parent_dir, self.all_data_dirs, is_valid)


	def preprocess_data(self, parent_dir, all_data_dirs):
		for directory in all_data_dirs:
			print 'Preprocessing ' + directory + '...............................................................'
			directory = os.path.join(parent_dir, directory)
			print directory
			file_path = os.path.join(directory, 'gt/gt.txt')
			data = np.genfromtxt(file_path, delimiter=',')
			
			pos_features = []
			pos_labels = []

			neg_features = []
			neg_labels = []

			orig_det = data[:, 0:6]
			sort_targets = orig_det[np.lexsort((orig_det[:, 0], orig_det[:, 1]))]
			total_frames = orig_det[-1, 0]

			index = 0
			num_sample = 0
			seq_number = self.seq_number
			
			# Generate 1 positive sample and 1 negative sample
			while index <= (sort_targets.shape[0] - seq_number):
				# print index
				curr_sample = sort_targets[index: index + seq_number, :]
				# if index == 0:
					# print curr_sample
				curr_person_id = curr_sample[0, 1]
				# print curr_person_id
				not_same_person = curr_sample[curr_sample[:, 1] != curr_person_id]
				
				if not_same_person.size == 0:
					pos_image_patches = []
					neg_image_patches = []
					
					for i in range(curr_sample.shape[0]):
						person_det = curr_sample[i, :]
						# print person_det
						# print person_det[2:]

						image_patch = self.extract_image_patch(directory, person_det[0], person_det[2:], self.patch_shape[:2])
						pos_image_patches.append(image_patch)
						neg_image_patches.append(image_patch)
					
					last_frame_id = curr_sample[seq_number - 2, 0]
					# if index == 0:
					# 	print last_frame_id

					false_detection = []
					while last_frame_id < total_frames:
						next_frame_detections = orig_det[orig_det[:, 0] == last_frame_id + 1]
						# if index == 0:
						# 	print next_frame_detections
						false_detections = next_frame_detections[next_frame_detections[:, 1] != curr_person_id]
						if false_detections.size != 0:
							false_detection = false_detections[0, :]
							break
						last_frame_id += 1

					# if index == 0:
					# 	print false_detection

					if false_detection.size != 0:
						print 'Generating sample ' + str(num_sample)
						num_sample += 1

						pos_features.append(pos_image_patches)
						pos_labels.append(1.0)

						image_patch = self.extract_image_patch(directory, false_detection[0], false_detection[2:], self.patch_shape[:2])
						neg_image_patches[6] = image_patch
						neg_features.append(neg_image_patches)
						neg_labels.append(0.0)

					index += 1
				else:
					index += seq_number - not_same_person.shape[0]

			pos_features, pos_labels = self._shuffle_tensors(pos_features, pos_labels)
			neg_features, neg_labels = self._shuffle_tensors(neg_features, neg_labels)

			num_valid_samples = int(self.valid_portion * pos_features.shape[0])
			training_pos_features = pos_features[num_valid_samples:, :, :, :, :]
			training_pos_labels = pos_labels[num_valid_samples:]
			
			valid_pos_features = pos_features[:num_valid_samples, :, :, :, :]
			valid_pos_labels = pos_labels[:num_valid_samples]
			
			training_neg_features = neg_features[num_valid_samples:, :, :, :, :]
			training_neg_labels = neg_labels[num_valid_samples:]

			valid_neg_features = neg_features[:num_valid_samples, :, :, :, :]
			valid_neg_labels = neg_labels[:num_valid_samples]

			training_features = np.concatenate((training_pos_features, training_neg_features))
			training_labels = np.concatenate((training_pos_labels, training_neg_labels))

			valid_features = np.concatenate((valid_pos_features, valid_neg_features))
			valid_labels = np.concatenate((valid_pos_labels, valid_neg_labels))

			data_file = os.path.join(directory, 'train_preprocess.cpkl')
			f = open(data_file, "wb")
			pickle.dump((training_features, training_labels), f, protocol=2)
			f.close()

			data_file = os.path.join(directory, 'valid_preprocess.cpkl')
			f = open(data_file, "wb")
			pickle.dump((valid_features, valid_labels), f, protocol=2)
			f.close()

			print('Current training sample size:' + str(np.array(training_features).shape))
			print('Current training label size:' + str(np.array(training_labels).shape))
			print('Current valid sample size:' + str(np.array(valid_features).shape))
			print('Current valid label size:' + str(np.array(valid_labels).shape))
	
	def load_preprocessed(self, parent_dir, data_dirs, is_valid):
		print 'Loading data set........'
		
		for i in range(len(data_dirs)):
			data_dir = data_dirs[i]
			
			if is_valid:
				data_file = os.path.join(parent_dir, data_dir + '/valid_preprocess.cpkl')
			else:
				data_file = os.path.join(parent_dir, data_dir + '/train_preprocess.cpkl')				
			f = open(data_file, 'rb')
			data = pickle.load(f)
			f.close()

			if i == 0:
				all_features = np.array(data[0])
				all_labels = np.array(data[1])
			else:
				all_features = np.concatenate((all_features, np.array(data[0])))
				all_labels = np.concatenate((all_labels, np.array(data[1])))

		self.features = all_features
		self.labels = all_labels
		self.total_samples = self.features.shape[0]
		self.num_batches = self.total_samples / self.batch_size

		# print self.total_samples
		# print self.features.shape
		# print self.labels.shape

	def next_batch(self):
		if self.pointer + self.batch_size <= self.total_samples:
			batch_image_patches = self.features[self.pointer: self.pointer+self.batch_size, :, :, :, :]
			batch_labels = self.labels[self.pointer: self.pointer+self.batch_size]
			self.pointer += self.batch_size
		else:
			batch_image_patches = self.features[self.pointer:, :, :, :, :]
			batch_labels = self.labels[self.pointer:]
			self.pointer = self.total_samples

		return batch_image_patches, batch_labels

	def reset_pointer(self):
		self.pointer = 0

	def _shuffle_tensors(self, tensor1, tensor2):
		tensor1 = np.array(tensor1)
		tensor2 = np.array(tensor2)
		perm = np.random.permutation(tensor1.shape[0])
		return tensor1[perm], tensor2[perm]

	def shuffle(self):
		self.features, self.labels = self._shuffle_tensors(self.features, self.labels)

		# perm = np.random.permutation(self.features.shape[0])
		# self.features = self.features[perm, :, :, :, :]
		# self.labels = self.labels[perm]

	def extract_image_patch(self, directory, frame_id, bbox, patch_shape):
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
		image_path = os.path.join(directory, 'img1/' + str(int(frame_id)).zfill(6) + '.jpg')
		# print image_path
		image = cv2.imread(image_path)

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
		# cv2.imshow('image', image)
		# cv2.waitKey(0)
		# cv2.destroyAllWindows()
		return image

# dataloader = DataLoader()