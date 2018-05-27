import numpy as np
import scipy.io
import os
import glob
import progressbar
from PIL import Image

#########################################################
# Pre-defines
#########################################################
kNumRelations = 8
kInputSize = 48
kWorkspacePath = "/home/mlpa/Workspace"
kDatasetPath = os.path.join(kWorkspacePath, "dataset/interacting_group_detection")
kSavePath = os.path.join(kDatasetPath, "pair_samples")


def crop_head(_image, _bbox):
    x1 = max([0, _bbox[0]])
    x2 = min([_image.size[0], _bbox[0] + _bbox[2] - 1])
    y1 = max([0, _bbox[1]])
    y2 = min([_image.size[1], _bbox[1] + _bbox[3] - 1])
    crop_area = (x1, y1, x2, y2)
    # crop head patch from input image
    target_patch = np.array(_image.crop(crop_area).resize((kInputSize, kInputSize), Image.NEAREST), dtype="uint8")
    target_patch = target_patch.reshape((kInputSize, kInputSize, 1))
    return target_patch


if "__main__" == __name__:

    image_file_paths = glob.glob(os.path.join(kDatasetPath, "image/*.jpg"))
    num_images = len(image_file_paths)
    if 0 == num_images:
        print("There is no images at %s\n" % os.path.join(kDatasetPath, "image"))
        exit()

    if not os.path.exists(kSavePath):
        os.makedirs(kSavePath)
        print("Saving path: %s\n" % os.path.abspath(kSavePath))

    print("Generate samples from %d images\n" % num_images)
    x1s, x2s, spatial_cues = [], [], []
    with progressbar.ProgressBar(max_value=num_images) as bar:
        for i, image_path in enumerate(image_file_paths):

            # load annotation
            gt_path = os.path.join(kDatasetPath, "ground_truth", os.path.basename(image_path).replace('.jpg', '.mat'))
            if not os.path.exists(gt_path):
                print("There is no annotation file for image %s\n", os.path.basename(image_path))
                continue
            mat = scipy.io.loadmat(gt_path)
            bbox = np.array(mat['bbox'])
            num_boxes = bbox.shape[0]

            # load image
            input_image = Image.open(image_path).convert("L")
            image_width, image_height = input_image.size

            # crop heads
            head_images = []
            for b_idx in range(num_boxes):
                head_images.append(np.expand_dims(crop_head(input_image, bbox[b_idx, :]), axis=0))

            # for spatial cues
            relative_boxes = []
            for b_idx in range(num_boxes):
                relative_boxes.append([bbox[b_idx, 0] / image_width, bbox[b_idx, 1] / image_height,
                                       bbox[b_idx, 2] / image_width, bbox[b_idx, 3] / image_height])

            # generate pairs
            for b_idx_1 in range(0, num_boxes - 1):
                for b_idx_2 in range(b_idx_1 + 1, num_boxes):

                    if bbox[b_idx_1][0] < bbox[b_idx_2][0]:
                        left, right = b_idx_1, b_idx_2
                    else:
                        left, right = b_idx_2, b_idx_1

                    x1s.append(head_images[left])
                    x2s.append(head_images[right])

                    rel_x_diff = (relative_boxes[left][0] - relative_boxes[right][0]) / relative_boxes[left][2]
                    rel_y_diff = (relative_boxes[left][1] - relative_boxes[right][1]) / relative_boxes[left][3]
                    rel_size_diff = relative_boxes[left][3] / relative_boxes[right][3]
                    spatial_cue = relative_boxes[left] + relative_boxes[right] + [rel_x_diff, rel_y_diff, rel_size_diff]
                    spatial_cues.append(np.expand_dims(spatial_cue, axis=0))

            bar.update(i)

    # save
    print("Reformatting samples...\n")
    x1s = np.concatenate(x1s, axis=0)
    x2s = np.concatenate(x2s, axis=0)
    spatial_cues = np.concatenate(spatial_cues, axis=0)

    print("Save samples...\n")
    np.save(os.path.join(kSavePath, "x1s.npy"), x1s)
    np.save(os.path.join(kSavePath, "x2s.npy"), x2s)
    np.save(os.path.join(kSavePath, "scs.npy"), spatial_cues)

    print("Done! Total %d samples are generated\n" % spatial_cues.shape[0])

    pass