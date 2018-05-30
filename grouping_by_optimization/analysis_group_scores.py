import pickle
import glob
import os
import progressbar
import numpy as np
from matplotlib import pyplot

pyplot.interactive(False)
kWorkspacePath = "/home/mlpa/Workspace"
kGroupScorePath = os.path.join(kWorkspacePath, "experimental_result/interaction_group_detection/grouping_log")


if "__main__" == __name__:

    log_files = glob.glob(os.path.join(kGroupScorePath, "*.pickle"))

    print("Load score logs...")
    gt_group_scores = []
    other_group_scores = []
    with progressbar.ProgressBar(max_value=len(log_files)) as loading_bar:
        for f_idx, file_path in enumerate(log_files):
            with open(file_path, "rb") as pickle_file:
                group_score_logs = pickle.load(pickle_file)
                for key, value in group_score_logs.items():
                    if value['is_gt']:
                        gt_group_scores.append(value['score'])
                    else:
                        other_group_scores.append(value['score'])
                loading_bar.update(f_idx)


    bins = np.linspace(-8, 2, 100)

    pyplot.hist(other_group_scores, bins, alpha=0.5, label='other')
    pyplot.hist(gt_group_scores, bins, alpha=1.0, label='gt')
    pyplot.legend(loc='upper right')
    pyplot.show()


    gt_group_scores = np.array(gt_group_scores)
    other_group_scores = np.array(other_group_scores)

    gt_mean, gt_min, gt_max = np.mean(gt_group_scores), np.min(gt_group_scores), np.max(gt_group_scores)
    ot_mean, ot_min, ot_max = np.mean(other_group_scores), np.min(other_group_scores), np.max(other_group_scores)
    gt_std, ot_std = np.std(gt_group_scores), np.std(other_group_scores)

    print("Ground truth scores: mean=%f, min=%f, max=%f, std=%f\n" %(gt_mean, gt_min, gt_max, gt_std))
    print("Others scores: mean=%f, min=%f, max=%f, std=%f\n" % (ot_mean, ot_min, ot_max, ot_std))


