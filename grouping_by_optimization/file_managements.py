import os
import csv
import glob
import progressbar

kWorkspacePath = "/home/mlpa/Workspace"
kDatasetPath = os.path.join(kWorkspacePath, "dataset/interacting_group_detection")
kResultPath = os.path.join(kWorkspacePath, "experimental_result/interaction_group_detection")
kFactorSavePath = os.path.join(kResultPath, "geometric_factors")

def move_processed_images():
    processed_file_list = glob.glob(os.path.join(kResultPath, "grouping_result/*_log.txt"))
    processed_file_names = [os.path.basename(file_path).replace("_log.txt", "") for file_path in processed_file_list]

    image_file_list = glob.glob(os.path.join(kDatasetPath, "image/*.jpg"))
    for image_path in image_file_list:
        if os.path.basename(image_path).replace(".jpg", "") in processed_file_names:
            os.rename(image_path, os.path.join(kDatasetPath, "image/done", os.path.basename(image_path)))


def move_log_files():
    processed_file_list = glob.glob(os.path.join(kResultPath, "grouping_result/*_log.txt"))
    if not os.path.exists(os.path.join(kResultPath, "grouping_result/logs")):
        os.makedirs(os.path.join(kResultPath, "grouping_result/logs"))
    for file_path in processed_file_list:
        os.rename(file_path, os.path.join(kResultPath, "grouping_result/logs", os.path.basename(file_path)))


def convert_result():
    processed_file_list = glob.glob(os.path.join(kResultPath, "grouping_result/*_log.txt"))
    processed_file_list = [file_path.replace("_log.txt", ".txt") for file_path in processed_file_list]

    with progressbar.ProgressBar(max_value=len(processed_file_list)) as file_convert_bar:
        for f_idx, file_path in enumerate(processed_file_list):
            # read geometric factor to count the number of heads
            image_name = os.path.basename(file_path).replace(".txt", "")
            with open(os.path.join(kFactorSavePath, image_name + ".csv")) as csvfile:
                reader = csv.reader(csvfile)
                next(reader)  # skip header
                geo_factors = []
                for row in reader:
                    geo_factors.append([float(x) for x in row])

            num_heads = int(geo_factors[-1][1])

            groups = []
            with open(file_path, "r+") as text_file:
                content = text_file.readlines()
                for row in content:
                    groups.append(set([int(head_idx) for head_idx in row.split(",")[:-1]]))

                new_result = [group for group in groups]
                for head_idx in range(num_heads):
                    is_found = False
                    for group in groups:
                        if head_idx in group:
                            is_found = True
                            break
                    if not is_found:
                        new_result.append(set([head_idx]))

                text_file.seek(0)
                for group in new_result:
                    for head in group:
                        text_file.write("%d " % head)
                    text_file.write("\n")
                text_file.truncate()

        file_convert_bar.update(f_idx)


if "__main__" == __name__:
    move_log_files()
