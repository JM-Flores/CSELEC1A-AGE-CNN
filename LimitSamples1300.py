# Remove files to limit each classification to 1300 samples

import os
import shutil


def moveFiles():
    source_folder = "D:\python\Final Project\age_types\25-30"
    destination_folder = "D:\python\Final Project\age_types\25-30-final"

    file_names = os.listdir(source_folder)

    for file_name in file_names[1300:]:
        shutil.move(os.path.join(source_folder, file_name), destination_folder)


def main():
    moveFiles()


if name == "main":
    try:
        main()
    except KeyboardInterrupt:
        exit()