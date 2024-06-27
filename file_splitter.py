import os
import random
import shutil

def select_and_move_files(src_dir, dst_dir, num_files=10):
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    # Get a list of all files in the source directory
    files = os.listdir(src_dir)

    # Check if there are enough files in the source directory
    if len(files) < num_files:
        print("Error: Not enough files in the source directory. Found {} files.".format(len(files)))
        return

    # Randomly select the specified number of files
    selected_files = random.sample(files, num_files)

    # Move each selected file to the destination directory
    for file_name in selected_files:
        src_path = os.path.join(src_dir, file_name)
        dst_path = os.path.join(dst_dir, file_name)
        shutil.move(src_path, dst_path)
        print("Moved {} to {}".format(file_name, dst_dir))

if __name__ == '__main__':
    # Set the source and destination directories
    source_directory = "/home/msa/Desktop/sign_language_translation/data/train/yes"
    destination_directory = "/home/msa/Desktop/sign_language_translation/data/test/yes"

    # Move 10 random files from source to destination
    select_and_move_files(source_directory, destination_directory, num_files=10)
