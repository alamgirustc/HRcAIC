import os
import subprocess

# Set the input and output directories
INFEATS_DIR = r'D:\Research\bottom_up_tsv'
OUTFOLDER = r'D:\Research\mscoco\feature\up_down_10_100'

# List of input files
input_files = [
    'karpathy_test_resnet101_faster_rcnn_genome.tsv',
    'karpathy_train_resnet101_faster_rcnn_genome.tsv.0',
    'karpathy_train_resnet101_faster_rcnn_genome.tsv.1',
    'karpathy_val_resnet101_faster_rcnn_genome.tsv',
]

# Loop through the input files and run the Python script for each
script_path = r'D:\Research\tools\create_feats.py'

for input_file in input_files:
    input_path = os.path.join(INFEATS_DIR, input_file)

    # Use the OUTFOLDER directly since all files should be loaded there
    output_folder_path = OUTFOLDER

    # Ensure the output folder exists, and create it if necessary
    os.makedirs(output_folder_path, exist_ok=True)

    # Define the output folder path (without any specific file name)
    output_path = output_folder_path

    command = f'python "{script_path}" --infeats "{input_path}" --outfolder "{output_path}"'
    subprocess.run(command, shell=True)

print("Processing completed for all files.")
