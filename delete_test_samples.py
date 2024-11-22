import os

def get_file_set(folder_path):
    """
    Get a set of file names in a folder.
    
    Parameters:
    - folder_path (str): Path to the folder.
    
    Returns:
    - set: Set of file names in the folder.
    """
    return {file for file in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, file))}

def get_files_from_subfolders(main_folder_path):
    """
    Get a set of file names from all subfolders inside the main folder.
    
    Parameters:
    - main_folder_path (str): Path to the main folder containing subfolders.
    
    Returns:
    - set: Set of file names across all subfolders.
    """
    all_files = set()
    for subfolder in os.listdir(main_folder_path):
        subfolder_path = os.path.join(main_folder_path, subfolder)
        if os.path.isdir(subfolder_path):
            all_files.update(get_file_set(subfolder_path))
    return all_files

def delete_common_files(main_folder_path, target_folders):
    """
    Delete common files in the target folders if they exist in the main folder and its subfolders.
    
    Parameters:
    - main_folder_path (str): Path to the main folder which you downloaded from google drive
    - target_folders (list of str): path to the folders of train and tier3 from xBD dataset
    """
    
    main_files = get_files_from_subfolders(main_folder_path)  # Collect all file names from the main folder's subfolders
    
    for folder in target_folders:
        target_files = get_file_set(folder)
        common_files = main_files.intersection(target_files)
        
        for file in common_files:
            file_path = os.path.join(folder, file)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

# Example usage
main_folder_path = ##### to be changed ###### add the folder path which you downloaded from google drive
target_folders = #['path_to_train_folder', 'path_to_tier3_folder']   Replace with the paths to train and tier3 folders from xBD

delete_common_files(main_folder_path, target_folders)
