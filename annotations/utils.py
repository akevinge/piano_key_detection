import os


def make_directory_force_recursively(path):
    """
    Creates a directory recursively, including all parent directories.
    If the directory already exists, no error is raised.

    Args:
        path (str): The path of the directory to create.
    """
    try:
        os.makedirs(path, exist_ok=True)
        print(f"Directory '{path}' created successfully (or already existed).")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")
