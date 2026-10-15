"""Fast logical file-size inventory using Windows directory-entry metadata."""
import os


def count_bytes(root):
    total=0;pending=[os.fspath(root)]
    while pending:
        folder=pending.pop()
        with os.scandir(folder) as entries:
            for entry in entries:
                try:
                    if entry.is_symlink():continue
                    if entry.is_dir(follow_symlinks=False):pending.append(entry.path)
                    elif entry.is_file(follow_symlinks=False):total+=entry.stat(follow_symlinks=False).st_size
                except FileNotFoundError:
                    pass  # Concurrent atomic replacement of a generated status file.
    return total
