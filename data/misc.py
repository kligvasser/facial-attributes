import os


IMAGE_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']


def get_path_list(path_dir, max_size=None, extentions=IMAGE_EXTENSIONS):
    paths = list()

    for dirpath, _, files in os.walk(path_dir):
        for filename in files:
            fname = os.path.join(dirpath, filename)
            if fname.endswith(tuple(extentions)):
                paths.append(fname)

    return sorted(paths)[:max_size]
