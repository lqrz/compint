import pkg_resources

def get_file(filename):
    return pkg_resources.resource_filename("trained_files", filename)