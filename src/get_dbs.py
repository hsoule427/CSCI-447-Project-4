from path_manager import pathManager

def get_db():
    pm = pathManager()
    db_list = pm.find_folders(pm.get_databases_dir())
    return db_list
