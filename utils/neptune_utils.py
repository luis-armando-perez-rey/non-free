import neptune.new as neptune
from typing import Optional

def initialize_neptune_run(user, project_name):
    run = neptune.init_run(project=f"{user}/{project_name}")
    return run

def reload_neptune_run(user, project_name, path:str):
    sys_id = load_neptune_id(path)
    run = neptune.init(project=f"{user}/{project_name}", run=sys_id)
    return run

def save_sys_id(run, path):
    # Save system id to txt file
    with open(path, "w") as f:
        f.write(run["sys/id"].fetch())

def load_neptune_id(path):
    with open(path, "r") as f:
        sys_id = f.read()
    return sys_id


