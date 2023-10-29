import os
from class_manipulate_data import ManipulateData
from class_update_meta_mlflow import UpdateMetaMLFlow

manipulate_data = ManipulateData()
path_mlflow = manipulate_data.get_path_mlflow()



if os.path.isdir(path_mlflow):
    update_meta_mlflow = UpdateMetaMLFlow()
    update_meta_mlflow.process_update(path_mlflow)

os.system(f"mlflow ui --backend-store-uri file:///{path_mlflow} &")
