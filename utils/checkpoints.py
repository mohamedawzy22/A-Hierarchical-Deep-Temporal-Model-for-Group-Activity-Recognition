# import os
# from datetime import datetime


# def get_experiment_path(base_path, exp_name=None):

#     # Generate experiment name using current date and time
#     if exp_name is None:
#         exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

#     # Create experiment folder
#     exp_path = os.path.join(base_path, exp_name)

#     # Create directory if it does not exist
#     os.makedirs(exp_path, exist_ok=True)

#     # Same folder will be used for saving checkpoints/results
#     save_path = exp_path

#     return exp_path, save_path

import os
from datetime import datetime

def get_experiment_path(base_path=None, exp_name=None):
    """
    (اختياري حاليًا بس مش هنستخدمه)
    ممكن تشيله لو مش محتاج experiments folders
    """
    if exp_name is None:
        exp_name = datetime.now().strftime("exp_%Y%m%d_%H%M%S")

    exp_path = os.path.join(base_path, exp_name)
    os.makedirs(exp_path, exist_ok=True)

    return exp_path