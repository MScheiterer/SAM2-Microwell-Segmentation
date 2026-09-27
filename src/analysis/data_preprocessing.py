import os
import json

def extract_params(filename, equals=None):
    """
    Extracts the NaCl percentage from the filename.
    
    Args:
        filename (str): The name of the image file.
        equals (str): The name of a specific batch to be analyzed. Format: e.g. '290-4-0.75x-3' (no percentage)
        
    Returns:
        dict: A dictionary with the file path and NaCl percentage.
    """
    if filename.endswith(".jpg"):
        name = filename.replace(",", ".")
        parts = name.split("-")
        prefix = "-".join(parts[:3])
        nacl_percentage = float(parts[3].strip("_").strip("%"))
        chip_part = parts[4]
        if equals is not None:
            batch_name = "-".join(["-".join(parts[:3]), chip_part])
            if f"{equals}.jpg" != batch_name:
                return None
        return {
            "nacl_percentage": nacl_percentage,
            "group": "-".join([prefix, chip_part])
        }
            

def data_preprocessing(path, equals=None, display=False):
    """
    Returns a dictionary of the form
    {
        "<batchname>": [
            {
                "file_path": <path>,
                "nacl_percentage": <nacl_percentage>
            }, 
            ...
        ],
        ...
    }
    
    Args:
        path (str): The path to the image directory to be analyzed
        equals (str): The name of a specific batch to be analyzed. Format: e.g. '290-4-0.75x-3' (no percentage)
    """
    data = {}
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            params = extract_params(filename, equals)
            if params is None:
                continue
            if params["group"] not in data.keys():
                data[params["group"]] = []
            data[params["group"]].append({
                "file_path": os.path.join(path, filename),
                "nacl_percentage": params["nacl_percentage"]
            })
    for group in data.keys():
        data[group] = sorted(data[group], key=lambda x: x["nacl_percentage"])
    if display:
        print(json.dumps(data, indent=4))
    return data