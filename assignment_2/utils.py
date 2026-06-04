def _get_param(job, key):
    """Get a param value list, checking model_params first."""
    if key in job.get("model_params", {}):
        return job["model_params"][key]
    return job[key]

def _set_param(run_description, key, value):
    """Set a param value, writing to model_params if it lives there."""
    if key in run_description.get("model_params", {}):
        run_description["model_params"][key] = value
    else:
        run_description[key] = value
