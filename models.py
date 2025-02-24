import ollama

# Get a list of all locally available models
all_models = ollama.list()  # This might return a list of strings or dicts

available_models = [str(model.model) for model in all_models.models if not str(model.model).__contains__('embed')]
