
def find_owner(model):
    if model == 'mGPT':
        return 'ai-forever'
    elif model in ['Llama-2-7b-hf', 'Meta-Llama-3-8B-Instruct', 'Meta-Llama-3-8B', 'Meta-Llama-3-70B-Instruct', 'Meta-Llama-3-70B', 
                   'Llama-3.1-8B', 'Llama-3.1-8B-Instruct', 'Llama-3.1-70B', 'Llama-3.1-70B-Instruct']:
        return 'meta-llama'
    elif model in ['Mistral-7B-Instruct-v0.2', 'Mistral-7B-v0.1', 'Mistral-7B-Instruct-v0.3', 'Mistral-7B-v0.3']:
        return 'mistralai'
    elif model in ['xglm-7.5B']:
        return 'facebook'
    elif model in ['bloomz-7b1']:
        return 'bigscience'
    elif model in ['gemma-2-27b-it', 'gemma-2-9b-it', 'gemma-2-27b', 'gemma-2-9b']:
        return 'google'
    elif model in ['salamandra-7b-instruct']:
        return 'BSC-LT'
    elif model in ['FLOR-6.3B-Instructed']:
        return 'projecte-aina'
    else:
        raise Exception("Model type not defined.") 