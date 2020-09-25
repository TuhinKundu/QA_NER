from transformers import *

from pathlib import Path


embedding_path = "t5_3b/"


Path(embedding_path).mkdir(parents=True, exist_ok=True)

MODELS = [(T5ForConditionalGeneration, T5Tokenizer, 't5-3b')]

for model_class, tokenizer_class, pretrained_weights in MODELS:
    # Load pretrained model/tokenizer
    tokenizer = tokenizer_class.from_pretrained(pretrained_weights)
    model = model_class.from_pretrained(pretrained_weights,
                                        output_hidden_states=True,
                                        output_attentions=False)
    '''
    for name, param in model.named_parameters():
        print(name)
        print(param.size())
    '''
    model.save_pretrained(embedding_path)
    tokenizer.save_pretrained(embedding_path)