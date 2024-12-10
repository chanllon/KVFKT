import tensorflow as tf
import numpy as np
from transformers import pipeline, set_seed

if __name__ == '__main__':
    generator = pipeline('text-generation', model='gpt2')
    set_seed(42)
    generator("Hello, I'm a language model,", max_length=30, num_return_sequences=5)