from fastapi import FastAPI, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
import numpy as np
import joblib
import random
import re

model = joblib.load('app/200_epoch_500KB_shakespeare.joblib')

app = FastAPI()

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
def render_home(request: Request):
    '''Render Home Page'''
    return templates.TemplateResponse("index.html", {"request": request})

def sample(preds, temperature):
    '''Sample model predictions at a given temperature'''
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

@app.post('/generate', response_class=HTMLResponse)
async def generate_text(request: Request, names: str = Form(...), output_length: int = Form(...)):
    '''Generate Shakespeare play with user variables and return the home page including the output'''

    # Hard-coded Dataset & Model Params
    temperature = 0.9
    sequence_length = 100

    # 200 Epoch 500KB Model
    char_to_idx = {'\n': 0, ' ': 1, '!': 2, "'": 3, ',': 4, '-': 5, '.': 6, '3': 7, ':': 8, ';': 9, '?': 10,
                   'A': 11, 'B': 12, 'C': 13, 'D': 14, 'E': 15, 'F': 16, 'G': 17, 'H': 18, 'I': 19, 'J': 20,
                   'K': 21, 'L': 22, 'M': 23, 'N': 24, 'O': 25, 'P': 26, 'Q': 27, 'R': 28, 'S': 29, 'T': 30,
                   'U': 31, 'V': 32, 'W': 33, 'X': 34, 'Y': 35, 'Z': 36, 'a': 37, 'b': 38, 'c': 39, 'd': 40,
                   'e': 41, 'f': 42, 'g': 43, 'h': 44, 'i': 45, 'j': 46, 'k': 47, 'l': 48, 'm': 49, 'n': 50,
                   'o': 51, 'p': 52, 'q': 53, 'r': 54, 's': 55, 't': 56, 'u': 57, 'v': 58, 'w': 59, 'x': 60,
                   'y': 61, 'z': 62}
    
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    
    # Create random seed from dataset characters
    chars = list(char_to_idx.keys())
    seed = ''.join([random.choice(chars) for _ in range(sequence_length)])

    # Construct placeholder output matrix
    output = ''
    window = seed
    output += window
    for _ in range(output_length):
        x_pred = np.zeros((1, sequence_length, len(char_to_idx)))
        for t, char in enumerate(window):
            if char in char_to_idx:
                x_pred[0, t, char_to_idx[char]] = 1

        # Populate matrix with model predictions
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[next_index]
        output += next_char
        window = window[1:] + next_char

    # Remove seed from generation
    output = output[sequence_length:]
    
    # Detect script character names (multi-word, ending with ':') in the generated output
    detected_names = re.findall(r'(?m)(?<=\n)([A-Za-z ]+)(?=:\n)', output)

    # Create name mapping
    names_list = [name.strip() for name in names.split(',') if name.strip()]
    random.shuffle(names_list)
    name_replacement_map = {}
    last_used_name = None

    for i, detected_name in enumerate(detected_names):
        available_names = [name for name in names_list if name != last_used_name]
        replacement_name = random.choice(available_names)
        name_replacement_map[(detected_name, i)] = replacement_name
        last_used_name = replacement_name

    # Replace names with user-specified characters
    for (old_name, index), new_name in name_replacement_map.items():
        # Use a unique regex for each occurrence to replace it separately
        output = re.sub(rf'(?m)(?<=\n){re.escape(old_name)}(?=:\n)', new_name, output, count=1)

    # Truncate Leading Text
    match = re.search(r'(?m)^[A-Za-z ]+:\n', output)
    if match:
        output = output[match.start():]
    
    return templates.TemplateResponse("index.html", {"request": request, "generation": output})
