# **LSTM Shakespeare Interface**

## **Details**

- **Author**: Nick Minnard
- **Course**: ARTI 405: A.I. Architecture & Design
- **Instructor**: Professor Gogolin
- **Semester**: Fall 2024


## Background 

This project is a child of another work called MIDIMIMIC, an music generation architecture that features a multi-layered character-based LSTM neural network, allowing it to train on text-encoded piano music and continuously generate similar but different songs. Since MIDIMIMIC operates on the character level, it was easily adapted and trained on datasets consisting of language. One of the more popular datasets to train small language models on, and the one chosen for this project, is called Tinyshakespeare. Using a truncated 500KB version of Tinyshakespeare, a model was trained to output similar but different Shakespeare plays, plagiarizing virtually nothing from the original works it trained on.

This repository only contains the code for deploying the Shakespeare model, and moreover, is missing the compressed model itself due to GitHub file size constraints. I repeat: without the model, this is only an interface.

### Example Generation

<img width="1007" alt="Example Generation" src="https://github.com/user-attachments/assets/b43fbed3-9b6c-48eb-b343-b5dd5198146e">
