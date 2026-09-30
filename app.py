from fastai.vision.all import *
import gradio as gr

def is_cat(x): return x[0].isupper()

learn = load_learner('model.pkl')

categories = ('Dog', 'Cat')

def classify_image(img):
    pred, idx, probs = learn.predict(img)
    return dict(zip(categories, map(float, probs)))

intf = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil", height=192, width=192),
    outputs=gr.Label(),
)
intf.launch()