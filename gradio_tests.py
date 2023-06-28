import gradio as gr


def analyse(homepage_url):
    # primary function
    return "output here"


gui = gr.Interface(
    fn=analyse,
    inputs=gr.Textbox(label="Homepage URL", lines=1, placeholder="Homepage URL, e.g. 'brightminded.com'..."),
    outputs=gr.Textbox(label="Output"),
    allow_flagging="manual",
    title="BrightGPT",
    description="<center>A website analysis and SEO reporting GPT-agent using a GUI and implementing vector storage</center>"
)

gui.launch(favicon_path="brightminded-logo.webp")
