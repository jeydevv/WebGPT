import os
import gradio as gr
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from urllib.request import urlopen
from urllib.request import Request

os.environ['OPENAI_API_KEY'] = ""


def get_html(url):
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8',
           'Connection': 'keep-alive'}
    req = Request(url, headers=hdr)
    response = urlopen(req)
    html = str(response.read())
    html = html.replace("{", " ")
    html = html.replace("}", " ")
    html = html.replace("\n", " ")
    response.close()
    return html


def get_html_vdb(url):
    html = get_html(url)

    first_snippet = html[:4000]

    with open('temp/page.txt', 'w') as f:
        f.write(html)

    loader = TextLoader("temp/page.txt")
    html = loader.load()

    # del file?

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    html_as_tokens = text_splitter.split_documents(html)
    vector_db = FAISS.from_documents(html_as_tokens, OpenAIEmbeddings())

    return vector_db, first_snippet


def get_analysis(vector_db, first_snippet, input_query, k=10):
    related_snippets = vector_db.similarity_search(input_query, k=k)
    page_content = first_snippet + " ".join([doc.page_content for doc in related_snippets])

    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)

    system_template = f"""
        You are a tool used to analyse websites to answer questions about the website and give SEO advice based off of the website's code: 
        {page_content}
        
        Only use the information provided in the code and do not guess anything.
        
        If you don't have enough information ask for more.
        
        Your answers should be detailed and do not use bullit points unless asked about SEO.
        
        When asked for the website's SEO performance, break it down into bullit points and give advice on each one.
        
        If you are not able to determine the SEO performance of a website, say why in detail.
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    chain = LLMChain(llm=chat, prompt=chat_prompt)

    response = chain.run(question=input_query, docs=page_content)
    return response, related_snippets


def analyse_seo(homepage_url):
    vdb, first_snippet = get_html_vdb(homepage_url)
    query = "does the website have good seo? break it down in bullit points with advice for each one. give praise when good seo is found."
    response_main, docs_main = get_analysis(vdb, first_snippet, query)
    return response_main


def analyse_question(homepage_url, question):
    vdb, first_snippet = get_html_vdb(homepage_url)
    query = question
    response_main, docs_main = get_analysis(vdb, "", query)
    return response_main


seo_gui = gr.Interface(
    fn=analyse_seo,
    inputs=gr.Textbox(label="Webpage URL", lines=1, placeholder="Webpage URL, e.g. 'https://brightminded.com'..."),
    outputs=gr.Textbox(label="SEO Breakdown"),
    allow_flagging="never",
    title="SEO",
    description="<center>Enter the URL of a webpage to get it's SEO breakdown and advice</center>"
)

question_gui = gr.Interface(
    fn=analyse_question,
    inputs=[gr.Textbox(label="Webpage URL", lines=1, placeholder="Webpage URL, e.g. 'https://brightminded.com'..."),
            gr.Textbox(label="Question", lines=1, placeholder="Your Question, e.g. 'What is this website about?'...")],
    outputs=gr.Textbox(label="Response"),
    allow_flagging="never",
    title="Q&A",
    description="<center>Enter the URL of a webpage and a question to get the answer</center>"
)

main_gui = gr.TabbedInterface(
    [seo_gui, question_gui],
    ["SEO Report", "Website Questions"],
    title="BrightGPT",
    css="styles.css"
)

main_gui.launch(favicon_path="brightminded-logo.webp")
