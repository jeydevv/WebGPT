# Imports
import os
import gradio as gr
from langchain import FAISS
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import TextLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from urllib.request import urlopen
from urllib.request import Request

# Set OpenAI API key system variable
os.environ['OPENAI_API_KEY'] = ""


def get_html(url):
    """
    get_html retrieves the plain text source of a webpage given its URL

    :param url: the URL of the webpage to be downloaded
    :return: a string of the HTML code
    """
    # Set custom header for HTML request
    hdr = {'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11',
           'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
           'Accept-Charset': 'ISO-8859-1,utf-8;q=0.7,*;q=0.3',
           'Accept-Encoding': 'none',
           'Accept-Language': 'en-US,en;q=0.8',
           'Connection': 'keep-alive'}
    req = Request(url, headers=hdr)

    # Read the plain text source code
    response = urlopen(req)
    html = str(response.read())
    response.close()

    # Strip curly brackets and newline characters
    html = html.replace("{", " ")
    html = html.replace("}", " ")
    html = html.replace("\n", " ")

    return html


def get_html_vdb(url):
    """
    get_html_vdb converts the plain text source of a website to a vector database

    :param url: the URL of the webpage to be converted to a vector database
    :return: a vector database of the webpage's plain text source, the first 4000 characters of the webpage (contains SEO-relevant data)
    """
    # Retrieve plain text source code
    html = get_html(url)

    # Save the first 4000 characters which contain SEO data
    first_snippet = html[:4000]

    # Write the plain text source code to a temp file
    with open('temp/page.txt', 'w') as f:
        f.write(html)

    # Read the temp file using a document loader so that FAISS can interpret it as a document
    loader = TextLoader("temp/page.txt")
    html = loader.load()

    # Split the plain text source code in 4000 character segments, tokenize and insert in a vector database
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000)
    html_as_tokens = text_splitter.split_documents(html)
    vector_db = FAISS.from_documents(html_as_tokens, OpenAIEmbeddings())

    return vector_db, first_snippet


def get_analysis(vector_db, first_snippet, input_query, k=10):
    """
    get_analysis generates the GPT-agent and returns the response to a given query

    :param vector_db: the vector database of a webpage
    :param first_snippet: the first 4000 characters of the webpage source code as a string (contains SEO-relevant data)
    :param input_query: the question that the GPT-agent is going to receive and respond do
    :param k: the amount of relevant segments to retrieve from the similarity search
    :return: the response from the GPT-agent, the relevant segments from the similarity search
    """
    # Retrieve code snippets relating to the query with a similarity search
    related_snippets = vector_db.similarity_search(input_query, k=k)
    # Concatenate related snippets into one string, for SEO analysis first_snippet is added on to the front
    page_content = first_snippet + " ".join([doc.page_content for doc in related_snippets])

    # Instantiate chat model, low temperature meaning less creativity
    chat = ChatOpenAI(model_name="gpt-3.5-turbo-16k", temperature=0.1)

    # System template to be fed to the model to tell it what it needs to do
    system_template = f"""
        You are a tool used to analyse websites to answer questions about the website and give SEO advice based off of the website's code: 
        {page_content}
        
        Only use the information provided in the code and do not guess anything.
        
        If you don't have enough information ask for more.
        
        Your answers should be detailed and do not use bullet points unless asked about SEO.
        
        When asked for the website's SEO performance, break it down into bullet points and give advice on each one.
        
        If you are not able to determine the SEO performance of a website, say why in detail.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # Query template can be edited if needed here
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # Set the chat prompt to feed to the LLM chain
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )

    # Instantiate LLM chain and feed the prompt to generate a response
    chain = LLMChain(llm=chat, prompt=chat_prompt)
    response = chain.run(question=input_query, docs=page_content)

    return response, related_snippets


def analyse_seo(homepage_url):
    """
    analyse_seo provides an SEO breakdown and advice of a given webpage

    :param homepage_url: the URL of the webpage to analyse
    :return: the GPT-agent's SEO breakdown
    """
    # Create the vector database and get the first code snippet
    vdb, first_snippet = get_html_vdb(homepage_url)

    # Set the query for an SEO breakdown of the given code and generate a response
    query = "does the website have good seo? break it down in bullet points with advice for each one. give praise when good seo is found."
    response_main, docs_main = get_analysis(vdb, first_snippet, query)

    return response_main


def analyse_question(homepage_url, question):
    """
    analyse_question provides an answer to any question about a given webpage

    :param homepage_url: the URL of the webpage to analyse
    :param question: the question to be asked about the given webpage
    :return: the GPT-agent's response to the given question
    """
    # Create the vector database
    vdb = get_html_vdb(homepage_url)[0]

    # Set the query to the given question and generate a response
    query = question
    response_main, docs_main = get_analysis(vdb, "", query)

    return response_main


# GUI construction for the "SEO Report" tab
seo_gui = gr.Interface(
    fn=analyse_seo,
    inputs=gr.Textbox(label="Webpage URL", lines=1, placeholder="Webpage URL, e.g. 'https://openai.com'..."),
    outputs=gr.Textbox(label="SEO Breakdown"),
    allow_flagging="never",
    title="SEO",
    description="<center>Enter the URL of a webpage to get it's SEO breakdown and advice</center>",
    examples=["https://openai.com"]
)

# GUI construction for the "Website Questions" tab
question_gui = gr.Interface(
    fn=analyse_question,
    inputs=[gr.Textbox(label="Webpage URL", lines=1, placeholder="Webpage URL, e.g. 'https://openai.com'..."),
            gr.Textbox(label="Question", lines=1, placeholder="Your Question, e.g. 'What is this website about?'...")],
    outputs=gr.Textbox(label="Answer"),
    allow_flagging="never",
    title="Q&A",
    description="<center>Enter the URL of a webpage and a question to get the answer</center>",
    examples=[["https://openai.com", "What is this website about?"]]
)

# GUI construction for the tabbed interface (the main GUI)
main_gui = gr.TabbedInterface(
    [seo_gui, question_gui],
    ["SEO Report", "Website Questions"],
    title="WebGPT",
    css="styles.css"
)

# Initialise the GUI
main_gui.launch(favicon_path="j.png")
