import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL

st.set_page_config(page_title="LangChain: Summarize text from YT or website")
st.title("Langchain: Summarize text from YT or Website")
st.subheader("Summarize URL")


with st.sidebar:
    groq_api_key = st.text_input("Groq API key", value="", type="password")
    video_info = st.checkbox("Add video info", value=False)

llm = ChatGroq(groq_api_key=groq_api_key, model="gemma2-9b-it")

prompt_template = """
Provide a summary of the following content in 300 words:
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

url = st.text_input("URL",label_visibility="collapsed")

if st.button("Summarize the content from YT or Website"):
    if not groq_api_key.strip() or not url.strip():
        st.error("Please provide the required information")
    elif not validators.url(url):
        st.error("Please enter a valid URL. It can may be a YT video url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                metadata = None
                if "youtube.com" in url:
                    if video_info:
                        loader = YoutubeLoader.from_youtube_url(youtube_url=url)
                        metadata = YoutubeLoaderDL.from_youtube_url(url, add_video_info=video_info).load()
                    else:
                        loader = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=video_info)
                else:
                    loader=UnstructuredURLLoader(urls=[url],
                                                 ssl_verified=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"})
                docs = loader.load()

                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success(summary)
                if metadata:
                    st.subheader("Video Metadata")
                    st.json(metadata[0].metadata, expanded=True)
        except Exception as e:
            st.exception(f"Exception: {e}")