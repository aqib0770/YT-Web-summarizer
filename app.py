import validators, streamlit as st
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import YoutubeLoader, UnstructuredURLLoader
from langchain_yt_dlp.youtube_loader import YoutubeLoaderDL
import os

st.set_page_config(page_title="YT Web summarizer")
st.title("Summarize text from YouTube or Website")
st.subheader("Enter the URL")


with st.sidebar:
    google_api_key = st.text_input("Groq API key", value="", type="password")
    st.markdown("Get your API key from [here](https://aistudio.google.com/prompts/new_chat)")
    video_info = st.checkbox("Add video info", value=False)
    if video_info:
        lang = st.selectbox("Transcript Language", ["en", "hi"], index=0)

os.environ["GROQ_API_KEY"] = google_api_key
llm = ChatGoogleGenerativeAI(api_key=google_api_key, model="models/gemini-2.0-flash-exp")

prompt_template = """
If the content is empty, repetitive or irrelevant, respons with "The content could not be summarized meaningfully..
Otherwise Provide a summary of the following content:
Content: {text}
"""

prompt = PromptTemplate(template=prompt_template, input_variables=["text"])

url = st.text_input("URL",label_visibility="collapsed")

if st.button("Summarize"):
    if not google_api_key.strip() or not url.strip():
        st.error("Please provide the required information")
    elif not validators.url(url):
        st.error("Please enter a valid URL. It can may be a YT video url or website url")
    else:
        try:
            with st.spinner("Waiting..."):
                metadata = None
                if "youtube.com" in url:
                    if video_info:
                        loader = YoutubeLoader.from_youtube_url(youtube_url=url, language=lang)
                        metadata = YoutubeLoaderDL.from_youtube_url(url, add_video_info=video_info).load()
                    else:
                        loader = YoutubeLoader.from_youtube_url(youtube_url=url, add_video_info=video_info)
                else:
                    loader=UnstructuredURLLoader(urls=[url],
                                                 ssl_verified=False,
                                                 headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:132.0) Gecko/20100101 Firefox/132.0"})
                docs = loader.load()

                if not docs or all(not doc.page_content.strip() for doc in docs):
                    st.error("No content or transcript found in the provided URL")
                    st.stop()
                chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
                summary = chain.run(docs)

                st.success(summary)
                if metadata:
                    st.subheader("Video Metadata")
                    st.json(metadata[0].metadata, expanded=True)
        except Exception as e:
            st.exception(f"Exception: {e}")