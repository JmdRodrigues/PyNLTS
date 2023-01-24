import os

import streamlit as st

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import matplotlib.pyplot as plt

import json

from tools.word_feature_vectors import word_feature_extraction, norm_1_sig, z_norm_sig
from tools.query_search import query_search

# from novainstrumentation import lowpass
# from tools.plot_tools import highlightSubsequences

import numpy as np

def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def initialization():
    # Store the initial value of widgets in session state
    if "visibility" not in st.session_state:
        st.session_state.visibility = "visible"
        st.session_state.disabled = False
        st.session_state.placeholder = "Enter text here..."


@st.cache(suppress_st_warning=True)
def upload_word_description_file():
    f = open("streamlit_content/word_descriptions.json")
    word_description = json.load(f)
    return word_description

@st.cache(suppress_st_warning=True)
def upload_file(file):
    s = np.loadtxt(file)
    if(np.ndim(s)>1):
        for i in range(np.shape(s)[1]):
            s[:, i] = z_norm_sig(s[:, i])
    else:
        s = z_norm_sig(s)
    return s

def header_upload_files():
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        uploaded_file = st.file_uploader("Choose a file", key="upload_file", on_change=upload_example)
    with c2:
        examples = os.listdir("data/")
        st.session_state["examples"] = examples
        select_examples = st.selectbox("Load an example:", ["example_"+str(i+1) for i in range(len(examples))], key="example_selection", on_change=load_example)
    #The first time, load example
    upload_example()

@st.cache(suppress_st_warning=True)
def upload_example():
    if st.session_state.upload_file is not None:
        st.session_state["signal"] = upload_file(st.session_state.upload_file)
    else:
        load_example()

@st.cache(suppress_st_warning=True)
def load_example():
    example_name = int(st.session_state.example_selection.split("_")[1])-1
    examples = st.session_state["examples"]
    st.session_state["signal"] = upload_file("data/"+examples[example_name])

def query_placeholder():
    # placeholder = st.empty()
    # input = placeholder.text_input("Query:", key="query")
    text_input = st.text_input(
        "Query:",
        label_visibility=st.session_state.visibility,
        placeholder=st.session_state.placeholder,
        key="query"
    )

    return text_input

def window_size_placeholder():
    win_size_input = st.text_input(
        "Window Size:",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder="Enter window size...",
        value=500
    )

    return win_size_input

def number_of_subsequences():
    nbr_k = st.text_input(
        "Nbr of Subsequences:",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder="k subsequences...",
        value=10
    )

    return nbr_k

def update_figure(s, fig, scores, k, win_size):
    # from the retrieved scores, display the segments found put scores in same dimension as SIGNAL
    x_scores = np.linspace(1, len(s), len(s))

    fig.data[-1]["y"] = scores

    half_win = int(win_size / 2)

    arg_k = np.zeros(k)


    if (np.ndim(s) > 1):
        for i in range(np.shape(s)[1]):
            fig.data[i]["y"] = []
            fig.add_trace(go.Scatter(y=norm_1_sig(s[:, i]) + (i - 1) * 1.5,
                                     mode="lines", line=go.scatter.Line(color="gray")),
                          row=1, col=1)
    else:
        fig.data[0]["y"] = []
        fig.add_trace(go.Scatter(y=norm_1_sig(s),
                                 mode="lines", line=go.scatter.Line(color="gray")),
                      row=1, col=1)

    cmap = plt.get_cmap('cool')

    for k_i in range(k):
        # find max
        argmax_ki = np.argmax(scores)
        max_ki = scores[argmax_ki]
        # save index to plot around it
        arg_k[k_i] = argmax_ki

        #remove the score and its surrounding
        a = argmax_ki
        b = a + win_size
        a_ = a - win_size
        b_ = a + win_size
        scores[a_:b_] = 0
        # highlight on plot
        if (np.ndim(s) > 1):
            for i in range(np.shape(s)[1]):
                #plot subsequences on signal
                s_ = norm_1_sig(s[:, i])
                fig.add_trace(go.Scatter(x=x_scores[a:b], y=s_[a:b]+(i-1)*1.5, mode="lines",
                            line=go.scatter.Line(color="rgba("+",".join([str(int(255*c_i)) for c_i in cmap(max_ki)])+")", width=5)), row=1, col=1)

        else:
            s_ = norm_1_sig(s)
            fig.add_trace(go.Scatter(x=x_scores[a:b], y=s_[a:b], mode="lines",
                                     line=go.scatter.Line(color="rgba("+",".join([str(int(255*c_i)) for c_i in cmap(max_ki)])+")", width=5)), row=1, col=1)

    fig.update(data=fig.data)

    return fig

def first_figure(s, scores):
    # fig = make_subplots(rows=3, cols=1, row_heights=[0.2, 0.6, 0.2], vertical_spacing=0.025, shared_xaxes=True)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    if (s.ndim > 1):
        for i, s_i in enumerate(s.T):
            s_ = norm_1_sig(s_i)
            fig.add_trace(
                go.Line(x=np.linspace(0, len(s), len(s)), y=s_ + (i-1)*1.5),
                row=1, col=1
            )
        fig.add_trace(
            go.Line(x=np.linspace(0, len(scores), len(scores)), y=scores),
            row=2, col=1
        )
    else:
        fig.add_trace(
            go.Line(x=np.linspace(0, len(s), len(s)), y=s),
            row=1, col=1
        )
        fig.add_trace(
            go.Line(x=np.linspace(0, len(scores), len(scores)), y=scores),
            row=2, col=1
        )

    # st.plotly_chart(fig)
    return fig

@st.cache(suppress_st_warning=True)
def run_static_code(s, win_size):
    # load style properties
    # % 1 - extract word feature vectors
    key1, key2, key3 = word_feature_extraction(s, win_size)

    return key1, key2, key3

def search(fig, keys, query_input, win_size):
    key1, key2, key3 = keys
    # 2 - performs search on the signal
    st.write(query_input)
    scores = query_search(st.session_state["signal"], key1, key2, key3, [query_input], int(win_size))

    fig_ = update_figure(st.session_state["signal"], fig, scores, int(k), int(win_size))
    chart_.write(fig_)

def update_search_word():
    word_ = st.session_state.select_word
    st.session_state.query = word_descriptions[word_][1]
def update_search_operator():
    operator_ = st.session_state.select_operator
    st.session_state.query = word_descriptions[operator_][1]


# init
local_css("streamlit_content/style.css")
word_descriptions = upload_word_description_file()
initialization()

st.title("QuoTS")

#load signal
header_upload_files()

#text
c1, c2, c3 = st.columns([6, 2, 2])
with st.container():
    with c1:
        query_input = query_placeholder()
    with c2:
        win_size = window_size_placeholder()
    with c3:
        k = number_of_subsequences()

key1, key2, key3 = run_static_code(st.session_state["signal"], int(win_size))

scores = np.zeros(len(st.session_state["signal"]))

#figure1
fig = first_figure(st.session_state["signal"], scores)
chart_ = st.empty()
chart_.write(fig)

if(query_input):
    keys = [key1, key2, key3]
    search(fig, keys, query_input, win_size)
    # # 2 - performs search on the signal
    # st.write(query_input)
    # scores = query_search(s, key1, key2, key3, [query_input], int(win_size))
    #
    # fig = update_figure(s, fig, scores, int(k), int(win_size))
    # chart_.write(fig)
    #
    # # highlightSubsequences(s, scores, int(k), int(win_size))

# sidebar
with st.sidebar:
    st.header("Get started here!!!")
    st.markdown("__Search__ for __patterns__ on time series like if you could __Google it__!!!")
    st.markdown("If you wish to find examples of a particular behavior with our system, "
                "you can simply describe the data with words from our vocabulary and operators.")
    st.markdown("For example, if you wish to find data that is suggestive of a high change, you can write a simple query, like:")
    st.markdown("__s1: high__")

    tab1, tab2 = st.tabs(["Words", "Operators"])

    with tab1:
        word = st.selectbox('Select a word from our vocabulary:',
                            ('up', 'down', 'noise', 'complex', 'simple', 'periodic', 'common', 'uncommon', 'high', 'low',
                             'top', 'bottom'), key="select_word", on_change=update_search_word)
        mkd_sentence = word_descriptions[word][0]
        # mkd_image = Image.open(word_descriptions[word][1])
        st.markdown(mkd_sentence)
        # st.image(mkd_image, width=150)

        # st.button('Try the word!', on_click=update_search_word)

    with tab2:
        operator = st.selectbox('Select an operator:',
                                ('not (!)', 'followed by', 'grouped followed by ([])', 'wildcard (.)'),
                                key="select_operator", on_change=update_search_operator)
        mkd_sentence = word_descriptions[word][0]
        mkd_search = word_descriptions[word][1]

        # st.button('Try the Operator!', on_click=update_search_operator)