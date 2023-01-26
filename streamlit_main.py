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
    if "count" not in st.session_state:
        st.session_state.count = 0
    # if "data_loaded" not in st.session_state:
    #     st.session_state.data_loaded = False


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
    st.session_state.data_loaded = True
    return s

def uploaded_example():
    if st.session_state.upload_file is not None:
        s = upload_file(st.session_state.upload_file)
        st.session_state["signal"] = s
        st.session_state.count = 0

def load_example():
    example_name = int(st.session_state.example_selection.split("_")[1]) - 1
    examples = st.session_state["examples"]
    s = upload_file("data/" + examples[example_name])
    st.session_state["signal"] = s
    st.session_state.count = 0
    return s

def load_sidebar():
    with st.sidebar:
        st.header("Get started here!!!")
        st.markdown("__Search__ for __patterns__ on time series like if you could __Google it__!!!")
        st.markdown("If you wish to find examples of a particular behavior with our system, "
                    "you can simply describe the data with words from our vocabulary and operators.")
        st.markdown(
            "For example, if you wish to find data that is suggestive of a high change, you can write a simple query, like:")
        st.markdown("__s1: high__")

        tab1, tab2 = st.tabs(["Words", "Operators"])

        with tab1:
            word = st.selectbox('Select a word from our vocabulary:',
                                ('noise', 'up', 'down', 'flat', 'symmetric', 'assymetric', 'complex', 'high', 'low',
                                 'peak', 'valley', 'step_up',
                                 'step_down', 'plateau_up', 'plateau_down', 'top', 'bottom', 'simple', 'quick', 'vval',
                                 'uval', 'middle', 'clean'), key="select_word")
                                 # 'top', 'bottom'), key="select_word", on_change=update_search_word)
            mkd_sentence = word_descriptions[word][0]
            # mkd_image = Image.open(word_descriptions[word][1])
            st.markdown(mkd_sentence)
            # st.image(mkd_image, width=150)

            # st.button('Try the word!', on_click=update_search_word)

        with tab2:
            operator = st.selectbox('Select an operator:',
                                    ('not (!)', 'followed by', 'grouped followed by ([])', 'wildcard (.)'),
                                    key="select_operator")
                                    # key="select_operator", on_change=update_search_operator)
            mkd_sentence = word_descriptions[operator][0]
            mkd_search = word_descriptions[operator][1]

            st.markdown(mkd_sentence)

def query_options():
    c1, c2, c3 = st.columns([6, 2, 2])
    with st.container():
        with c1:
            query_input = query_placeholder()
        with c2:
            win_size = window_size_placeholder()
        with c3:
            k = number_of_subsequences()

def query_placeholder():
    # placeholder = st.empty()
    # input = placeholder.text_input("Query:", key="query")
    text_input = st.text_input(
        "Query:",
        label_visibility=st.session_state.visibility,
        placeholder=st.session_state.placeholder,
        key="query",
        on_change=search
    )

    return text_input

def window_size_placeholder():
    win_size_input = st.text_input(
        "Window Size:",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder="Enter window size...",
        value=500,
        key="win_size",
        on_change=recaculate_wfvs
    )

    return win_size_input

def number_of_subsequences():
    nbr_k = st.text_input(
        "Nbr of Subsequences:",
        label_visibility=st.session_state.visibility,
        disabled=st.session_state.disabled,
        placeholder="k subsequences...",
        value=10,
        key="k",
        on_change=update_figure_for_k
    )

    return nbr_k

def search():
    # st.write(query_input)
    ##check if wfv is initialized, otherwise, run it
    # if("wfv1" not in st.session_state):
    #     if ("win_size" not in st.session_state):
    #         run_static_code(st.session_state["signal"], 500)
    #     else:
    #         run_static_code(st.session_state["signal"], int(st.session_state.win_size))

    scores = query_search(st.session_state["signal"], st.session_state.wfv1, st.session_state.wfv2, st.session_state.wfv3, [st.session_state.query], int(st.session_state.win_size))
    st.session_state.scores = scores
    fig_ = update_figure(st.session_state["signal"], scores, int(st.session_state.k), int(st.session_state.win_size))
    st.session_state.chart1 = fig_

def load_uploaders():
    # loading
    # make two columns for loading or selecting the example that loads the signal to be analysed
    c1, c2 = st.columns([0.5, 0.5])
    with c1:
        # file uploader (when changed, uploads the dropped signal)
        # for now, must be a .txt with tab separated values
        uploaded_file = st.file_uploader("Choose a file", key="upload_file", on_change=uploaded_example)
        # uploaded_file = st.file_uploader("Choose a file", key="upload_file")
    with c2:
        # example selector. Loads the signals from the examples in the folder ./data/
        # when changed, loads the signal
        examples = os.listdir("data/")
        st.session_state["examples"] = examples
        select_examples = st.selectbox("Load an example:", ["example_" + str(i + 1) for i in range(len(examples))],
                                       index=0, key="example_selection", on_change=load_example)

def recaculate_wfvs():
    key1, key2, key3 = run_static_code(st.session_state.signal, int(st.session_state.win_size))
    st.session_state["wfv1"] = key1
    st.session_state["wfv2"] = key2
    st.session_state["wfv3"] = key3
    #afterwards, search in case of having a query
    if st.session_state.query is not None:
        search()


@st.cache(suppress_st_warning=True)
def run_static_code(s, win_size):
    # load style properties
    # % 1 - extract word feature vectors
    key1, key2, key3 = word_feature_extraction(s, win_size)
    return key1, key2, key3

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

def update_figure(s, scores, k, win_size):
    # from the retrieved scores, display the segments found put scores in same dimension as SIGNAL
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    x_scores = np.linspace(1, len(s), len(s))

    fig.add_trace(
        go.Line(x=x_scores, y=scores),
        row=2, col=1
    )
    arg_k = np.zeros(k)

    if (np.ndim(s) > 1):
        for i in range(np.shape(s)[1]):
            # fig.data[i]["y"] = []
            fig.add_trace(go.Scatter(y=norm_1_sig(s[:, i]) + (i - 1) * 1.5,
                                     mode="lines", line=go.scatter.Line(color="gray")),
                          row=1, col=1)
    else:
        # fig.data[0]["y"] = []
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

def update_figure_for_k():
    if("scores" in st.session_state):
        fig_ = update_figure(st.session_state["signal"], st.session_state.scores, int(st.session_state.k),
                             int(st.session_state.win_size))
        st.session_state.chart1 = fig_
    else:
        search()


#load css file
local_css("streamlit_content/style.css")
#load word descriptions for side bar description
word_descriptions = upload_word_description_file()
#initialize a few variables
initialization()
#open sidebar
load_sidebar()
#title
st.title("Quots")
#load uploaders
load_uploaders()
# if(st.session_state.data_loaded):
query_options()
#plot first figure in case
if(st.session_state.count==0):
    print("first plot")
    #get signal from example
    s = load_example()
    # plot first figure
    scores = np.zeros(len(s))
    fig1 = first_figure(s, scores)
    # compute wfvs after plotting the figure for the first time
    key1, key2, key3 = run_static_code(st.session_state.signal, int(st.session_state.win_size))
    st.session_state["wfv1"] = key1
    st.session_state["wfv2"] = key2
    st.session_state["wfv3"] = key3
    #display figure
    chart_ = st.empty()
    chart_.write(fig1)
    st.session_state.count = 1
    st.session_state.chart1 = fig1

else:
    print("other plot")
    chart_ = st.empty()
    chart_.write(st.session_state.chart1)
