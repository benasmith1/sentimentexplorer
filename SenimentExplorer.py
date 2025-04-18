
# Import Packages 
# Searching and Parsing 
# Import Googlesearch
from serpapi import GoogleSearch
from newspaper import Article
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import openai

# Visualization
from bokeh.plotting import figure, show, output_file, save
from bokeh.models import ColumnDataSource, HoverTool, CustomJS, Div
from bokeh.layouts import row

import streamlit as st
import numpy as np
import pandas as pd

import boto3
from botocore.exceptions import ClientError

import json5
from multiprocessing import Pool

import random
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

st.set_page_config(
    page_title = "Sentiment Explorer",
    page_icon = "😎",
    # layout = "wide"
)

# Connect to OpenAI 
client = openai.Client(api_key=st.secrets["openaikey"])

st.header("Sentiment Explorer 🔎 😃")

st.markdown("""Find out what people think about a company or product! To explore, search the company or product in the search bar below.  """)
st.markdown("""The app works by:  """)
st.markdown("""
        - Performing a google search for your input query  
        - Parsing each webpage for its overall sentiment  
        - Using OpenAI to summarize common likes and dislikes within these webpages  """)
st.markdown("""Specify the number of search results you would like to return. More results gives more accurate information but takes longer to parse. Searching 
            for 50 results should take approximately 10 seconds.""")
st.markdown("""Try out the following example to see opinions about Spotify Wrapped! Including the word \"Opinions\" in the search helps avoid returning promotional webpages. 
            If the brand is less known and you do not expect there to be a large number of webpages, you might omit the word \"Opinions.\"
            You might also try \"Netflix Opinions\" or \"Waymo Opinions\"""")
st.markdown("""Now 7x faster beacuse of parallel computing. Whoohoo! Code can be found <a href="https://github.com/benasmith1/sentimentexplorer">here</a>""", unsafe_allow_html=True)
with st.form("Form entry"):
    query = st.text_input("Search Bar", value="Spotify Wrapped Opinions")
    num_results = st.number_input("Number of search results", min_value=10, max_value=100, value=50)
    submit_button = st.form_submit_button("Search")

# Search the web with Serp https://serpapi.com/
def get_search_results(query, num_results):
    params = {
        "q": query,
        "num": num_results,
        "hl": "en",
        "gl": "us",
        "api_key": st.secrets["serpapi_key"] 
    }
    
    search = GoogleSearch(params)
    results = search.get_dict()

    # Extract URLs from the search results
    search_results = [result["link"] for result in results.get("organic_results", [])]
    
    return search_results

# Get Sentiments of webpages with VaderSentiment
def get_sentiment(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        sentimentanalyzer = SentimentIntensityAnalyzer()
        sentiment = sentimentanalyzer.polarity_scores(text)
        retval = [url, sentiment, text]
        
    except:
        #st.write("failed to parse: " + str(url))
        retval = "Failed"
    return(retval)

# Use Open AI to get popular phrases in the positive/ negative/ neutral webpages
def get_popular_words(sentiment_name, urls):
    # Ask OpenAI to find popular words from URLs.
    prompt = (
        f"Here are some websites with {sentiment_name} sentiment: {', '.join(urls)}. "
        f"""What are 10 popular phrases found in these websites that may explain this sentiment? 
        Only list phrases that have to do with {query} and only list phrases with a {sentiment} sentiment. 
        I want to know the reasons the article writers are expressing the sentiment. Don't just say: \"negative or positive sentiment about..\", 
        give concrete examples. These don't need to be but will hopefully give actionable insights for {query}.
        Make sure to select quotes/ paraphrasing from a range of these websites and not just a couple. List the phrases with bullet 
        points but don't inlude the url or source."""
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

# Get the color of a brand - this should be improved maybe with a list of popular brands/ color wheel selector that the user can customize the graph with?
def get_color(query):
    prompt = (
        f"For this query: {query}. There should be a company or product associated with it. "
        f"If there is a color associated with this product/company, return it along with a lighter shade, both in hex strings, in this form: \"#original_color #lighter_color\". "
        f"If no color is associated, return: \"#0ddab2 #0ffbcc\" "
        f"Respond with only the array and nothing else."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    
    # Parse the response content
    try:
        response_text = response.choices[0].message.content.strip()
        #st.write(f"Response Text: {response_text}")
        response_text = json5.loads(response_text)
        if len(response_text) == 2:
            return response_text
        else:
            return ["#0ddab2", "#0ffbcc"]
    except:
        return ["#0ddab2", "#0ffbcc"]  # Default fallback

# Make the sentiment graph
def make_graph(sentiment_list):

    # Extract compound values and URLs
    compounds = [entry[1]['compound'] for entry in sentiment_list]
    urls = [entry[0] for entry in sentiment_list]

    # Define bins
    bins = np.linspace(-1, 1, 11)
    bin_labels = [f"{round(bins[i], 2)} to {round(bins[i+1], 2)}" for i in range(len(bins)-1)]

    # Assign each URL to a bin
    df = pd.DataFrame({'url': urls, 'compound': compounds})
    df['bin'] = pd.cut(df['compound'], bins=bins, labels=bin_labels, include_lowest=True)

    # Count URLs per bin
    bin_counts = df['bin'].value_counts().reindex(bin_labels, fill_value=0)

    # Group URLs by bin as HTML links
    url_map = df.groupby('bin')['url'].apply(
        lambda x: '<br>'.join([f'<a href="{url}" target="_blank">{url}</a>' for url in x])
    ).reindex(bin_labels, fill_value='No URLs in this bin')

    # Data source for the bar chart
    source = ColumnDataSource(data=dict(
        x=bin_labels,
        y=bin_counts.values,
        urls=url_map.values
    ))

    # Sidebar (initially empty)
    sidebar = Div(
        text="<b>Click a bar to see URLs here</b>", 
        width=400, 
        #height=600, 
        style={
            'max-height': '600px',
            'overflow-y': 'auto', 
            'border': '1px solid black', 
            'padding': '10px'
        }
    )

    # Create bar chart
    p = figure(
        x_range=bin_labels,
        height=600,
        width=800,
        title=f"Distribution of Sentiments of Webpages Mentioning {query} - Click a bar to See URLs",
        tools="tap,pan,wheel_zoom,box_zoom,reset,save",
        toolbar_location="above",
        x_axis_label="Sentiment (Negative < -0.05 < Neutral  < 0.05 < Positive )",
        y_axis_label="Count of Webpages"
    )

    # Plot bars and enable selection
    bars = p.vbar(
        x='x', 
        top='y', 
        width=0.9, 
        source=source, 
        color=colors[1], 
        line_color='black', 
        selection_color=colors[0]  # Highlight when clicked
    )

    # Hover effect
    hover = HoverTool(
        tooltips=[("Count", "@y")],
        mode="vline"
    )
    p.add_tools(hover)

    # Sidebar update on click using CustomJS
    bars.data_source.selected.js_on_change("indices", CustomJS(args=dict(source=source, sidebar=sidebar), code="""
        var selected = source.selected.indices[0];
        if (selected !== undefined) {
            var urls = source.data['urls'][selected];
            sidebar.text = urls;  // Update sidebar with URLs
        } else {
            sidebar.text = "<b>Click a bar to see URLs here</b>";
        }
    """))

    # Aesthetics
    p.xgrid.grid_line_color = None
    p.y_range.start = 0
    p.xaxis.major_label_orientation = 1.2

    # Layout: Chart + Sidebar
    layout = row(p, sidebar)
    return(layout)

# When user presses search
if submit_button:

    # progress bar
    my_bar = st.progress(7, text="Fetching search results...")

    search_results = get_search_results(query, num_results)
    progress = 3

    sentiment_list = []
    completed_tasks = 0

    # get sentiments
    if __name__ == '__main__':
        with Pool(7) as p:  # Parallel computing yay - Adjust the number of processes as needed 
            #sentiment_list = p.map(get_sentiment, search_results)
            for result in p.imap_unordered(get_sentiment, search_results):
                sentiment_list.append(result)
                completed_tasks += 1  # Update counter
                progress = min(round((completed_tasks / num_results) * 100), 95) # for progress bar
                my_bar.progress(min(progress,95), text="Analyzing sentiment of webpages...") # progress bar!
                progress += round((100/num_results)*7)

    my_bar.progress(min(progress,95), text="Creating Graph...")
    sentiment_list = [x for x in sentiment_list if x != "Failed"] #removes fails


    # Make graph
    colors = get_color(query) 
    layout = make_graph(sentiment_list)
    st.bokeh_chart(layout, use_container_width=True)

    my_bar.progress(min(progress,100), text="Creating Graph...")

    my_bar.empty()

    # # Save and show
    # output_file("SentimentsGraph.html")
    # save(layout)

    # with open('SentimentsGraph.html', 'r', encoding='utf-8') as file:
    #     sentiment_graph_data = file.read()


    # st.components.v1.html(sentiment_graph_data, height=650, width=1500)

    #st.write(sentiment_list)


    # get popular words
    my_bar_2 = st.progress(0, text="Getting popular phrases...")
    progress = 3


    # Step 1: Group URLs by Sentiment
    sentiment_bins = {"Positive": [], "Negative": [], "Neutral": []}

    for url, sentiment_dict, text in sentiment_list:
        if sentiment_dict['compound'] >= 0.05:
            sentiment_bins["Positive"].append(url)
        elif sentiment_dict['compound'] <= -0.05:
            sentiment_bins["Negative"].append(url)
        else:
            sentiment_bins["Neutral"].append(url)


    # col1, col2, col3 = st.columns([1,1,1])

    # cols = [col1, col2, col3]
    # col = 0
    # Step 2: Loop through each sentiment and print the results

    words_dict = {}
    progress =17
    for sentiment, urls in sentiment_bins.items():
        # with cols[col]:
        if urls:
            words_dict[sentiment] = f"{get_popular_words(sentiment, urls)}"
        else:
            words_dict[sentiment] = f"\nNo URLs for {sentiment} sentiment."
        my_bar_2.progress(progress, text="Getting popular phrases...")
        progress += 25

        # col += 1
    my_bar_2.empty()

    for sentiment, words in words_dict.items():
        st.markdown(f"<h3>Popular phrases for {sentiment} sentiment: </h3>", unsafe_allow_html=True)
        st.write(f"{words}")

# Add a footer
st.markdown(
    """
    <hr>
    <p style = "text-align: center; color: #777; font-size: 14px; font-family: Arial, sans-serif;">
    Made with ❤️ by Bena Smith.
    </p>
    """,
    unsafe_allow_html = True,
)