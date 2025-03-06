
# Import Packages 
# Searching and Parsing 
#import googlesearch
from googlesearch import search
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

import json

# Connect to OpenAI and Analyze Popular Words
client = openai.Client(api_key=st.secrets["openaikey"])


with st.form("Form entry"):
    query = st.text_input("Search Bar", value="Spotify Wrapped Opinions")
    num_results = st.number_input("Number of search results", min_value=10, max_value=100, value=50)
    submit_button = st.form_submit_button("Search")


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

    
def get_popular_words(sentiment_name, urls):
    # Ask OpenAI to find popular words from URLs.
    prompt = (
        f"Here are some websites with {sentiment_name} sentiment: {', '.join(urls)}. "
        f"What are 10 popular phrases found in these websites that may explain this sentiment? Only list phrases that have to do with {query}. I want to know the reasons the article writers are expressing the sentiment. Don't just say: \"negative or positive sentiment about..\", give concrete examples. These don't need to be but will hopefully tell us what {query} can improve upon or what they are doing correctly. Make sure to select quotes/ paraphrasing from a range of these websites and not just a couple. List the phrases with bullet points but don't inlude the url."
    )
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()

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
        colors = response_text.split(" ")  # Remove quotes and split into a list
        return colors
    except:
        return ["#0ddab2", "#0ffbcc"]  # Default fallback


if submit_button:

    # Search query
    my_bar = st.progress(0, text="Fetching search results...")

    search_results = search(query, num_results=num_results)


    progress = 3

    sentiment_list = []

    # get sentiments
    for url in search_results:
        my_bar.progress(min(progress,100), text="Analyzing sentiment of webpages...")
        sentiment = get_sentiment(url)
        if sentiment != "Failed":
            sentiment_list.append(sentiment)
        progress += round(100/num_results)


    my_bar.empty()

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
        height=600, 
        style={
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
        color=get_color(query)[1], 
        line_color='black', 
        selection_color=get_color(query)[0]  # Highlight when clicked
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

    st.bokeh_chart(layout, use_container_width=True)


    # # Save and show
    # output_file("SentimentsGraph.html")
    # save(layout)

    # with open('SentimentsGraph.html', 'r', encoding='utf-8') as file:
    #     sentiment_graph_data = file.read()


    # st.components.v1.html(sentiment_graph_data, height=650, width=1500)

    #st.write(sentiment_list)




    # Step 1: Group URLs by Sentiment
    sentiment_bins = {"Positive": [], "Negative": [], "Neutral": []}

    for url, sentiment_dict, text in sentiment_list:
        if sentiment_dict['compound'] >= 0.05:
            sentiment_bins["Positive"].append(url)
        elif sentiment_dict['compound'] <= -0.05:
            sentiment_bins["Negative"].append(url)
        else:
            sentiment_bins["Neutral"].append(url)



    # Step 2: Loop through each sentiment and print the results
    for sentiment, urls in sentiment_bins.items():
        if urls:
            words = get_popular_words(sentiment, urls)
            st.write(f"\nPopular phrases for {sentiment} sentiment:\n{words}")
        else:
            st.write(f"\nNo URLs for {sentiment} sentiment.")
