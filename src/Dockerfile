FROM python:2.7.11

# Install packages.
ADD requirements.txt .

# This is here because we want to use caching. It takes a lot to download nltk data.
RUN pip install nltk
# Get NLTK data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data all

RUN pip install -r requirements.txt

