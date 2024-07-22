import pandas as pd
import json
import re
import unicodedata
import time
import datetime
from tqdm import tqdm
import asyncio
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import traceback
import uuid

import cohere
from pinecone import Pinecone
from telethon import TelegramClient
from telethon.sessions import StringSession
from pyairtable import Api
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer

#set working directory to the folder with the script
import os
os.chdir(os.path.dirname(os.path.abspath(__file__)))

import nltk
nltk.download('punkt')

keys_path = 'keys/'
data_path = 'TG_data/'

with open(keys_path+'api_keys.json') as f:
  api_keys = json.loads(f.read())

start_date = datetime.datetime.now() - datetime.timedelta(days=4) # minimum date for TelegramClient, to keep in 100K limit.
# set to True if you want to save the pickle file (unreliable, probably due to different pandas versions, better to save to csv)
save_pickle = False

# load TG credentials
api_id = api_keys['api_id']
api_hash = api_keys['api_hash']
session_string = api_keys['session_string']

# load cohere credentials
cohere_key = api_keys['cohere_key_prod']
co = cohere.Client(cohere_key)

# initialize pinecone
pine_key = api_keys['pine_key']
pine_index = api_keys['pine_index']
pc = Pinecone(pine_key)
pine_index = pc.Index(pine_index)


# load airtable credentials
base_id = api_keys['airtable_db']
table_name = api_keys['airtable_table']
api_token = api_keys['airtable_token']
api = Api(api_token)
table = api.table(base_id, table_name)

# Steps (per each channel):
# - identify last_id (channels.csv)
# - download from TG as per last_id
# - process messages: cleaning, deduplicating, summary
# - create embeds
# - date format into int
# - transform into pinecone format
# - upsert into pinecone
# - add into main files (pkl) - optional
# - iterate over channels
# - update last_id in channels.csv
# - create session_stats file
# - update total_stats file

# %% [markdown]
# ============FUNCTIONS============

# clean text
def clean_text(text):
    # Unicode range for emojis
    emoji_pattern = re.compile("["
                               "\U0001F600-\U0001F64F"  # Emoticons
                               "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
                               "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
                               "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
                               "]+", flags=re.UNICODE)

    # Remove emojis
    text = emoji_pattern.sub(r'', str(text))
    # Regular expression for URLs
    url_pattern = re.compile(r"http\S+|www\S+")
    # Remove URLs
    text = url_pattern.sub(r'', str(text))
    # remove /n
    text = text.replace('\n', ' ')
    # Remove any remaining variation selectors
    text = ''.join(char for char in text if unicodedata.category(char) != 'Mn')

    #Remove Foreign Agent text
    pattern = re.compile(r'[А-ЯЁ18+]{3,}\s[А-ЯЁ()]{5,}[^\n]*ИНОСТРАННОГО АГЕНТА')
    text = pattern.sub('', text)
    name1 = 'ПИВОВАРОВА АЛЕКСЕЯ ВЛАДИМИРОВИЧА'
    text = text.replace(name1, '')

    return text

# save to pickle
def save_to_pickle(df, channel):
    # load old pickle if exists
    new_len = df.shape[0]
    try:
        df_old = pd.read_pickle(f'{data_path}/{channel}.pkl')
        df = pd.concat([df_old, df], ignore_index=True)
        df.drop_duplicates(subset=['id'], inplace = True) # remove duplicates
        df.to_pickle(f'{data_path}/{channel}.pkl')
        print(f"Saved {new_len} for {channel} messages to pickle.")
    except:
        df.to_pickle(f'{data_path}/{channel}.pkl')
        print(f"Saved {new_len} messages to pickle.")

# summarize the news (select most important sentences)
def summarize(text, language="russian", sentences_count=2):
    parser = PlaintextParser.from_string(text, Tokenizer(language))
    summarizer = LsaSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join([str(sentence) for sentence in summary])

# NEED MORE FLEXIBLE MODEL
# summarize the news - need to keep length upto 750 characters

def process_new_messages(df, channel, stance):
    # add channel name & stance
    df.loc[:, 'channel'] = channel
    df.loc[:, 'stance'] = stance
    df.loc[:, 'cleaned_message'] = df['message'].apply(clean_text) #remove emojis, urls, foreign agent text
    df.drop_duplicates(subset=['id'], inplace = True) # remove duplicates
    df = df[~df.cleaned_message.str.len().between(0, 30)].copy() #remove empty or too short messages
    # summarize cleaned_messages: 3 sentences if length > 750, 4 sentences if length > 1500
    df.loc[:, 'summary'] = df['cleaned_message'].apply(lambda x: summarize(x, sentences_count=3) if len(x) > 750 else summarize(x, sentences_count=4) if len(x) > 500 else x)
    return df

#function to get new messages from channel
async def get_new_messages(channel, last_id, start_date):
    async with TelegramClient(StringSession(session_string), api_id, api_hash
                            , system_version="4.16.30-vxCUSTOM"
                            ) as client:
        # COLLECT NEW MESSAGES
        data = [] # for collecting new messages
        # check if last_id is integer (=set)
        try:
            offset_id = int(last_id)
        except:
            offset_id = 0
        async for message in client.iter_messages(channel, reverse=True
                                                  , offset_id=offset_id
                                                  , offset_date=start_date):
            data.append(message.to_dict())
        # if no new messages, skip
    print(f"Channel: {channel}, N of new messages: {len(data)}")
    if len(data) == 0:
        return None
    # create df from collected data
    df = pd.DataFrame(data)
    # return df
    return df

# function for COHERE embeddings
# decorator for exponential backoff
@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=1, max=10))
def get_embedding(text, model = 'embed-multilingual-v3.0', input_type = 'clustering'):
    response = co.embed(
        texts = text,
        model = model,
        input_type = input_type
                )
    return response.embeddings


def get_embeddings_df(df, text_col='summary', model="embed-multilingual-v3.0"):
    df.loc[:, 'embeddings'] = get_embedding(
                                    df[text_col].to_list(), 
                                    model=model
                                    )
    print(f"Embeddings for {df.shape[0]} messages collected.")
    return df

@retry(stop=stop_after_attempt(4), wait=wait_random_exponential(multiplier=1, max=10))
def upsert_to_pinecone(df, index, batch_size=100):
    # create df for pinecone
    meta_col = ['cleaned_message', 'summary', 'stance', 'channel', 'date', 'views']
    #rename embeddings to values
    df4pinecone = df[meta_col+['id', 'embeddings']].copy()
    df4pinecone = df4pinecone.rename(columns={'embeddings': 'values'})
    # convert date to integer (as pinecone doesn't support datetime)
    df4pinecone['date'] = df4pinecone['date'].apply(lambda x: int(time.mktime(x.timetuple())))
    # id as channel_id + message_id (to avoid duplication and easier identification)
    df4pinecone['id'] = df4pinecone['channel'] + '_' + df4pinecone['id'].astype(str)
    # convert to pinecone format
    df4pinecone['metadata'] = df4pinecone[meta_col].to_dict('records')
    df4pinecone = df4pinecone[['id', 'values', 'metadata']]
    if df4pinecone.empty:
        print("DataFrame is empty. No records to upsert.")
        return
    for i in range(0, df4pinecone.shape[0], batch_size):
        index.upsert(vectors=df4pinecone.iloc[i:i+batch_size].to_dict('records'))
    print(f"Upserted {df4pinecone.shape[0]} records. Last id: {df4pinecone.iloc[-1]['id']}")

def update_airtable_last_id(channel, last_id):
    matching_records = table.all(formula=f"{{channel_name}}='{channel}'")
    if matching_records:
        if len(matching_records) > 1:
            print(f"Warning: More than one matching record found for channel {channel}. Only the first one will be updated.")
        record_id = matching_records[0]['id']
        table.update(record_id, {'last_id': int(last_id)})
    else:
        print(f"No matching record found for channel {channel}")

def log_summary_to_airtable(parsed_channels, missed_channels, new_channels):
    log_table_name = api_keys['airtable_log_table']
    log_table = api.table(base_id, log_table_name)
    
    execution_date = datetime.datetime.now()
    primary_key = f"{execution_date.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    
    summary = {
        'id': primary_key,  # This will be the primary key
        'execution_date': execution_date.isoformat(),
        'parsed_channels_count': len(parsed_channels),
        'parsed_channels': ', '.join(parsed_channels),
        'missed_channels_count': len(missed_channels),
        'missed_channels': ', '.join(missed_channels),
        'new_channels_count': len(new_channels),
        'new_channels': ', '.join(new_channels)
    }
    
    log_table.create(summary)
    
    print(f"Execution summary logged with ID: {primary_key}")

# ===RUN======================================================================
# create session_stats
df_channel_stats = pd.DataFrame() # fix N of posts per channel per day
session_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") # to name session stats file

# ITERATE OVER CHANNELS (from Airtable) TO UPDATE PINCONE INDEX
records = table.all()
df_channels = pd.DataFrame([record['fields'] for record in records])
df_channels = df_channels[df_channels['status'] == 'Active'] # keep only active channels

parsed_channels = []
missed_channels = []
new_channels = []

for i, channel, last_id, stance in tqdm(df_channels[['channel_name', 'last_id', 'stance']].itertuples(), total=df_channels.shape[0], desc="Processing channels"):
    print(f"Starting channel: {channel}, last_id: {last_id}")
    try:
        # get & clean new messages
        df = asyncio.run(get_new_messages(channel, last_id, start_date=start_date))
        if df is None or df.empty:
            continue
        
        # clean, summarize, add channel name & stance
        df = process_new_messages(df, channel, stance)
        
        # get embeddings
        df = get_embeddings_df(df, text_col='summary', model="embed-multilingual-v3.0")
        
        # upsert to pinecone
        upsert_to_pinecone(df, pine_index)

        # save session stats for channel
        df_channel_stats[channel] = df['date'].dt.date.value_counts()
        df_channel_stats.to_csv(f'session_stats/channel_stats_{session_time}.csv', sep=';', index=True)

        # Update last_id in Airtable
        new_last_id = df['id'].max()
        update_airtable_last_id(channel, new_last_id)
        
        parsed_channels.append(channel)
        
        # Check if this is a new channel (last_id was empty or zero AND we parsed messages)
        if (last_id == '' or last_id == '0' or last_id == 0) and len(df) > 0:
            new_channels.append(channel)

    except Exception as e:
        missed_channels.append(channel)
        print(f"!!! ERROR occurred with channel {channel}: {str(e)}")
        traceback.print_exc()
        continue

# After the loop, log the summary to Airtable
log_summary_to_airtable(parsed_channels, missed_channels, new_channels)

print(f"Parsed channels: {len(parsed_channels)}")
print(f"Missed channels: {len(missed_channels)}")
print(f"New channels: {len(new_channels)}")