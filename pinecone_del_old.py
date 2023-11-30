import pinecone
import json
from datetime import datetime, timedelta

keys_path = 'keys/'

with open(keys_path+'api_keys.json') as f:
  data = json.loads(f.read())

# load pinecone credentials
pine_key = data['pine_key']
pine_env = data['pine_env']

# initialize pinecone
pinecone.init(api_key=pine_key, environment=pine_env)
index_name = 'tg-news'

index = pinecone.Index(index_name)

# remove records from index older than N days
days=40
end_date = (datetime.today() - timedelta(days=days)).strftime('%Y-%m-%d')
end_date = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp())
# create dump embedding with all zeros of 1536 dimensions
zero_emb = [0]*1536

filter = {
    "date": { "$lte": end_date}
    }

# loop until all records older than 30 days are deleted
num_deleted = 0
while True:
    # get 1K random ids of records older than 30 days
    res = index.query(zero_emb, top_k=1000, include_metadata=False, filter=filter)
    if res['matches'] == []:
        print(f'No more records older than {days} days')
        print(f'Deleted {num_deleted} records older than {days} days')
        print('!!!! FINISHED !!!!')
        break
    ids = [r.id for r in res['matches']]
    delete_response = index.delete(ids=ids)
    if delete_response == {}: 
        num_deleted += len(ids)
        print(f'Deleted {len(ids)} records older than {days} days')


