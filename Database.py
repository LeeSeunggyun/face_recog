import pymongo
from database_access import id, pw, cluster

access_id = id
access_pw = pw
access_cluster = cluster

# Connect to MongoDB Atlas
client = pymongo.MongoClient(
    f"mongodb+srv://{access_id}:{access_pw}@{access_cluster}.tyzdx07.mongodb.net/?retryWrites=true&w=majority&appName={access_cluster}"
)
# mongo
collection = "deepface"
db = client.deepface