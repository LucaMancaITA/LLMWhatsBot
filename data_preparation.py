
# Import modules
import os
import json
from whatstk import WhatsAppChat

from utils.data_utils import (
    remove_emoji_from_username, remove_emoji_from_message,
    dataset_preprocessing
)


# Open the configuration file
with open("./config/training.json", "r", encoding="utf-8") as file:
    config = json.load(file)

# Read the chat txt file
filepath = os.path.join(config["datadir"], "_chat.txt")
chat = WhatsAppChat.from_source(filepath=filepath)
df = chat.df

# Remove emoji
df = remove_emoji_from_username(df, "Fede")
df = remove_emoji_from_message(df)

df = dataset_preprocessing(df)
df.to_csv(os.path.join(config["datadir"], "processed_df.csv"))
