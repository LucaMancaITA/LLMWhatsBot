
# Import modules
import emoji
import pandas as pd


def remove_emoji_from_username(data_frame, user_name):
    """Remove emoji from the username column.

    Args:
        data_frame (pd.DataFrame): input dataframe. Columns are: date,
                                   username and message.
        user_name (str): username.

    Returns:
        pd.DataFrame: processed dataframe.
    """
    data_frame.loc[
        data_frame['username'].str.contains(user_name), 'username'] = user_name

    return data_frame


def remove_emoji_from_message(data_frame):
    """Remove emoji from message column.

    Args:
        data_frame (pd.DataFrame): input dataframe. Columns are: date,
                                   username and message.

    Returns:
        pd.DataFrame: processed dataframe.
    """
    data_frame = data_frame[data_frame["message"] != "\u200eimage omitted\n\u200e"]
    data_frame = data_frame[data_frame["message"] != "\u200eimage omitted"]
    data_frame['message'] = data_frame['message'].apply(lambda s: emoji.replace_emoji(s, ''))
    data_frame.reset_index(inplace=True, drop=True)

    return data_frame


def dataset_preprocessing(data_frame):
    """Dataset preprocessing to prepare it for LLM fine-tuning.

    Args:
        data_frame (pd.DataFrame): input dataframe. Columns are: date,
                                   username and message.

    Returns:
        pd.DataFrame: preprocessed dataframe.
    """
    # Calculate time passed since previous message
    data_frame["date_previous"] = data_frame["date"].shift(periods=1)
    data_frame["time_delta"] = (data_frame["date"]-data_frame["date_previous"]).dt.total_seconds()

    # Concat message and author
    data_frame["chat_message"] = data_frame["username"] + ": " + data_frame["message"]

    # Remove first line, its just a WhatsApp test line
    data_frame = data_frame[1:]

    # Convert messages into conversations (a conversation has multiple messages)
    # Step 1: Concat each message with the previous conversation
    query = []
    answer = []
    conversation = ""
    session_ix = 0
    sessions_ixs = []

    for _, row in data_frame.iterrows():
        if row["time_delta"] < 3600: # This defines on how close messages should be to be in the same conversation
            session_ix = session_ix + 1
            sessions_ixs.append(session_ix)
            if conversation == "":
                conversation = row["chat_message"]
                query.append(conversation)
                answer.append("")
            else:
                conversation = conversation + "| " + row["chat_message"]
                query.append(conversation)
                answer.append(row["chat_message"])
        else:
            session_ix = 0
            conversation = ""


    out_data_frame = pd.DataFrame(
        data={
            "query": query[:-1],
            "answer": answer[1:],
            "session_ix": sessions_ixs[:-1]
        }
    )

    # Step 2: Filter only for the last message of the conversation
    out_data_frame["model_helper_idx"] = out_data_frame["session_ix"] - out_data_frame["session_ix"].shift(-1)
    out_data_frame = out_data_frame[out_data_frame["model_helper_idx"]>-1]

    # Reset dataframe index
    out_data_frame.reset_index(inplace=True, drop=True)

    return out_data_frame["query"]
