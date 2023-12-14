# LLMWhatsBot

Fine-tune a LLM on WhatsApp data using PEFT (Parameter-Efficient Fine-Tuning) library developed by HuggingFace.


## Usage

First thing to do is to export the WhatsApp chat that you want to use as fine-tuning dataset; a quick guide can be found [here](https://faq.whatsapp.com/1180414079177245/?cms_platform=android).
Once you have exported the chat, save it under a `data/` directory. Then, you need to preprocess it: this can be done by running the `data_prepration.py` script which extracts the queries (basically entire chunks of conversation) and store them into a pandas dataframe.

To fine-tune the LLM model using PEFT you can run the `finetune.py` script. The configuration variables, such as base model id, peft model id (if you want to start already from another peft model loaded in the HuggingFace Hub - I did so to retrieve a small model fine-tuned on italian language), and LORA paramaters, can be set in the `config/training.json`. Inside the `run/` folder, there is also a jupyter notebook named as `run_finetune.ipynb` to run the fine-tuning from Google Colab.

Once the model is fune-tuned you can test it (again inside the `run/` directory there is a simple notebook example (`run_inference.ipynb`) to simulate a chat bot.

