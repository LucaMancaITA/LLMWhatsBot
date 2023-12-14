
# Import modules
import json
import transformers


# Open the configuration file
with open("./config/training.json", "r", encoding="utf-8") as file:
    config = json.load(file)
USER1 = config["users"]["user1"]
USER2 = config["users"]["user2"]

class GenerationCallback(transformers.TrainerCallback):
    """Callback to generate an answer sample during training."""
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 5 == 0:
          input_text = f"{USER1}: Ciao, come va?| {USER2}:"
          input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
          generated_ids = self.trainer.model.generate(input_ids=input_ids.to('cuda'))
          generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
          print("Generated Text:", generated_text)
