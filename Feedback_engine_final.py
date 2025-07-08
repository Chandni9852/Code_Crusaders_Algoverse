from typing import List, Dict
import pyttsx3

CLASS_LABELS = {
0: "Normal",
1: "Reversal",
2: "Corrected"
}

def generate_feedback(predictions: List[Dict]) -> List[str]:
  """
  Converts model predictions into helpful feedback messages.

  rust
  Copy
  Edit
  Args:
      predictions: List of dictionaries with keys:
                  - 'class' (int)
                  - 'xmin', 'ymin' (coordinates, optional)

  Returns:
      List of feedback strings.
  """
  messages = []

  for pred in predictions:
      cls = pred.get("class")
      x = int(pred.get("xmin", 0))  # Optional position
      y = int(pred.get("ymin", 0))
      label = CLASS_LABELS.get(cls, "Unknown")

      if label == "Reversal":
          messages.append(f"⚠️ Reversed letter detected at ({x}, {y}). Let's try correcting it.")
      elif label == "Corrected":
          messages.append(f"✅ Letter corrected at ({x}, {y}). Nice fix!")
      elif label == "Normal":
          messages.append(f"👍 Properly written letter at ({x}, {y}). Keep going!")

  return messages

def speak_feedback(messages: List[str], rate: int = 150):
  """
  Uses text-to-speech to speak the feedback aloud.

  csharp
  Copy
  Edit
  Args:
      messages: list of strings to read
      rate: speech rate (default = 150 wpm)
  """
  engine = pyttsx3.init()
  engine.setProperty("rate", rate)

  for msg in messages:
      engine.say(msg)

  engine.runAndWait()