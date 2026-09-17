"""
Velocity Transcript - shared config
Single source of truth for which word folders we're training on.
Both batch_preprocess.py and session_split.py import this, so the
vocabulary only needs to be changed in one place.
"""

WORD_FOLDERS = {
    "Hello": "data/raw/Greetings_1of2/Greetings/48. Hello",
    "How are you": "data/raw/Greetings_1of2/Greetings/49. How are you",
    "Alright": "data/raw/Greetings_1of2/Greetings/50. Alright",
    "Good Morning": "data/raw/Greetings_1of2/Greetings/51. Good Morning",
    "Good afternoon": "data/raw/Greetings_1of2/Greetings/52. Good afternoon",
    "Good evening": "data/raw/Greetings_2of2/Greetings/53. Good evening",
    "Good night": "data/raw/Greetings_2of2/Greetings/54. Good night",
    "Thank you": "data/raw/Greetings_2of2/Greetings/55. Thank you",
    "Pleased": "data/raw/Greetings_2of2/Greetings/56. Pleased",
}