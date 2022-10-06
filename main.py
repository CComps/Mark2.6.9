import speech_recognition as sr  # pip install speechrecognition
import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
listener = sr.Recognizer()


def speak(audio):
    print("    ")
    print("-------------------")
    print("    ")
    print(f"Mark: {audio}")
    print("    ")
    print("-------------------")
    print("    ")




def takeCommand():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source, 0, 15)

        try:
            query = r.recognize_google(audio, language='sk-sk')  # en-in
            print(f"your saying: {query}\n")  # User query will be printed.

        except Exception as e:
            return "None"
    return query


lemmatizer = WordNetLemmatizer()
intents = json.loads(open("intents.json", encoding="utf-8").read())

words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))
model = load_model('chatbotmodel.h5')


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words


def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def get_response(intents_list, intents_json):
    tag = intents_list[0]["intent"]
    list_of_intents = intents_json["intents"]
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i["response"])
            break
    return result


speak("Četbot je zapnutý")


while True:
    print("niečo povädz: ")
    message = takeCommand()
    # message = input("niečo povädz: ")

    if "None" in message:
        pass

    elif "Dovidenia" in message:
        break

    else:
        ints = predict_class(message)
        res = get_response(ints, intents)
        with open("log.log", "a") as f:
            f.write(f"Text: {message}; AI: {res};\n")
        speak(res)
