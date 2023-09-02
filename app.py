import whisper
import os
import json
import requests
# Text extraction imports
from kor.extraction import create_extraction_chain
from kor.nodes import Object, Text, Number
# LLM import
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
# Web server imports
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
# Twillio imports
from twilio.twiml.voice_response import VoiceResponse
from twilio.rest import Client

os.environ["TWILLIO_ACCOUNT_SID"] = "YOUR API KEY"
os.environ["TWILLIO_AUTH_TOKEN"] = "YOUR API KEY"
os.environ["OPENAI_API_KEY"] = "YOUR API KEY"
openai_api_key = os.environ['OPENAI_API_KEY']
llm = ChatOpenAI(model="gpt-3.5-turbo",
                 temperature=0,
                 max_tokens=800,
                 openai_api_key=openai_api_key)

order_schema = Object(
    id="order",
    description="User is sending in an order for some produce",
    attributes=[
        Text(id="item", description="user want to purchase this item", examples=[
             ("Give me 14 bags of cored pineapples", "cored pineapples"), ("Can I get 12 mushroom kebabs", "mushroom kebabs")]),
        Text(id="quantity", description="user wants this many of the item", examples=[
             ("Give me 14 bags of cored pineapples", "14"), ("Can I get 12 mushroom kebabs", "12")])
    ],
    many=True,
    examples=[
        ("Hi Mani, hows it going. Could I get 6 boxes of cored pineapples, 3 platters of chopped pineapples, 30 mushroom kebabs, and 12 chicken kebabs?", [
         {"item": "cored pineapples", "quantity": "6"}, {"item": "chopped pineapples", "quantity": "3"}, {"item": "mushroom kebabs", "quantity": "30"}, {"item": "chicken kebabs", "quantity": "12"}])
    ]
)
whisperModel = whisper.load_model("base")
app = Flask(__name__)
CORS(app)


@app.route('/transcribe', methods=['POST'])
def transcribe():
    print("Transcribing...")
    filename_wav = "audio_file.wav"
    form_data = request.form
    parsed_object = dict(form_data)
    downloadUrl = parsed_object['RecordingUrl']
    response = requests.get(downloadUrl)
    if response.status_code == 200:
        with open(filename_wav, "wb") as file:
            file.write(response.content)
        print(f"WAV audio file downloaded as '{filename_wav}'")
    else:
        print("Failed to download the audio file")
    result = whisperModel.transcribe(filename_wav, fp16=False)["text"]
    print("Transcription: ", result)
    print("Extracting keywords...")
    chain = create_extraction_chain(llm, order_schema)
    output = chain.run(result)["data"]
    print("--------------------")
    print(output)
    print("--------------------")
    return Response(status=200)


@app.route('/record', methods=['POST'])
def record():
    response = VoiceResponse()
    # Use <Say> to give the caller some instructions
    response.say('Hi, please speak your order after the beep')
    # Use <Record> to record the caller's message
    response.record(recordingStatusCallback='/transcribe')
    # End the call with <Hangup>
    response.hangup()
    return str(response)
