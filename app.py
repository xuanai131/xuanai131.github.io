#!/home/pi/Python_env/LLM/bin/python3
from flask import Flask
from flask import render_template
import sys
from flask import request,Response,redirect, url_for,jsonify
from urllib.parse import urlparse
import cv2
import base64
import numpy as np
import time
import threading
from flask_socketio import SocketIO
import requests
import os 
from openai import OpenAI
# import setting
# import soundfile as sf
# import sounddevice as sd
import multiprocessing
import urllib.request
import re
import json
from colorama import Fore, Back, Style
# from gevent import monkey
# monkey.patch_all()
sys.path.append("../Bot")
from Database_handle import *
from Global_variable import *
import Helper_Utilities
from langchain_core.messages import HumanMessage, AIMessage
# import voice_record
import barcode
from barcode.writer import ImageWriter
from PIL import Image
from io import BytesIO
import base64
import uuid
import sys
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
import setting
import threading
import time
import logging
from Tools import *
reset_time = 60
timer = None
image_queue = []
lock = threading.Lock()
new_frame_event = threading.Event()
OpenAIHistoryConversation = []
graph = Helper_Utilities.CreateGraph(OpenAIHistoryConversation)
# print(graph)
# DoRecord = voice_record.Voice_Record()
# sys.path.append("database")

app = Flask(__name__)
socketio = SocketIO(app)
# from Voice_handle import VoiceHandle
response = ""
response_tool = ""
response_to_rasp = None
camera_st = False
user_input_st = False
user_input_interrupt_signal = False
user_input_message = ""
SavedHistoryConversation = []  # Conversation to save when create new session
# def reset_list():
#     global SavedHistoryConversation
#     print('[app] ', Fore.RED +"called reset history and go back to the waiting position")
#     print('[app] ', Style.RESET_ALL)
#     with lock:
#         SavedHistoryConversation.clear()
# def reset_timer():
#     global timer
#     print('[app] ', Fore.RED +"called reset timer")
#     if timer:
#         timer.cancel()
#     timer = threading.Timer(reset_time, reset_list)
#     timer.start()
#     print('[app] ', Style.RESET_ALL)
# def append_item(item):
#     global timer
#     with lock:
#         SavedHistoryConversation.append(item)
# #         reset_timer()
# log = logging.getLogger('werkzeug')
# log.setLevel(logging.ERROR) 
def run_graph(inputs):
    for s in graph.stream(inputs):
        if "__end__" not in s:
            # print("[app] s in if statement: ",s)
            print('[app] ------------------------')
        else:
            # print("[app] s in else statement: ", s)
            try:
                answer = AIMessage(s.get('__end__')['messages'][-1].content)
                print(answer)
                attitude = s.get('__end__')['inspector']
                print("[app] The inspector check that the response is ",attitude)
                res = handle_event_response(attitude,answer)
                return res
            except:
                pass

def get_Chat_response(text):
    query = HumanMessage(text)
    OpenAIHistoryConversation.append(query)
    # reset_timer()
    inputs = {
        "messages": OpenAIHistoryConversation
    }
    
    answer = run_graph(inputs)
    print("[app] answer of run_graph", answer)
    OpenAIHistoryConversation.append(answer)
    # reset_timer()
    return answer.content
# voicehandle = VoiceHandle(wake_words=['porcupine', 'jarvis'],
#                           get_chat_response_func=get_Chat_response)
# voicehandle.run()

def handle_event_response(attitude,answer):
    global OpenAIHistoryConversation,redirect_state
    print("[app] check history: ",OpenAIHistoryConversation)
    if (attitude == "bad"):
        print('[app] ', Fore.RED +"in the bad response")
        print('[app] ', Style.RESET_ALL)
        redirect_state = "Book_researcher"
        Helper_Utilities.write_state(redirect_state)
        inputs = {
        "messages": OpenAIHistoryConversation
    }
        res  = run_graph(inputs)
        return res
    else:
        redirect_state = "supervisor"
        Helper_Utilities.write_state(redirect_state)
        return answer

@app.route("/")
def hello_world():
    return render_template('index.html',camera_state=camera_st)

def LoadBookCovers(book_ids):
    images = []
    for book_id in book_ids:
        images.append(SearchBookByID(book_id))
    return images

@app.route("/image", methods=["GET","POST"])
def get_image():
    global image_of_book
    if request.method == 'POST':
        msg = request.get_json()
        print("[app] MSG : ", msg['id'])
        image_of_book = re.sub(r'[^0-9,]', '', msg['id']).split(',')
        # print(image_of_book)
        # print(type(image_of_book))

        images = LoadBookCovers(image_of_book)
        # print(type(images[0]))
        # return render_template('index.html')
        socketio.emit('book_images', {'visible': True, 'image' : images})
        return "load success"
        # video_feed_url = url_for('image')
        # return render_template('index.html', video_feed_url = "response")
    else:
        socketio.emit('book_images', {'visible': False, 'image' : images})
        return image_of_book
    
@app.route("/user_input_mess",methods = ["POST","GET"])
def get_message_user_tool():
    global user_input_message
    if request.method == 'POST':
        data = request.form.get("msg")
        print("[app] /user input message posted ",data)
        # message = data["message"]
        user_input_message = data
        # socketio.emit('user_input_message',{"message":message})
        return "posted data."
    else:
        return  str(user_input_message)
@app.route("/user_input_state",methods = ["POST","GET"])
def get_user_input_state():
    global user_input_st
    global user_input_message
    if request.method == 'POST':
        data = request.get_json()
        user_input_message = ""  
        print("[app] User input state posted ",data)
        if data and "input_st" in data:
            state = data["input_st"]
            user_input_st = state
            socketio.emit('user_input_state',{"state":state})
            if (state == True):
                while(user_input_message == ""):
                    continue
                print("[app] Checking the message is: ",user_input_message)
                result = user_input_message
                user_input_message = ""
                return result
            else:
                return "state is false"
        else:
            return "Invalid JSON data."
    else:
        return str(user_input_st)
@app.route("/user_input_state_interrupt",methods = ["POST","GET"])
def get_user_input_state_interrupt():
    global user_input_message
    if request.method == 'POST':
        print("[app] Change request to the barcode scan")
        user_input_message = "***INTERRUPT***" 
    else:
        return user_input_message
    return "success"

@app.route("/return_form",methods = ["POST","GET"])
def return_form():
    if request.method == 'POST':
        data = request.get_json()
        print('[app] Received data', data)
        if data['message'] == 'start':
            socketio.emit('return_form_visiblity', {'visible': True})
        else:
            socketio.emit('return_form_visiblity', {'visible': False})
        return data
@app.route("/student-book_info",methods = ["POST","GET"])
def student_book_info():
    if request.method == 'POST':
        data = request.get_json()  
        socketio.emit('student-book_info_socket', data)
        return ''
@app.route("/borrow_book_student_info",methods = ["POST","GET"])
def return_student_info():
    if request.method == 'POST':
        data = request.get_json()  
        socketio.emit('return_student_info_socket', data)
        return ''
@app.route("/tool_action", methods=["POST","GET"])
def chat_from_tool():
    global response_tool
    global response_to_rasp
    if request.method == 'POST':
        data = request.get_json()  # Get JSON data from the request
        if data and "message" in data:
            msg = data["message"]
            if msg:
                SavedHistoryConversation.append("Lib : "+ msg )
                # reset_timer()
                # send_mess(msg)
                # response = msg
                response_to_rasp = msg
                # voicehandle.response_generated_by_app = msg
                t = time.localtime(time.time())
                # voicehandle.response_generated_by_app = msg
                socketio.emit('update_html', {'data': msg,"time": str(t.tm_hour)+ " "+ str(t.tm_min) + " "+str(t.tm_sec)})
                return response 
            else:
                return  "No message received."
        else:
            return "Invalid JSON data."
    else:
        return  str(response_tool)

# Save history when create new session
@app.route("/saved_history", methods=["GET","POST"])
def saved_history():
    if request.method == 'POST':
        filename = "history.json"
        print("--------------[app] Something was sent to saved_history---------------")
        with open(filename,'r+', encoding='utf-8') as file:
            # First we load existing data into a dict.
            file_data = json.load(file)
            # Join new_data with file_data inside emp_details
            file_data["HISTORY"].append(SavedHistoryConversation)
            # Sets file's current position at offset.
            file.seek(0)
            # convert back to json.
            json.dump(file_data, file, indent = 10, ensure_ascii=False) 
        
        SavedHistoryConversation.clear()
        OpenAIHistoryConversation.clear()
        return "Invalid JSON data."

@app.route("/get", methods=["GET","POST"])
def chat():
    global response
    global response_to_rasp
    # response = ""
    if request.method == 'POST':
        msg = request.form.get("msg")
        print("[app] Message: ", msg)
        if msg:
            SavedHistoryConversation.append("User : "+ msg )
            # reset_timer()
            response = get_Chat_response(msg)
            response_to_rasp = response
            SavedHistoryConversation.append("Lib : "+ response )
            # reset_timer()
            # send_mess(response)
            # voicehandle.response_generated_by_app = response
            return response
        else:
            return "No message received."
    else:
        return response

def generate_frames(image_data):
    global count
    if not image_data:
        print("[app] Received empty image data")
        return None

    try:
        # Decode received image from base64
        image_bytes = base64.b64decode(image_data)
        np_arr = np.frombuffer(image_bytes, np.uint8)

        if np_arr.size == 0:
            print("[app] Received corrupted image data")
            return

        # Decode image using OpenCV
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        # count += 1
        # cv2.imwrite(f'images/image_{count}.jpg', frame)

        # Convert frame to jpeg format
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        return frame

        # yield (b'--frame\r\n'
        #        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    except Exception as e:
        print("Error decoding image:", e)
@app.route('/listening_for_query', methods=['POST','GET'])
def voice_status_background_update():
    if request.method == 'POST':
        data = request.get_json()
        socketio.emit('voice_status_background', data)
    return ''
@app.route('/query_voice', methods=['POST','GET'])
def voice_query_background_update():
    if request.method == 'POST':
        data = request.get_json()
        socketio.emit('query_voice_background', data)
        print('[app] query_voice_background ', data)
    return ''


# def send_mess(str, topic="response_to_raspberry0"):
#     print('[app] send_mess with message', str)
#     requests.post(url=setting.IP_ADDRESS+"/"+topic, json={"message":str})

@app.route('/response_to_raspberry0', methods=['POST','GET'])
def response_to_raspberry():
    # global response
    global response_to_rasp
    # response_to_send = ""
    # print("[app] response_to_raspberry : ", response_to_rasp)
    if request.method == 'GET':
        if response_to_rasp != None:
            print("[app] response_to_raspberry : ", response_to_rasp)
            msg = response_to_rasp
            response_to_rasp = None
            return {'response': msg}
        else:
            return {'response': None}
    # if request.method == 'GET':
    #     if response_to_send != "": return response_to_send

    return ""
        


update_status_flag = 0
@app.route('/update_status_from_voice_button', methods=['POST','GET'])
def update_from_voice_button():
    global update_status_flag
    if request.method == 'POST':
        # print('voicehandle.listening_for_query', voicehandle.listening_for_query)
        # if voicehandle.listening_for_query == False:
        #     voicehandle.responding_to_user = False
        #     voicehandle.listening_for_wake_word = False
        #     voicehandle.listening_for_query = True
        #     # print('voicehandle.responding_to_user: ', voicehandle.responding_to_user)
        # else:
        #     voicehandle.reset_all()
        # print("[app] ----------------------------")
        update_status_flag = 1
        # return "eeee"
    if request.method == 'GET':
        if update_status_flag == 1:
            update_status_flag = 0
            return '1'
        else:
            return '0'
    return ''
@app.route('/camera_status', methods=['POST','GET'])
def camera_status_update():
    global camera_st
    if request.method == 'POST':
        data = request.form['camera_status']
        print('[app] Image data', str(data))
        if (str(data) == "True"):
            camera_st = True
            socketio.emit('container_visibility_change', {'visible': True})
        else:
            camera_st = False
            socketio.emit('container_visibility_change', {'visible': False})
        return 'Image received'
    else:
        return str(camera_st)


@app.route('/camera', methods=['POST'])
def camera():
    global image_queue
    
    image_data = request.form['image_base64']
    frame = generate_frames(image_data)
    if frame is not None:
        image_queue.append(frame)
    new_frame_event.set()
    return 'Image received'

def generate():
    global image_queue
    while True:
        new_frame_event.wait()  # Wait for new frames
        new_frame_event.clear()  # Reset event
        while image_queue:
            with lock:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + image_queue.pop(0) + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
@app.route('/download_audio', methods=['POST'])
def download_audio_from_url():
    data = request.get_json().get('url')
    # print("the url in front end side :",data)
    # data = "https://chunk.lab.zalo.ai/a745e9c971a198ffc1b0/a745e9c971a198ffc1b0/"
    # download_audio_in_web(data,'audio.wav')
    # data = request.get_json().get('data')
    # text_to_speech(data,'audio.wav')
    audio_thread = threading.Thread(target=text_to_speech, args=(data,'audio.wav'))
    audio_thread.start()
    audio_thread.join()
    return "success"
@app.route('/user_info', methods=['POST','GET'])
def receive_user_signup_info():
    if request.method == 'POST':
        username = request.form.get("username")
        email = request.form.get("email")
        phone= request.form.get("phone")
        is_exist = SearchAllbyUsername(email)
        id = generate_barcode_base64()
        if (is_exist):
            # print("////////////////// username: ", is_exist)
            return "0"
        else:
            InsertUserInfo(username, email, phone, id)
            send_img_to_email(id, email)
            return "1"
    else:
        return response
    
def generate_barcode_base64():
    existing_IDs = SearchAllAccountBarcode()
    # existing_IDs = []
    while True:
        unique_id = uuid.uuid4().int
    
    # Convert the ID to a string and ensure it fits the length for EAN13
        id = str(unique_id)[:12]
        if id not in existing_IDs:
            break
    # Generate barcode
    EAN = barcode.get_barcode_class('ean13')  # You can change the barcode type if needed
    ean = EAN(id, writer=ImageWriter())
    
    # Save barcode to a BytesIO object
    buffer = BytesIO()
    ean.write(buffer, options={'write_text': False})  # You can add options as needed
    
    # Rewind the buffer to the beginning
    buffer.seek(0)
    
    # Open the image with PIL and convert it to Base64
    image = Image.open(buffer)
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    return img_str

def send_img_to_email(base64_image, receiver_email):
    sender_email = 'ronin792002@gmail.com'
    # receiver_email = 'vuvu3921@gmail.com'
    password = 'Lamvu2002'

    # Create the MIMEMultipart object
    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = 'Test Email from Python'

    # Email body
    body = 'This is a test email sent from Python script.'
    # base64_image = 'iVBORw0KGgoAAAANSUhEUgAAAAUA... (rest of the base64 string)'
    msg.attach(MIMEText(body, 'plain'))

    # Decode the base64 string
    image_data = base64.b64decode(base64_image)

    # Create a MIMEBase object
    part = MIMEBase('application', 'octet-stream')
    part.set_payload(image_data)

    # Encode the payload using base64 encoding
    encoders.encode_base64(part)

    # Add header to the attachment
    part.add_header('Content-Disposition', 'attachment; filename="image.png"')

    # Attach the MIMEBase object to the MIMEMultipart object
    msg.attach(part)

    # Attach the body with the msg instance
    # msg.attach(MIMEText(body, 'plain'))

    try:
        # Create SMTP session
        server = smtplib.SMTP('smtp.gmail.com', 587)  # Use Gmail's SMTP server
        server.starttls()  # Enable security

        # Login to the server
        server.login(sender_email, "ojeq xnrh rwbg sqxj")

        # Send the email
        server.send_message(msg)

        # Terminate the session
        server.quit()

        print('[app] Email sent successfully!')

    except Exception as e:
        print(f'[app] Failed to send email: {e}')

@app.route('/type_of_book', methods=['POST','GET'])
def get_infomation_of_all_book():
    if request.method == 'POST':
        kind = request.form.get("type")
        page = request.form.get("page")
        print("[app] kind ", kind)
        print("[app] page ", page)
        result = SearchAllBookbyKindOfBook(kind)
        len_of_type = len(result)
        print('[app] Result of book kind search: ', len(result))
        print('[app] Type of result: ', type(result))
        start_index = (int(page)-1)*20
        end_index = int(page)*20
        ans = [result[start_index:end_index], len_of_type]
        return ans
    else:
        return response

@app.route('/set_goal_result', methods=['POST','GET'])
def goal_reached():  
    if request.method == 'POST':
        # status = request.form.get("status")
        data = request.get_json()  # Get JSON data from the request
        print(data)
        if data and "status" in data:
            status = data["status"]
            print("[app] receive goal result")
            if status == "Reached":
                requests.post(url=setting.IP_ADDRESS+"/"+"tool_action", json={"message":"Đã đến nơi ! Bạn có cần tôi giúp gì nữa không"})
            else:
                requests.post(url=setting.IP_ADDRESS+"/"+"tool_action", json={"message":"Xin lỗi đã xảy ra một số sự cố trong quá trình di chuyển. Hiện tại tôi không thể chỉ dẫn cho bạn"})
        return "Done"

    else:
        return "Get method"

@app.route('/check_voice_IP', methods=['POST','GET'])
def check_voice_IP():  
    if request.method == 'POST':
        return '1'

if __name__ == '__main__':
    host = setting.IP if len(setting.IP) > 1 else 'localhost'
    port = int(sys.argv[2]) if len(sys.argv) > 2 else 5001
    socketio.run(app, host=host, port=port)
    # socketio.run(app, host="0.0.0.0", port=port)





