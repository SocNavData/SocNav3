import js
import json
import random
import asyncio

import pyodide

from slider import Slider
import tasks


MAX_ANSWERS = 300
MAX_VIDEOS = 5619-1


videoSource1 = js.document.getElementById('myVideoSource1')
videoSource2 = js.document.getElementById('myVideoSource2')
video = js.document.getElementById('myVideo')
description = js.document.getElementById('myDescription')
canvas = js.document.getElementById('myCanvas')
prev_btn = js.document.getElementById('prev-btn')
next_btn = js.document.getElementById('next-btn')
send_btn = js.document.getElementById('send-btn')
age = js.document.getElementById('age')
country = js.document.getElementById('country')

structure = {
    'answers': None,
    'indices': None,
    'descriptions': None,
    'country': None,
    'age': None,
    'gender': None
}

slider = Slider(canvas, structure)

def reload_video_function(structure):
    js.console.log("Reload!")
    question_index = int(js.eval("questionIndex"))
    js.console.log(f"Question index: {question_index=}")
    text = structure["descriptions"][question_index]
    js.console.log(f"Text {text=}")
    description.innerText = text
    if structure["indices"] is not None and len(structure["indices"]) > question_index:
        videoSource1.src = f"static/videos/{str(structure['indices'][question_index]).zfill(9)}.mp4"
        video.load()
        # video.pause()
    count = js.document.getElementById('myCounter')
    if question_index <= 50:
        count.innerHTML = f"<span style=\"color: #440000\">{question_index+1}/50 (up to {MAX_ANSWERS})<span>"
    else:
        count.innerHTML = f"<span style=\"color: #007700\">{question_index+1}/50 (up to {MAX_ANSWERS})<span>"


def maybe_show_value(slider):
    question_index = int(js.eval("questionIndex"))
    if question_index in structure["answers"].keys():
        # print(structure["answers"].keys())
        js.eval(f"video_watched = 1;")
        slider.set_value(structure["answers"][question_index])
    else:
        js.console.log(f"{question_index} not in structure")
        slider.value = None
    draw()


def load_data(structure):
    global slider
    data = js.window.localStorage.getItem('socnav_data')
    if data:
        # print("There was data saved!")
        data = json.loads(data)
        structure["indices"] = data["indices"]
        structure["descriptions"] = data["descriptions"]
        structure["answers"]  = data["answers"]

        try:
            structure["age"] = data["age"]
        except:
            structure["age"] = 18
            js.alert("There was an issue recovering the demographic information. Please go to the demographic page before submitting the data.")
        js.document.getElementById("age").value = structure["age"]
        js.console.log("age", structure["age"])

        try:
            structure["country"] = data["country"]
        except:
            structure["country"] = "GB"
        js.document.getElementById("country").value = structure["country"]
        js.console.log("country", structure["country"])

        try:
            structure["gender"] = data["gender"]
        except:
            structure["gender"] = "not-say"
        js.document.getElementById("gender").value = structure["gender"]
        js.console.log("gender", structure["gender"])



        for k in [kk for kk in structure["answers"].keys()]:
            structure["answers"][int(k)] = structure["answers"][k]
            del structure["answers"][k]
        # print('answers', structure["answers"])
        question = int(data['questionIndex'])
        # print('question', question)
        js.eval(f"currentPage = {data['currentPage']};")
        js.eval(f"questionIndex = {question};")
        if question in structure["answers"].keys():
            # print("si la")
            js.eval(f"answer_set = 1;")
            js.eval(f"video_watched = 1;")
        else:
            # print("no la")
            js.eval(f"answer_set = 0;")
        js.eval(f"reload_video = 1;")
        js.eval("document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));")
        act = js.document.getElementById(f"page-{data['currentPage']}")
        act.classList.add('active')
        reload_video_function(structure)
        return True
    
    # reload_video_function(structure)
    return False


def save_data(structure):

    try:
        js.console.log("age" + structure["age"])
        js.console.log("country" + structure["country"])
        js.console.log("gender" + structure["gender"])
    except:
        js.console.log("cannot show stuff yet?")
        js.console.log(js.eval("currentPage"))

    data = {
        "indices": structure["indices"],
        "descriptions": structure["descriptions"],
        "answers": structure["answers"],
        "age": structure["age"],
        "gender": structure["gender"],
        "country": structure["country"],
        "currentPage": int(js.eval("currentPage")),
        "questionIndex": int(js.eval("questionIndex")),
        "answer_set": int(js.eval("answer_set")),
        "reload_video": int(js.eval("reload_video")),
        "video_watched": int(js.eval("video_watched"))
    }
    js.window.localStorage.setItem('socnav_data', json.dumps(data))




if load_data(structure) is True:
    js.eval("showPage(currentPage);")
    question = int(js.eval(f"questionIndex;"))
    if question in structure["answers"].keys():
        v = structure["answers"][question]
        slider.set_value(v)
else:
    structure["answers"] = {}
    structure["indices"] = [random.randint(1, MAX_VIDEOS) for _ in range(MAX_ANSWERS)]
    structure["descriptions"] = tasks.generate_descriptions()
    tasks.fix_fixed_tasks(structure)






def getTouchPos(canvas, touchEvent):
    rect = canvas.getBoundingClientRect()
    touch = touchEvent.touches.item(0)
    return touch.clientX - rect.left, touch.clientY - rect.top

def draw(event=None):
    global slider
    slider.draw(event)

def mousedown(event):
    global slider
    slider.on()
    slider.update_pose(event.offsetX, event.offsetY)
    slider.draw(event)
    event.preventDefault()

def touchdown(event):
    global slider
    slider.on()
    x, y = getTouchPos(canvas, event)
    slider.update_pose(x, y)
    slider.draw(event)
    event.preventDefault()

def move(event):
    global slider
    if slider.active is True:
        slider.update_pose(event.offsetX, event.offsetY)
    slider.draw(event)

def touchmove(event):
    global slider
    if slider.active is True:
        x, y = getTouchPos(canvas, event)
        slider.update_pose(x, y)
    slider.draw(event)

def mouseup(event):
    global slider
    slider.off()
    slider.draw(event)

draw(None)



# Attach event listeners to handle drawing
# mousedown
mousedown_proxy = pyodide.ffi.create_proxy(mousedown)
touchdown_proxy = pyodide.ffi.create_proxy(touchdown)
canvas.addEventListener("dragstart",  mousedown_proxy)
canvas.addEventListener('mousedown',  mousedown_proxy)
canvas.addEventListener("touchstart", touchdown_proxy)
# mouseover
mouseover_proxy = pyodide.ffi.create_proxy(move)
touchover_proxy = pyodide.ffi.create_proxy(touchmove)
canvas.addEventListener("mouseover", mouseover_proxy)
canvas.addEventListener("dragmove",  mouseover_proxy)
canvas.addEventListener("mousemove", mouseover_proxy)
canvas.addEventListener("touchmove", touchover_proxy)
# mouseup
mouseup_proxy = pyodide.ffi.create_proxy(mouseup)
canvas.addEventListener("mouseup",     mouseup_proxy)
canvas.addEventListener("mouseout",    mouseup_proxy)
canvas.addEventListener("dragend",     mouseup_proxy)
canvas.addEventListener("touchend",    mouseup_proxy)
canvas.addEventListener("touchcancel", mouseup_proxy)




def watched_handler(event):
    slider.draw()
watched_handler_proxy = pyodide.ffi.create_proxy(watched_handler)
video.addEventListener("ended", watched_handler_proxy)

def prev_button_handler(event):
    reload_video = int(js.eval("reload_video"))
    current_page = int(js.eval("currentPage"))
    if current_page == 5 and reload_video == 1:
        js.eval("reload_video = 0;")
        reload_video_function(structure)
    js.eval("answer_set = 1;")
    js.eval(f"video_watched = 1;")
    js.console.log("in prev_button_handler")
    maybe_show_value(slider)

prev_button_handler_proxy = pyodide.ffi.create_proxy(prev_button_handler)
prev_btn.addEventListener("click", prev_button_handler_proxy)

def next_button_handler(event):
    js.eval("reload_video = 0;")
    reload_video_function(structure)
    question_index = int(js.eval("questionIndex"))
    if question_index in structure["answers"].keys():
        js.eval("answer_set = 1;")
        js.eval(f"video_watched = 1;")
    else:
        js.eval("answer_set = 0;")
        js.eval(f"video_watched = 0;")
    maybe_show_value(slider)

    structure["age"] = js.document.getElementById("age").value
    structure["country"] = js.document.getElementById("country").value
    structure["gender"] = js.document.getElementById("gender").value

    save_data(structure)

next_button_handler_proxy = pyodide.ffi.create_proxy(next_button_handler)
next_btn.addEventListener("click", next_button_handler_proxy)



async def submit_data(event, confirm=False):
    if confirm is False:
        confirmed = js.window.confirm("Are you sure you want to send your ratings and leave the survey at this point?")
    
    if confirmed:
        global structure
        structure['age'] = age.value
        structure['country'] = country.value
        structure["gender"] = js.document.getElementById('gender').value
        str_to_send = json.dumps(structure).replace('"', '\\"')

        a = 'fetch("submit", {\
                method: "POST", \
                headers: { "Content-Type": "text/plain" }, \
                body: "'
        # a = 'fetch("https://vps-fa03b8f8.vps.ovh.net:5421/submit", {\
        #         method: "POST", \
        #         headers: { "Content-Type": "text/plain" }, \
        #         body: "'
        b = '"})\
            .then(response => { \
                if (response.ok) { \
                    window.localStorage.clear(); \
                    document.documentElement.innerHTML = "<h1>Response received. Thanks!</h1>"; \
                    return response.text();  \
                } else { \
                    console.error("Error"); \
                    return Promise.reject("Error"); \
                } \
            }) \
            .then(data => console.log(data)) \
            .catch(error => console.error("Error:", error));'
        js.eval(a +  str_to_send + b)

send_button_handler_proxy = pyodide.ffi.create_proxy(submit_data)
send_btn.addEventListener("click", send_button_handler_proxy)

