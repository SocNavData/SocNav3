import js
import math

MIN_ANSWERS = 50

def project_point_to_segment(x1, y1, x2, y2, x, y):
    # Calculate the square of the length of the segment
    segment_length_squared = (x2 - x1) ** 2 + (y2 - y1) ** 2
    # If the segment length is zero, return the distance between the point and the single endpoint
    if segment_length_squared == 0:
        return math.sqrt((x - x1) ** 2 + (y - y1) ** 2)
    # Calculate the projection of point (x, y) onto the line defined by the segment
    t = ((x - x1) * (x2 - x1) + (y - y1) * (y2 - y1)) / segment_length_squared
    # Clamp t to the range [0, 1] to ensure the projection falls on the segment
    t = max(0, min(1, t))
    # Find the coordinates of the projection point on the segment
    prj_x = x1 + t * (x2 - x1)
    prj_y = y1 + t * (y2 - y1)
    return prj_x, prj_y, t


def distance_point_to_segment(x1, y1, x2, y2, x, y):
    prj_x, prj_y, _ = project_point_to_segment(x1, y1, x2, y2, x, y)
    # Calculate the distance between the point and the projection
    distance = math.sqrt((x - prj_x) ** 2 + (y - prj_y) ** 2)    
    return distance


class Slider(object):
    def __init__(self, canvas, structure):
        super().__init__()
        self.value = None
        self.canvas = canvas
        self.radius = 14
        self.shown_message = False
        self.ctx = self.canvas.getContext('2d')
        self.structure = structure
        self.off()

    def draw(self, event=None):
        canvas = self.canvas
        radius = self.radius
        x_start = int(2*radius)
        x_end = int(canvas.width - 2*radius)
        ctx = self.ctx
        ctx.clearRect(0, 0, canvas.width, canvas.height)

        video_watched = int(js.eval("video_watched"))
        if video_watched != 1:
            return

        ctx.strokeStyle = "black"

        ctx.clearRect(0, 0, canvas.width, canvas.height)

        slider_offset = 15

        ctx.lineWidth = 2
        ctx.beginPath()
        ctx.moveTo(x_start, canvas.height/2+slider_offset)
        ctx.lineTo(x_end, canvas.height/2+slider_offset)
        ctx.moveTo(x_start, canvas.height/2-10+slider_offset)
        ctx.lineTo(x_start, canvas.height/2+10+slider_offset)
        ctx.moveTo((x_start+x_end)//2, canvas.height/2-7+slider_offset)
        ctx.lineTo((x_start+x_end)//2, canvas.height/2+7+slider_offset)
        ctx.moveTo(x_end, canvas.height/2-10+slider_offset)
        ctx.lineTo(x_end, canvas.height/2+10+slider_offset)
        ctx.stroke()

        if self.value is not None:
            ctx.beginPath()
            x = x_start + self.value*(x_end-x_start)
            y = canvas.height//2+slider_offset
            ctx.arc(int(x), int(y), int(radius), 0, 6.28)
            r = str(hex(int((1-self.value)*255))).split('x')[-1].zfill(2)
            g = str(hex(int((  self.value)*255))).split('x')[-1].zfill(2)
            b = "00"
            ctx.lineWidth = 1
            ctx.fillStyle = "#"+r+g+b+"FF"
            ctx.fill()
            ctx.stroke()

        ctx.strokeStyle = "black"
        ctx.fillStyle = "#000000"
        ctx.font = "20px serif"
        ctx.textAlign = "left"
        ctx.textBaseline = "top"
        ctx.fillText("extremely", 3,                0)
        ctx.fillText("bad", 3,                26)
        ctx.textAlign = "center"
        ctx.fillText("fair",  canvas.width//2,  0)
        ctx.textAlign = "right"
        ctx.fillText("extremely",  canvas.width-3,     0)
        ctx.fillText("good",       canvas.width-3,    26)


    def update_pose(self, x, y):
        video_watched = int(js.eval("video_watched"))
        # if video_watched != 1:
        #     js.window.confirm("Please, watch the video before providing a rating.")
        #     return

        radius = self.radius
        x_start = int(2*radius)
        x_end = int(self.canvas.width - 2*radius)

        x1 = x_start
        y1 = radius
        x2 = x_end
        y2 = radius
        _, _, t = project_point_to_segment(x1, y1, x2, y2, x, y)
        self.set_value(t)

    def set_value(self, v):
        question_index = int(js.eval("questionIndex"))
        self.value = v
        self.structure["answers"][question_index] = self.value
        if question_index == MIN_ANSWERS - 1 and self.shown_message == False:
            js.alert(f"Thank you for submitting your {MIN_ANSWERS} ratings. Feel free to rate more trajectories. When you are done, please click on \"send\".")
            self.shown_message = True
        js.eval("answer_set = 1;")

    def on(self):
        self.active = True

    def off(self):
        self.active = False
