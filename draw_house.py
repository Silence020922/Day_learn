from turtle import *
from random import random,randint
bgcolor("black")
mode("standard")
home()
speed(0)

# Bottom rectangle
color("saddlebrown")
penup()
goto(-200, -285)
pendown()
begin_fill()
for i in range(2):
    forward(400)
    left(90)
    forward(20)
    left(90)
end_fill()

# Second from bottom rectangle
penup()
goto(-175, -265)
pendown()
color("chocolate")
begin_fill()
for i in range(2):
    forward(350)
    left(90)
    forward(20)
    left(90)
end_fill()

# Main part of building
penup()
goto(-150, -245)
pendown()
color("sandybrown")
begin_fill()
for i in range(2):
    forward(300)
    left(90)
    forward(335)
    left(90)
end_fill()

# Second from top rectangle
penup()
goto(-175, 90)
pendown()
color("chocolate")
begin_fill()
for i in range(2):
    forward(350)
    left(90)
    forward(20)
    left(90)
end_fill()

# Top rectangle
penup()
goto(-150, 110)
pendown()
color("sienna")
begin_fill()
for i in range(2):
    forward(300)
    left(90)
    forward(20)
    left(90)
end_fill()

# Windows
x = -125
y = 30
light_color = ['black','khaki']
def window():
    global x # Ensures that x can be used inside of this function
    color(light_color[round(random()+0.3)])
    penup()
    home()
    goto(x, y)
    pendown()
    begin_fill()
    for i in range(4):
        forward(40)
        left(90)
    end_fill()
    x = x + 70 

hideturtle()
delay(0)
temp = 1
while temp<2:
    temp += 1
    y=30
    for i in range(4): # This loop will draw 4 rows of windows
        for i in range(4): # This loop will draw one row of 4 windows
            window()
        x = -125 # Ensures all rows of windows start from the same x-position
        y = y - 85 # Moves the next row of windows down lower than the previous
# fill other landscapes
# draw moon
width,height = 800,200
penup()
home()
delay(0)
setx(width/2-150)
sety(height/2+50)
pendown()
color("red","yellow")
hideturtle()
begin_fill()
circle(50,180)
right(30)
circle(58,-120)
end_fill()

# under blinking stars
t = Turtle(visible = False,shape='circle')
t.pencolor("white")
t.fillcolor("white")
t.penup()
t.setheading(-90)
t.goto(width/2,randint(-height/2,height/2))
stars = []
for i in range(50):
    star = t.clone()
    s =random() /3
    star.shapesize(s,s)
    star.speed(int(s*30))
    star.setx(width/2 + randint(1,width))
    star.sety( randint(height-70,height+100))
    star.showturtle()
    stars.append(star)
while True:
    y=30
    for i in range(randint(1,4)): # This loop will draw 4 rows of windows
        for i in range(randint(1,4)): # This loop will draw one row of 4 windows
            window()
        x = -125 # Ensures all rows of windows start from the same x-position
        y = y - 85 # Moves the next row of windows down lower than the previous
    for star in stars:
        star.setx(star.xcor() - 3 * star.speed())
        if star.xcor()<-width/2:
            star.hideturtle()
            star.setx(width/2 + randint(1,width))
            star.sety( randint(height-70,height+100))
            star.showturtle()