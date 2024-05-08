import turtle as t
from turtle import *
import random as r
import time

screensize(bg='black')
t.penup()
t.goto(-20, -300) # 定位
t.color("red")
t.write("Merry Christmas，姓名！", align="center", font=("WenQuanYi Zen Hei", 30, "bold")) #改这个地方就行，其他的不建议改

t.goto(0,0)
t.pendown()
n = 100.0
t.pensize(6.5)
speed("fastest")
left(90)
forward(3 * n)
color("orange", "yellow")
begin_fill()
left(126)
#画五角星
for i in range(5):
    forward(n / 5)
    right(144)
    forward(n / 5)
    left(72)
end_fill()
right(126)

 #彩灯
def drawlight():
    if r.randint(0, 80) == 0:
        color('tomato')
        circle(2)
    elif r.randint(0, 80) == 1:
        color('orange')
        circle(3)
    elif r.randint(0, 80) == 2  :
        color('red')
        circle(4)
    else:
        linewidth = 5
        color('dark green')
 
 
color("dark green")
backward(n * 4.8)
 
 #画树
def tree(d, s):
    if d <= 0: return
    forward(s)
    tree(d - 1, s * .8)
    right(120)
    tree(d - 3, s * .5)
    drawlight()
    right(120)
    tree(d - 3, s * .5)
    right(120)
    backward(s)
 
 
tree(15, n)
backward(n / 2)

 #底部装饰
for i in range(40):
    a = 200 - 400 * r.random()
    b = 10 - 20 * r.random()
    up()
    forward(b)
    left(90)
    forward(a)
    down()
    if r.randint(0, 1) == 0:
        color('white')
    else:
        color('pink')
    circle(2)
    up()
    backward(a)
    right(90)
    backward(b)
 
 #画雪
def drawsnow():
    t.ht()
    t.pensize(2)
    for i in range(200):
        t.pencolor("white")
        t.pu()  # pu=penup
        t.setx(r.randint(-350, 350))
        t.sety(r.randint(-100, 350))
        t.pd()  # pd=pendown
        dens = 6
        snowsize = r.randint(2, 10)
        for j in range(dens):
            # t.forward(int(snowsize))
            t.fd(int(snowsize))
            t.backward(int(snowsize))
            t.right(int(360 / dens))

#画星星
def drawstar():
    t.ht()
    t.pensize(2)
    for i in range(30):
        t.pencolor("yellow")
        t.pu()  # pu=penup
        t.setx(r.randint(-350, 350))
        t.sety(r.randint(150, 350))
        t.pd()  # pd=pendown
        dens = 4
        starsize = r.randint(2, 6)
        for j in range(dens):
            if j %2 == 1:
                t.fd(int(starsize))
                t.backward(int(starsize))
                t.right(int(360 / dens))
            else:
                t.fd(int(starsize*2))
                t.backward(int(starsize*2))
                t.right(int(360 / dens))

#调用函数 结束
drawstar()
drawsnow()

t.done()  # 完成