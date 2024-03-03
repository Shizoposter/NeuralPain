# -*- coding: utf-8 -*-
# -*- coding: cp1251 -*-
#!/usr/bin/python

from tkinter import *
import os.path
import os
import cv2
import pytesseract
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf 
import pylab 
import random
from PIL import ImageGrab, Image
import subprocess
def NumberClassCreate():
	NumberGuessWindow = Toplevel()
	NumberGuessWindow.focus_set()
	NumberGuessWindow['bg'] = '#808080'
	NumberGuessWindow.geometry('500x500')
	NumberGuessWindow.resizable(False, False)
	enterPhotoName_Label = Label(
		NumberGuessWindow,
		text = 'Введите полное название файла',
		font = ('Candara', 17),
		fg = 'black',
		bg = '#808080'
	).place(x = 50, y = 150)
	global enterPhotoName_Entry
	enterPhotoName_Entry = Entry(
		NumberGuessWindow,	
	 	width = 15,
	  	bd = 2,
	   	bg = '#b1b5b1',
	    font = ('Candara', 20))
	enterPhotoName_Entry.place(x = 390, y = 147)
	enterPhotoName_Button = Button(
		NumberGuessWindow,
		width = 10,
		height = 1,
		font = ('Candara', 14),
		text = 'Enter',
		bd = 2,
		bg = '#666666',
		activebackground = '#333333',
		activeforeground = '#b3b3b3').place(x = 560, y = 150)

def Analysis():
	for key, value in _coords_and_idDICT.items():
		if value == (x, y):
			global _buttonID
			global _firstButton_active, _secondButton_active, _moveCount, _openedCardsCount
			_buttonID = key
			(ctypes.cast(_buttonID, ctypes.py_object).value)['state'] = 'disabled'
			if len(_coords_and_idDICT) > 0:
				if _firstButton_active == False and _secondButton_active == False:
					_firstButton_active = True
					_twoCardsDICT[_buttonID] = ((ctypes.cast(_buttonID, ctypes.py_object).value))['text']
				elif _firstButton_active == True and _secondButton_active == False:
					if _openedCardsCount < diff - 1:	
						_secondButton_active = True
						_twoCardsDICT[_buttonID] = ((ctypes.cast(_buttonID, ctypes.py_object).value))['text']
						_moveCount += 1
						for key, value in _twoCardsDICT.items():
							_twoCardsMASSIVE.append(key)
							_twoCardsValues.append(value)
						if _twoCardsValues[0] == _twoCardsValues[1]:
							for i in _twoCardsMASSIVE:
								del _coords_and_idDICT[i]
								((ctypes.cast(i, ctypes.py_object).value))['state'] = 'disabled'
							for i in _coords_and_idDICT:
								((ctypes.cast(i, ctypes.py_object).value))['state'] = 'normal'
							_openedCardsCount += 1

def Canvas_paint(event):
	color = 'black'
	x1, y1 = (event.x-3), (event.y-3)
	x2, y2 = (event.x+4), (event.y+4)
	global new_window_canvas
	new_window_canvas.create_oval(x1,y1,x2,y2,fill = color, outline = color)

def canvas_save():
	file_name = '1.png'
	new_window_canvas.postscript(file = file_name + '.eps')
				img = Image.open(file_name + '.eps') 
				img.save(file_name + '.png', 'png')
	x=root.winfo_rootx()+root.winfo_x()
	y=root.winfo_rooty()+root.winfo_y()
	x1=x+root.winfo_width()
	y1=y+root.winfo_height()
	ImageGrab.grab().crop((x,y,x1,y1)).save(f'C://Users//yotah//OneDrive//Рабочий стол//project//stuff//{file_name}')

	ImageGrab.grab(bbox=(
		new_window_canvas.winfo_rootx(),
		new_window_canvas.winfo_rooty(),
		new_window_canvas.winfo_rootx() + new_window_canvas.winfo_width(),
		new_window_canvas.winfo_rooty() + new_window_canvas.winfo_height())).save(file_name)

	foo = Image.open('1.png')
	foo = foo.resize((28,28))
	foo.save('2.png', quality=150	)

	img_1 = cv2.imread('1.png')
	img2_1 = cv2.resize(img_1, None, fx=9, fy=9)  # Увеличение изображения в 9 раз
	balance = pytesseract.image_to_string(img2_1, config='outputbase digits')
	print(balance)

	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = tf.keras.utils.normalize(x_train, axis = 1)
	x_test = tf.keras.utils.normalize(x_test, axis = 1)

	model = tf.keras.models.load_model('handwritten.model')

	try:
					img = cv2.imread(f'2.png')[:,:,0]
					img = np.invert(np.array([img]))
					prediction = model.predict(img)
					print(f'Наверное, это цифра {np.argmax(prediction)}')
					plt.imshow(img[0], cmap = plt.cm.binary)
					pylab.show()
				#except:
				#	print('Error!')
				finally:
					os.remove('2.png')
					os.remove('1.png')

def canv_create():
	new_window = Toplevel()
	new_window.focus_set()
	new_window['bg'] = '#808080'
	new_window.geometry('500x500')
	new_window.resizable(False, False)
	global new_window_canvas
	new_window_canvas = Canvas(new_window, width=500, height=500, bg='white', bd = 5)
	new_window_canvas.pack(expand = YES, fill = BOTH)
	new_window_canvas.bind('<B1-Motion>', Canvas_paint)


def GuessTheNumber():
	if os.path.isfile(enterPhotoName_Entry.get()):
		img = cv2.imread('screen.png')
		img2 = cv2.resize(img, None, fx=9, fy=9)  # Увеличение изображения в 9 раз
		balance = pytesseract.image_to_string(img2, config='outputbase digits')
		print(balance)
	
	root.geometry('300x400')
	canvas_button = Button(root, width = 20, height = 2, bg = '#666666', text = 'Готово.', font = ('Candara', 18), activeforeground = '#b3b3b3', activebackground = '#333333', command = canvas_save).place(x = 10, y = 10)
	reply_button = Button(root, width = 20, height = 2, bg = '#666666', text = 'Заново.', font = ('Candara', 18), activeforeground = '#b3b3b3', activebackground = '#333333', command = canv_create).place(x = 10, y = 50)

	mnist = tf.keras.datasets.mnist
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	x_train = tf.keras.utils.normalize(x_train, axis = 1)
	x_test = tf.keras.utils.normalize(x_test, axis = 1)
	model = tf.keras.models.Sequential()
	model.add(tf.keras.layers.Flatten(input_shape = (28,28)))
	model.add(tf.keras.layers.Dense(128, activation = 'relu'))
	model.add(tf.keras.layers.Dense(128, activation = 'relu'))
	model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

	model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

	model.fit(x_train, y_train, epochs = 20)
	model.save('handwritten.model')
	
	
	

	canv_create()
	

	#canvas_button = Button(new_window, width = 15, height = 1, bg = '#666666', text = 'Готово.', font = ('Candara', 16), activeforeground = '#b3b3b3', activebackground = '#333333', command = canvas_save).place(x = 200, y = 430)
	image_number = 1
				while os.path.isfile(f'digits/digit{image_number}.png'):
					try:
						img = cv2.imread(f'digits/digit{image_number}.png')[:,:,0]
						img = np.invert(np.array([img]))
						prediction = model.predict(img)
						print(f'Наверное, это цифра {np.argmax(prediction)}')
						plt.imshow(img[0], cmap = plt.cm.binary)
						pylab.show()
					except:
						print('Error!')
					finally:
						image_number += 1


root = Tk()
root.geometry('300x100')
root.title('NeuralWeb - Yota')
root.resizable(False, False)
root['bg'] = '#808080'
button_color = '#666666'
button_click_color = '#333333'
button_click_color_text = '#b3b3b3'


ms_button_activate1 = Button(
	root,
	width = 20,
	height = 2,
	bg = button_color,
	fg = 'black',
	bd = 2,
	text = 'Определение числа.',
	font = ('Candara', 18),
	activeforeground = button_click_color_text,
	activebackground = button_click_color,
	command = GuessTheNumber).place(x = 10, y = 10)

root.mainloop()

