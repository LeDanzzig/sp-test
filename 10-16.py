# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:25:23 2020

@author: danzz
"""

#i=0#while hex(i) ends 401 print i


#i = 0
#while i < 10000:
   # i += 401
 
#print(i) # 10



#input ()
#dlina = input ("введите число ")
#dlina = int(dlina)
#rez = "%0.3X" % int(dlina)
#print (rez)


konec = input ("введите число ")
our_number = "_" #инициализируем строку
i_number = "_" #инициализируем еще одну строку
i_number16 = "_" #инициализируем еще одну строку
leng = 0 #инициализируем длинну
i = 0 #инициализируем стартовое число
result = 0 #инициализируем результат число
our_number = str(konec) #конвертируем число в строку
leng = len(our_number) #узнаем длинну нашей строки
while i <= 1000000000:  #1000000 в данном случае ограничение цикла по поиску числа. если все до милллиона не подходит - тогда выходим, что бы не получился бесконечный цикл
  i_number = str(i) #конвертируем текущее число i в строку и запоминаем текущую строку в переменную i_number
  if i_number.find(our_number, len(i_number) - leng, len(i_number))>0: #если в строке i_number содержится подстрока our_number то
    i_number16 = "%0.3X" % int(i)  #переводим текущее число i  в шестнадцатиричную строку и сохраняем её в i_number16
    if i_number16.find(our_number, len(i_number16) - leng, len(i_number16))>0: #если в строке i_number16 содержится подстрока our_number то это то, что мы искали
      result = i
      break
    else: 
      i += 1
      continue
  else: 
    i += 1
    continue
if result != 0:
    print (result)
else:
	print ("Ответа до 1000000000 нет")