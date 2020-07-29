import os, configparser, ast, time, datetime
from datetime import datetime, timedelta

def ReadConfig(sectionName:str, valueName:str):
	settingsFile = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) + '/config.ini'
	r = None
	if os.path.isfile(settingsFile):
		config = configparser.ConfigParser()
		config.read(settingsFile)
		try: 
			r = ast.literal_eval(config.get(sectionName, valueName))
		except Exception as e:
			print('Unable to read value ', valueName, ' from settings file.')
			print(e)
	return r

def ReadConfigBool(sectionName:str, valueName:str):
	r = ReadConfig(sectionName, valueName)
	if r == None: r = False
	r = bool(r)
	return r

def ReadConfigString(sectionName:str, valueName:str):
	r = ReadConfig(sectionName, valueName)
	if not r == None: r = str(r)
	return r

def ReadConfigList(sectionName:str, valueName:str):
	r = ReadConfig(sectionName, valueName)
	if not r == None: r = list(r)
	return r

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r
	
def FileExists(f): return 	os.path.isfile(f)

def GetMyDateFormat(): return '%m/%d/%Y'

def ToDate(givenDate):
#returns datetime, converting from string if necessary
	if type(givenDate) == str:
		if givenDate.find('-') > 0 :
			r = datetime.strptime(givenDate, '%Y-%m-%d')
		else:
			r = datetime.strptime(givenDate, GetMyDateFormat())
	else:
		r = givenDate
	return r

def ToDateTime(givenDate):
	#returns datetime object
	r = datetime.combine(ToDate(givenDate), datetime.min.time())
	return r

def DateFormatDatabase(givenDate):
	#returns datetime object
	r = datetime.combine(ToDate(givenDate), datetime.min.time())
	return r

def GetDateTimeStamp():
	d = datetime.now()
	return d.strftime('%Y%m%d%H%M')

def GetTodaysDate():
	d = datetime.now()
	#return d.strftime('%m/%d/%Y')
	return d.date()

def DateDiffDays(startDate:datetime, endDate:datetime):
	delta = endDate-startDate
	return delta.days
		
def DateDiffHours(startDate:datetime, endDate:datetime):
	delta = endDate-startDate
	return int(delta.total_seconds() / 3600)

def AddDays(startDate, days:int):
	startDate = ToDate(startDate)
	return startDate + timedelta(days=days) 

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r
	
	
