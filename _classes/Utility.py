import os, configparser, ast, sys
import numpy as np, pandas as pd
import urllib.error, urllib.request as webRequest
from datetime import datetime, timedelta, date

def get_git_commit():
	import subprocess
	try:
		return subprocess.check_output(
			['git', 'rev-parse', 'HEAD'],
			stderr=subprocess.DEVNULL
		).decode('utf-8').strip()
	except Exception:
		return None

def get_env_versions(include_machine_learning:bool = False):
	if include_machine_learning:
		import keras
		return {
			'python': sys.version.split()[0],
			'numpy': np.__version__,
			'pandas': pd.__version__,
			'keras': keras.__version__
			}
	else:
		return {
			'python': sys.version.split()[0],
			'numpy': np.__version__,
			'pandas': pd.__version__
			}
	

def ReadConfig(section_name:str, value_name:str, verbose: bool = False):
	settingsFile = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) + '/config.ini'
	r = None
	if os.path.isfile(settingsFile):
		config = configparser.ConfigParser()
		config.read(settingsFile)
		if section_name in config:
			try: 
				r = config.get(section_name, value_name)
			except Exception as e:
				if verbose: print('Unable to read value ', value_name, ' from settings file.')
				if verbose: print(e)
	if r==None: WriteConfig(section_name, value_name, '')
	return r

def WriteConfig(section_name:str, value_name:str, value:str, verbose: bool = False):
	settingsFile = os.path.join(os.path.abspath(os.path.dirname(os.path.dirname(__file__)))) + '/config.ini'
	config = configparser.ConfigParser()
	if os.path.isfile(settingsFile): config.read(settingsFile)
	if section_name not in config: config.add_section(section_name)
	config.set(section_name, value_name, str(value))
	try: 
		with open(settingsFile, 'w') as configfile: config.write(configfile)
	except Exception as e:
		if verbose: print('Unable to write value ', value_name, ' from settings file.')
		if verbose: print(e)

def ReadConfigBool(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	if r == None: 
		r = False
	elif len(r) > 0:
		r = ast.literal_eval(r)
	return bool(r)

def ReadConfigInt(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	if r == None: 
		r = 0
	elif len(r) > 0: 
		r = ast.literal_eval(r)
	return int(r)

def ReadConfigString(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	if r == None: 
		r = ''
	elif len(r) > 0: 
		if "'" in r or '"' in r: r = ast.literal_eval(r)
	return str(r)

def ReadConfigList(section_name:str, value_name:str):
	r = ReadConfig(section_name, value_name)
	if r == None: 
		r = []
	elif len(r) > 0: 
		r = ast.literal_eval(r)
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

def ToDate(given_date):
#returns date object, converting from string or datetime if necessary
	if type(given_date) == str:
		if given_date.find('-') > 0 :
			r = datetime.strptime(given_date, '%Y-%m-%d').date()
		else:
			r = datetime.strptime(given_date, GetMyDateFormat()).date()
	elif isinstance(given_date, datetime):
		r = given_date.date()
	elif isinstance(given_date, np.datetime64):
		r = pd.Timestamp(given_date).date()
	else:
		r = given_date
	return r

def ToDateTime(given_date):
	#returns datetime object
	if type(given_date) == str: given_date = ToDate(given_date)
	if isinstance(given_date, datetime):
		r = given_date
	elif isinstance(given_date, np.datetime64):
		r = pd.Timestamp(given_date).date()
	elif isinstance(given_date, date): 
		r = datetime.combine(given_date, datetime.min.time())
	else:
		r = given_date
	return r

def ToTimestamp(given_date):
	"""
	Returns a pandas.Timestamp, safely converting from
	str, date, datetime, np.datetime64, or Timestamp.
	"""
	if given_date is None:
		return None

	if isinstance(given_date, pd.Timestamp):
		return given_date

	if isinstance(given_date, np.datetime64):
		return pd.Timestamp(given_date)

	if isinstance(given_date, datetime):
		return pd.Timestamp(given_date)

	if isinstance(given_date, date):
		return pd.Timestamp(datetime.combine(given_date, datetime.min.time()))

	if isinstance(given_date, str):
		# pandas handles almost all formats safely
		return pd.to_datetime(given_date)

	# Last-resort: assume caller knows what they’re doing
	return pd.Timestamp(given_date)

def DateFormatDatabase(given_date):
	#returns datetime object, technically should be np.datetime64
	r = ToDateTime(given_date)
	return r

def GetDateTimeStamp():
	d = datetime.now()
	return d.strftime('%Y%m%d%H%M')

def GetLatestBDay():
	d = datetime.now()
	return d.date()

def GetLatestBDay():
	d = pd.offsets.BDay().rollback(datetime.now())
	return d.date()
	
def GetTodaysDateString():
	d = datetime.now() #--1980-01-01
	return d.strftime('%Y-%m-%d')

def DateDiffDays(start_date:datetime, end_date:datetime):
	delta = end_date-start_date
	return delta.days
		
def DateDiffHours(start_date:datetime, end_date:datetime):
	delta = end_date-start_date
	return int(delta.total_seconds() / 3600)

def AddDays(start_date, days:int):
	#This will return a timestamp object
	start_date = ToDate(start_date)
	return start_date + timedelta(days=days) 

def CreateFolder(p:str):
	r = True
	if not os.path.exists(p):
		try:
			os.mkdir(p)	
		except Exception as e:
			print('Unable to create folder: ' + p)
			f = False
	return r

def ListToString(l:list):
	r = "['"
	for i in l:
		r += i + "','"
	r += "']"
	r = r.replace(",''","")
	return r

def PandaIsInIndex(df:pd.DataFrame, value):
	try:
		x = df.loc[value]
		r = True
	except:
		r = False
	return r
