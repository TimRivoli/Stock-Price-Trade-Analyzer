import os, configparser, ast

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